from collections import deque

import torch
import torch.distributed as dist
from datasets import DatasetDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

from colpali_engine.data.sampler import SingleDatasetBatchSampler
from colpali_engine.loss.late_interaction_losses import ColbertLoss


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        if isinstance(kwargs["train_dataset"], DatasetDict):
            dataset_list = list(kwargs["train_dataset"].values())
        elif isinstance(kwargs["train_dataset"], list):
            dataset_list = kwargs["train_dataset"]
        else:
            dataset_list = None

        if dataset_list is not None:
            kwargs["train_dataset"] = ConcatDataset(dataset_list)

        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = dataset_list

    def get_train_dataloader(self):
        ######## adapted from Transformers Trainer (gross) ########
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.dataset_list is None:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            ######### don't set batch size, mutually exclusive from batch sampler ######
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            ###### batch_sampler set instead of sampler in trainer code #######
            dataloader_params["batch_sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return dataloader

    def _get_train_sampler(self):
        if self.dataset_list is None:
            return super()._get_train_sampler()

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return SingleDatasetBatchSampler(
            self.dataset_list,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
        if "neg_doc_input_ids" in inputs:
            neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
            loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        if "labels" in inputs:
            loss = self.loss_func(query_outputs, doc_outputs, inputs["labels"])
        else:
            loss = self.loss_func(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            if "neg_doc_input_ids" in inputs:
                neg_doc_outputs = model(**{k[8:]: v for k, v in inputs.items() if k.startswith("neg_doc")})
                loss = self.loss_func(query_outputs, doc_outputs, neg_doc_outputs)
                return loss, None, None

            if "labels" in inputs:
                loss = self.loss_func(query_outputs, doc_outputs, inputs["labels"])
            else:
                loss = self.loss_func(query_outputs, doc_outputs)
            return loss, None, None


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes with proper shape handling and synchronization.
    """

    @staticmethod
    def forward(ctx, input):
        if not dist.is_available() or not dist.is_initialized():
            return input

        # Store context for backward pass
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        ctx.original_shape = input.shape

        # Get each sequence length
        local_seq_len = torch.tensor(input.size(1), device=input.device, dtype=torch.long)

        # Use all_reduce to find the maximum sequence length
        dist.all_reduce(local_seq_len, op=dist.ReduceOp.MAX)
        max_seq_len = local_seq_len.item()

        # Store both original and padded sequence lengths
        ctx.original_seq_len = input.size(1)
        ctx.max_seq_len = max_seq_len

        # Pad to maximum sequence length
        if input.size(1) < max_seq_len:
            batch_size, seq_len, hidden_size = input.shape
            padding_size = max_seq_len - seq_len  # Find how much padding is needed
            padding = torch.zeros(
                batch_size,
                padding_size,
                hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            input_padded = torch.cat([input, padding], dim=1)
        else:
            input_padded = input

        # Gather tensors from all processes
        gathered_tensors = [torch.zeros_like(input_padded) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, input_padded)

        # Concatenate along batch dimension
        result = torch.cat(gathered_tensors, dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        if not dist.is_available() or not dist.is_initialized():
            return grad_output

        # Split gradient back to individual ranks
        batch_size = ctx.original_shape[0]

        start_idx = ctx.rank * batch_size
        end_idx = start_idx + batch_size

        # Extract this rank's gradient with the PADDED sequence length first
        grad_slice_padded = grad_output[start_idx:end_idx, : ctx.max_seq_len, :]

        # Then trim to original sequence length to match input shape
        grad_slice = grad_slice_padded[:, : ctx.original_seq_len, :]

        return grad_slice


class DistContrastiveTrainer(ContrastiveTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        # feed only kwargs with 'doc_' prefix
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})

        # Gather embeddings from all devices if using distributed training
        if self.accelerator.num_processes > 1:
            if dist.is_initialized():
                query_outputs = GatherLayer.apply(query_outputs)
                doc_outputs = GatherLayer.apply(doc_outputs)
            else:
                print("WARNING: torch.distributed not initialized, skipping gather")

        loss = self.loss_func(query_outputs, doc_outputs)

        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss


class ContAccumTrainer(ContrastiveTrainer):
    def __init__(self, cache_size=7, **kwargs):
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self.query_cache = deque(maxlen=self.cache_size)
        self.doc_cache = deque(maxlen=self.cache_size)

    def _pad_to_max_length(self, tensors):
        # Find maximum sequence length
        max_seq_len = max(tensor.size(1) for tensor in tensors)

        padded_tensors = []
        for tensor in tensors:
            if tensor.size(1) < max_seq_len:
                padding = torch.zeros(
                    tensor.size(0),
                    max_seq_len - tensor.size(1),
                    tensor.size(2),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                padded_tensor = torch.cat([tensor, padding], dim=1)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)

        return padded_tensors

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """Override to ensure clean evaluation without cache contamination."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])

            eval_loss_func = ColbertLoss(temperature=0.02)
            loss = eval_loss_func(query_outputs, doc_outputs)

            return loss, None, None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})

        num_current_queries = query_outputs.size(0)

        # Concat the embeddings with the cache
        if len(self.query_cache) > 0:
            all_query_tensors = [query_outputs] + [q.detach() for q in self.query_cache]
            padded_query_tensors = self._pad_to_max_length(all_query_tensors)
            query_with_cache = torch.cat(padded_query_tensors, dim=0).to(query_outputs.device)
        else:
            query_with_cache = query_outputs

        if len(self.doc_cache) > 0:
            all_doc_tensors = [doc_outputs] + [d.detach() for d in self.doc_cache]
            padded_doc_tensors = self._pad_to_max_length(all_doc_tensors)
            doc_with_cache = torch.cat(padded_doc_tensors, dim=0).to(doc_outputs.device)
        else:
            doc_with_cache = doc_outputs

        loss = self.loss_func(query_with_cache, doc_with_cache, num_current_queries)

        # Cache the embedding outputs for future use
        self.query_cache.append(query_outputs.detach().clone())
        self.doc_cache.append(doc_outputs.detach().clone())

        return (loss, (query_with_cache, doc_with_cache)) if return_outputs else loss
