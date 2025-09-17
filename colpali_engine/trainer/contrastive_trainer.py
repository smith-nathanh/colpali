import torch
import torch.distributed as dist
from datasets import DatasetDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

from colpali_engine.data.sampler import SingleDatasetBatchSampler


class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        loss_func,
        is_vision_model,
        *args,
        retrieval_k: int = 5,
        compute_retrieval_metrics: bool = True,
        full_corpus_eval: bool = True,
        doc_block_size: int = 64,
        **kwargs
    ):
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
        self.retrieval_k = retrieval_k  # k for nDCG@k (default 5)
        self.compute_retrieval_metrics = compute_retrieval_metrics
        self.full_corpus_eval = full_corpus_eval
        self.doc_block_size = doc_block_size

    # ------------------------------------------------------------------
    # Diagnostics (no-op placeholder so base trainer doesn't error if
    # subclasses expect _log_diagnostics to exist)
    # ------------------------------------------------------------------
    def _log_diagnostics(self, *args, **kwargs):  # noqa: D401
        """Hook for diagnostic logging (implemented in ContAccumTrainer)."""
        return

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
            self._log_diagnostics(query_outputs, doc_outputs)
            return (loss, (query_outputs, doc_outputs, neg_doc_outputs)) if return_outputs else loss

        if "labels" in inputs:
            loss = self.loss_func(query_outputs, doc_outputs, inputs["labels"])
        else:
            loss = self.loss_func(query_outputs, doc_outputs)
        self._log_diagnostics(query_outputs, doc_outputs)
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

    # ------------------------------------------------------------------
    # Retrieval metrics (in-batch) – MRR and nDCG@k
    # ------------------------------------------------------------------
    @staticmethod
    def _colbert_scores(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute ColBERT MaxSim scores between all query/doc pairs in a batch.

        Args:
            query_embeddings: (B, Q, D)
            doc_embeddings: (B, P, D)
        Returns:
            scores: (B, B) where scores[i,j] is similarity of query i to doc j.
        """
        # einsum produces (B,C,Q,P); then max over doc tokens dim, sum over query tokens (ColBERT MaxSim)
        # Mirrors diagnostic computation but isolated here.
        scores = torch.einsum("bqd,csd->bcqs", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        return scores

    def _compute_batch_ranks(self, scores: torch.Tensor) -> torch.Tensor:
        """Given (B,B) score matrix with diagonal = positive pairs, return 1-based ranks for each query.

        Rank is computed by counting how many docs have strictly higher score.
        """
        # Get diagonal scores (positive pairs)
        pos_scores = scores.diagonal()  # (B,)

        # For each query, count how many documents have higher scores than the positive doc
        # pos_scores.unsqueeze(1) broadcasts to (B, 1), scores is (B, B)
        # This creates a boolean matrix where True means the score is higher than positive
        higher_scores = scores > pos_scores.unsqueeze(1)  # (B, B)

        # Count how many docs have higher scores (this gives 0-based rank)
        # Add 1 to get 1-based ranks
        ranks = higher_scores.sum(dim=1) + 1  # (B,)

        return ranks

    def _compute_retrieval_metrics(self, eval_dataloader) -> dict:
        model = self.model
        model.eval()
        k = self.retrieval_k
        device = next(model.parameters()).device
        mrr_sum = torch.zeros(1, device=device)
        ndcg_sum = torch.zeros(1, device=device)
        total_queries = torch.zeros(1, device=device)

        for batch in eval_dataloader:
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                query_embeddings = model(
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                )  # (B,Q,D)
                doc_embeddings = model(**{k2[4:]: v for k2, v in batch.items() if k2.startswith("doc")})  # (B,P,D)

                scores = self._colbert_scores(query_embeddings, doc_embeddings)  # (B,B)
                ranks = self._compute_batch_ranks(scores)  # (B,)
                ranks_f = ranks.float()
                # MRR contribution
                mrr_sum += (1.0 / ranks_f).sum()
                # nDCG@k (single relevant doc): contribution only if rank<=k
                ndcg_batch = torch.where(
                    ranks_f <= k,
                    1.0 / torch.log2(ranks_f + 1.0),
                    torch.zeros_like(ranks_f),
                ).sum()
                ndcg_sum += ndcg_batch
                # Recall@k (single relevant doc): 1 if rank<=k else 0
                # We do not return it here to keep output stable, but easy to add if needed
                total_queries += torch.tensor(ranks.size(0), device=ranks.device, dtype=torch.float)

        mrr = (mrr_sum / total_queries).item() if total_queries.item() > 0 else 0.0
        ndcg = (ndcg_sum / total_queries).item() if total_queries.item() > 0 else 0.0
        return {"eval_mrr": mrr, f"eval_ndcg@{k}": ndcg}

    def _compute_full_corpus_retrieval_metrics(self, eval_dataloader) -> dict:
        """Compute retrieval metrics by ranking each query against the entire eval doc pool.

        Two-pass approach over the eval dataloader to keep memory bounded:
        1) Encode and store all doc embeddings on CPU.
        2) For each query batch, score against doc embeddings in blocks, then compute ranks/metrics.
        """
        model = self.model
        model.eval()
        k = self.retrieval_k
        device = next(model.parameters()).device

        # First pass: collect doc embeddings for every example, in dataloader order
        doc_embeds: list[torch.Tensor] = []  # list of (P,D) per sample
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                docs = model(**{k2[4:]: v for k2, v in batch.items() if k2.startswith("doc")})  # (B,P,D)
                # Split along batch into per-sample tensors and move to CPU to save VRAM
                for i in range(docs.size(0)):
                    doc_embeds.append(docs[i].detach().cpu())

        num_docs = len(doc_embeds)
        if num_docs == 0:
            return {"eval_mrr": 0.0, f"eval_ndcg@{k}": 0.0}

        # Second pass: compute query embeddings and score vs all docs in blocks
        mrr_sum = 0.0
        ndcg_sum = 0.0
        recall_sum = 0.0
        total_queries = 0

        # Recreate dataloader to iterate queries in the same order
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        q_start_idx = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                queries = model(
                    input_ids=batch["query_input_ids"],
                    attention_mask=batch["query_attention_mask"],
                )  # (B,Q,D)
                B = queries.size(0)

                # Build scores against all docs blockwise to avoid OOM
                scores_full = torch.empty((B, num_docs), device=device, dtype=queries.dtype)

                # Iterate blocks of docs
                block = self.doc_block_size
                for d_start in range(0, num_docs, block):
                    d_end = min(d_start + block, num_docs)
                    block_list = doc_embeds[d_start:d_end]
                    # Pad docs in this block to same token length
                    max_p = max(t.size(0) for t in block_list)
                    padded = []
                    for t in block_list:
                        if t.size(0) < max_p:
                            pad = torch.zeros((max_p - t.size(0), t.size(1)), dtype=t.dtype)
                            t_pad = torch.cat([t, pad], dim=0)
                        else:
                            t_pad = t
                        padded.append(t_pad)
                    docs_block = torch.stack(padded, dim=0).to(device)  # (Db, P, D)

                    # Compute ColBERT MaxSim: (B,Q,D) x (Db,P,D) -> (B,Db)
                    # einsum -> (B,Db,Q,P); max over P; sum over Q
                    block_scores = (
                        torch.einsum("bqd,cpd->bcqp", queries, docs_block).max(dim=3)[0].sum(dim=2)
                    )  # (B,Db)
                    scores_full[:, d_start:d_end] = block_scores

                # Positive doc indices for these queries are their absolute indices in eval order
                pos_indices = torch.arange(q_start_idx, q_start_idx + B, device=device)
                # Extract positive scores
                pos_scores = scores_full[torch.arange(B, device=device), pos_indices]
                # Compute 1-based ranks: count how many docs score strictly higher
                ranks = (scores_full > pos_scores.unsqueeze(1)).sum(dim=1) + 1  # (B,)

                ranks_f = ranks.float()
                # Accumulate metrics
                mrr_sum += (1.0 / ranks_f).sum().item()
                ndcg_sum += torch.where(
                    ranks_f <= k, 1.0 / torch.log2(ranks_f + 1.0), torch.zeros_like(ranks_f)
                ).sum().item()
                recall_sum += (ranks_f <= k).float().sum().item()
                total_queries += B
                q_start_idx += B

        mrr = mrr_sum / max(1, total_queries)
        ndcg = ndcg_sum / max(1, total_queries)
        recallk = recall_sum / max(1, total_queries)
        return {"eval_mrr": mrr, f"eval_ndcg@{k}": ndcg, f"eval_recall@{k}": recallk}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Run standard HF evaluation to get loss etc.
        results = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # Compute retrieval metrics if eval dataset provided and enabled
        if (eval_dataset is None and self.eval_dataset is None) or not self.compute_retrieval_metrics:
            return results
        try:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            retrieval_metrics = (
                self._compute_full_corpus_retrieval_metrics(eval_dataloader)
                if self.full_corpus_eval
                else self._compute_retrieval_metrics(eval_dataloader)
            )
            results.update(retrieval_metrics)
            if retrieval_metrics:
                self.log(retrieval_metrics)
        except Exception as e:  # noqa: BLE001
            if self.args.local_rank in (-1, 0):
                print(f"[Retrieval metrics skipped] Error: {type(e).__name__}: {e}")
                # Add more specific debugging for the previous error
                import traceback

                print(f"[Retrieval metrics debug] Full traceback:\n{traceback.format_exc()}")
        return results


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
        self._log_diagnostics(query_outputs, doc_outputs)
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss


class ContAccumTrainer(ContrastiveTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accumulated_inputs = []
        self.accumulation_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_outputs = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})

        # During eval, compute loss normally without accumulation
        if not model.training:
            loss = self.loss_func(query_outputs, doc_outputs)
            self._log_diagnostics(query_outputs, doc_outputs)
            return (loss, (query_outputs, doc_outputs)) if return_outputs else loss

        self.accumulation_step += 1
        # For the final step, compute fresh embeddings for ALL accumulated batches
        if self.accumulation_step >= self.args.gradient_accumulation_steps:
            all_queries = []
            all_docs = []

            # Re-compute embeddings for all accumulated inputs
            for stored_inputs in self.accumulated_inputs:
                q_out = model(
                    input_ids=stored_inputs["query_input_ids"],
                    attention_mask=stored_inputs["query_attention_mask"],
                )
                d_out = model(**{k[4:]: v for k, v in stored_inputs.items() if k.startswith("doc")})
                all_queries.append(q_out)
                all_docs.append(d_out)

            # Add current batch
            all_queries.append(query_outputs)
            all_docs.append(doc_outputs)

            # Compute loss on all fresh embeddings
            combined_queries = self._pad_and_concat(all_queries)
            combined_docs = self._pad_and_concat(all_docs)

            loss = self.loss_func(combined_queries, combined_docs)
            self._log_diagnostics(combined_queries, combined_docs)

            # Clear accumulated data
            self.accumulated_inputs.clear()
            self.accumulation_step = 0

            return (loss, (combined_queries, combined_docs)) if return_outputs else loss
        else:
            # Store the inputs (not the embeddings) for later recomputation
            current_inputs = {
                "query_input_ids": inputs["query_input_ids"].clone(),
                "query_attention_mask": inputs["query_attention_mask"].clone(),
            }
            for k, v in inputs.items():
                if k.startswith("doc"):
                    current_inputs[k] = v.clone()

            self.accumulated_inputs.append(current_inputs)

            # Return zero loss for intermediate steps (no backward)
            return torch.tensor(0.0, device=query_outputs.device, requires_grad=True)

    def _pad_and_concat(self, tensor_list):
        """Pad tensors to same length and concatenate along batch dimension"""
        if len(tensor_list) == 1:
            return tensor_list[0]

        # Find maximum sequence length
        max_seq_len = max(tensor.size(1) for tensor in tensor_list)

        padded_tensors = []
        for tensor in tensor_list:
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

        return torch.cat(padded_tensors, dim=0)

    def _log_diagnostics(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor):
        """Compute and log diagnostic metrics to the Trainer logger (e.g., W&B) at logging steps.

        Metrics:
          - pos_score_mean / neg_score_mean / pos_neg_diff
          - score_mean / score_min / score_max
          - query_tok_norm_mean/std, doc_tok_norm_mean/std
          - frac_nonzero_query, frac_nonzero_doc (token utilization)
        Heavy ops are skipped unless we're at a logging step & in training.
        """
        # Only log on main process when global step matches logging interval
        if not hasattr(self, "state") or self.state.global_step is None:
            return
        if self.state.global_step % max(1, self.args.logging_steps) != 0:
            return
        if hasattr(self.args, "local_rank") and self.args.local_rank not in (-1, 0):
            return

        try:
            with torch.no_grad():
                # scores: (B,B) using ColBERT max-sim
                # einsum -> (B, B, Q, D); max over D; sum over Q
                scores = torch.einsum("bqd,csd->bcqs", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
                pos_scores = scores.diagonal()
                neg_masked = scores - torch.eye(scores.size(0), device=scores.device) * 1e6
                neg_scores = neg_masked.max(dim=1)[0]

                # Token norms
                q_norms = query_embeddings.norm(dim=-1)
                d_norms = doc_embeddings.norm(dim=-1)
                frac_nonzero_query = (q_norms > 0).float().mean()
                frac_nonzero_doc = (d_norms > 0).float().mean()

                # Intra-pair token usage (only within the same batch index)
                try:
                    pair_sims = torch.bmm(query_embeddings, doc_embeddings.transpose(1, 2))  # (B,Q,D)
                    # Active doc tokens selected by at least one query token (via max per query token)
                    max_doc_indices = pair_sims.argmax(dim=2)  # (B,Q)
                    # unique counts per batch element
                    active_counts = []
                    for b in range(max_doc_indices.size(0)):
                        active_counts.append(len(torch.unique(max_doc_indices[b])))
                    active_counts = torch.tensor(active_counts, device=query_embeddings.device, dtype=torch.float)
                    doc_active_frac = (active_counts / doc_embeddings.size(1)).mean().item()
                except RuntimeError:
                    doc_active_frac = float("nan")

                metrics = {
                    "diag/pos_score_mean": pos_scores.mean().item(),
                    "diag/neg_score_mean": neg_scores.mean().item(),
                    "diag/pos_neg_diff": (pos_scores - neg_scores).mean().item(),
                    "diag/score_mean": scores.mean().item(),
                    "diag/score_min": scores.min().item(),
                    "diag/score_max": scores.max().item(),
                    "diag/query_tok_norm_mean": q_norms.mean().item(),
                    "diag/query_tok_norm_std": q_norms.std(unbiased=False).item(),
                    "diag/doc_tok_norm_mean": d_norms.mean().item(),
                    "diag/doc_tok_norm_std": d_norms.std(unbiased=False).item(),
                    "diag/frac_nonzero_query": frac_nonzero_query.item(),
                    "diag/frac_nonzero_doc": frac_nonzero_doc.item(),
                    "diag/query_tokens_per_example": float(query_embeddings.size(1)),
                    "diag/doc_tokens_per_example": float(doc_embeddings.size(1)),
                    "diag/doc_active_frac": doc_active_frac,
                }
                self.log(metrics)
        except RuntimeError as e:
            # Fail silently to avoid interrupting training
            if self.args.local_rank in (-1, 0):
                print(f"[Diagnostics skipped] {e}")
