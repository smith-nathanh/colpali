import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    PreTrainedModel,
    TrainerCallback,
    TrainingArguments,
)

from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)
from colpali_engine.trainer.contrastive_trainer import ContAccumTrainer
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    train_dataset: Union[ColPaliEngineDataset, List[ColPaliEngineDataset]]
    eval_dataset: Optional[ColPaliEngineDataset] = None
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = ColbertLoss()
    pretrained_peft_model_name_or_path: Optional[str] = None
    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
            self.tr_args.output_dir = self.output_dir

        if isinstance(self.tr_args.learning_rate, str):
            print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        # Check if model already has PEFT config
        model_has_peft = hasattr(self.model, "peft_config") and self.model.peft_config

        if self.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

        if self.peft_config is not None:
            if model_has_peft:
                print("Model already has PEFT config. Skipping new PEFT configuration to preserve existing setup.")
            else:
                print("Configurating PEFT model")
                self.model = get_peft_model(self.model, self.peft_config)

        # Enable training for all existing LoRA adapters
        if hasattr(self.model, "peft_config") and self.model.peft_config:
            print("Found existing LoRA adapters. Setting all adapters to trainable.")
            for adapter_name in self.model.peft_config.keys():
                # Set adapter to trainable
                if hasattr(self.model, "set_adapter_trainable"):
                    self.model.set_adapter_trainable(adapter_name, True)
                else:
                    # Fallback: manually set requires_grad for adapter parameters
                    for name, param in self.model.named_parameters():
                        if "lora_" in name.lower():
                            param.requires_grad = True

        # Always print trainable parameters at the end
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        else:
            # Fallback for non-PEFT models
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / all_params
            print(
                f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {trainable_percent:.4f}%"
            )

        print_gpu_utilization()


class CacheClearCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Clear cache at the beginning of each epoch"""
        print(f"[CALLBACK] on_epoch_begin called - epoch {state.epoch}")

        # Clear the trainer's cache
        cache_sizes_before = (
            # len(self.trainer.accumulated_queries),
            # len(self.trainer.accumulated_docs),
            len(self.trainer.accumulated_inputs) if hasattr(self.trainer, "accumulated_inputs") else 0,
        )
        # self.trainer.accumulated_queries.clear()
        # self.trainer.accumulated_docs.clear()
        self.trainer.accumulated_inputs.clear()

        print(f"[CALLBACK] Cache cleared at epoch {state.epoch}")
        print(
            f"[CALLBACK] Cache sizes before clear: inputs={cache_sizes_before}"
        )  # query={cache_sizes_before[0]}, doc={cache_sizes_before[1]},


class ColModelTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.train_dataset = self.config.train_dataset
        self.eval_dataset = self.config.eval_dataset
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
        )

    def train(self) -> None:
        trainer = ContAccumTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )
        # print which trainer is being used
        print(f"[SETUP] Using trainer: {type(trainer).__name__}")

        trainer.args.remove_unused_columns = False

        # Only add cache callback for ContAccumTrainer
        # if isinstance(trainer, ContAccumTrainer):
        #     cache_callback = CacheClearCallback(trainer)
        #     trainer.add_callback(cache_callback)
        #     print(f"[SETUP] Added cache clearing callback for {type(trainer).__name__}")

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
