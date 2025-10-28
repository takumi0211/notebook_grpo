"""Reusable GRPO training utilities for GPT-OSS 20B finetuning with Unsloth."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import IterableDataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import PreTrainedTokenizerBase

from data_reward import load_prompt_dataset, reward_fn

MODEL_ID = "unsloth/gpt-oss-20b"
OUT = "runs/grpo_gptoss20b_lora4_tes"

TOTAL_STEPS = 100
PROMPTS_PER_STEP = 12
NUM_GENERATIONS = 8
MAX_PROMPT_LEN = 1000
MAX_COMPLETION_LEN = 4000
SEED = 42

# Disable Accelerate's batch dispatching so IterableDataset samples containing strings
# (our raw prompts) are not concatenated across processes, preventing TypeError.
ACCELERATOR_CONFIG = {"dispatch_batches": False}

MAX_SEQ_LENGTH = MAX_PROMPT_LEN + MAX_COMPLETION_LEN + 16
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
LOAD_IN_4BIT = True


@dataclass
class TrainingArtifacts:
    """Bundle components needed to run and resume GRPO training."""

    trainer: GRPOTrainer
    tokenizer: PreTrainedTokenizerBase


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StepStream(IterableDataset):
    """Yield exactly *k* prompts per trainer step with reward columns."""

    KEEP_KEYS = {
        "prompt",
        "reward_action_0",
        "reward_action_1",
        "reward_action_2",
        "reward_action_3",
    }

    def __init__(self, base_dataset, k: int) -> None:
        super().__init__()
        self.base = base_dataset
        self.k = k
        self.n = len(base_dataset)
        self.keys = [key for key in self.KEEP_KEYS if key in getattr(base_dataset, "features", {})]

    def __iter__(self):  # type: ignore[override]
        while True:
            for idx in random.sample(range(self.n), self.k):
                row = self.base[idx]
                sample = {}
                for key in self.keys:
                    value = row[key]
                    if key == "prompt":
                        sample[key] = value
                    else:
                        sample[key] = torch.atleast_1d(torch.tensor(value, dtype=torch.float32))
                yield sample


def create_step_stream(prompts_per_step: int = PROMPTS_PER_STEP, dataset=None) -> StepStream:
    base = dataset if dataset is not None else load_prompt_dataset()
    return StepStream(base, prompts_per_step)


def build_model_and_tokenizer(
    model_id: str = MODEL_ID,
    seed: int = SEED,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """Load GPT-OSS with Unsloth kernels and wrap it with LoRA adapters."""

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.config.use_cache = False
    return model, tokenizer


def build_trainer(
    output_dir: str = OUT,
    model_id: str = MODEL_ID,
    total_steps: int = TOTAL_STEPS,
    prompts_per_step: int = PROMPTS_PER_STEP,
    num_generations: int = NUM_GENERATIONS,
    max_prompt_len: int = MAX_PROMPT_LEN,
    max_completion_len: int = MAX_COMPLETION_LEN,
    seed: int = SEED,
    dataset=None,
    learning_rate: float = 5e-5,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> TrainingArtifacts:
    _set_seed(seed)
    if model is None or tokenizer is None:
        model, tokenizer = build_model_and_tokenizer(model_id=model_id, seed=seed)
    else:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = False
    stream = create_step_stream(prompts_per_step=prompts_per_step, dataset=dataset)

    args = GRPOConfig(
        output_dir=output_dir,
        max_steps=total_steps,
        learning_rate=learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        seed=seed,
        num_generations=num_generations,
        generation_batch_size=prompts_per_step * num_generations,
        per_device_train_batch_size=prompts_per_step,
        gradient_accumulation_steps=1,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        temperature=1.0,
        report_to=[],
        logging_steps=1,
        accelerator_config=ACCELERATOR_CONFIG,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=reward_fn,
        train_dataset=stream,
    )
    return TrainingArtifacts(trainer=trainer, tokenizer=tokenizer)


def run_training(artifacts: Optional[TrainingArtifacts] = None) -> TrainingArtifacts:
    if artifacts is None:
        artifacts = build_trainer()
    artifacts.trainer.train()
    artifacts.trainer.save_model(OUT)
    artifacts.tokenizer.save_pretrained(OUT)
    return artifacts


def main() -> None:
    run_training()
    print("✅ finished")


if __name__ == "__main__":
    main()
