#!/usr/bin/env python3
"""Build a LlamaFactory tokenized cache without loading or training the model."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoProcessor, AutoTokenizer

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams.parser import _parse_train_args


def load_data_tokenizer_module(model_args):
    """Load only the tokenizer/processor needed by preprocessing.

    Importing ``llamafactory.model.load_tokenizer`` also imports the training
    model loader (and therefore optional TRL dependencies).  A tokenization
    probe should be runnable in a lightweight environment, so reproduce the
    small data-relevant portion of that loader here.
    """
    init_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        **init_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        **init_kwargs,
    )
    if "Processor" not in processor.__class__.__name__:
        processor = None
    if processor is not None:
        processor.tokenizer = tokenizer
        for name in (
            "image_max_pixels", "image_min_pixels", "image_do_pan_and_scan",
            "crop_to_patches", "video_max_pixels", "video_min_pixels",
            "video_fps", "video_maxlen", "use_audio_in_video",
            "audio_sampling_rate",
        ):
            setattr(processor, name, getattr(model_args, name))
    return {"tokenizer": tokenizer, "processor": processor}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenized-path", required=True)
    args = parser.parse_args()

    output = Path(args.tokenized_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Tokenized cache already exists: {output}")

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["dataset_dir"] = args.dataset_dir
    config["tokenized_path"] = args.tokenized_path
    # Preprocessing does not execute a model forward pass.  Disable the
    # accelerator dtype validation performed by TrainingArguments so this
    # utility can also run on a CPU-only login node.
    config["bf16"] = False
    config["fp16"] = False
    config.pop("deepspeed", None)
    # A deterministic one-process build also works in restricted login
    # environments where Python multiprocessing managers cannot bind sockets.
    config["preprocessing_num_workers"] = 1
    # The standalone process is deliberately not launched by torchrun/Ray.
    # Parse the same data/model arguments but avoid distributed-only checks.
    config.pop("ray_num_workers", None)
    config.pop("ray_init_kwargs", None)
    model_args, data_args, training_args, _, _ = _parse_train_args(config)
    tokenizer_module = load_data_tokenizer_module(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    dataset_module = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )
    print(f"Tokenized {len(dataset_module['train_dataset'])} rows into {output}")


if __name__ == "__main__":
    main()
