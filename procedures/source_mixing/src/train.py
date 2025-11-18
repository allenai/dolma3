# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ai2-olmo-core @ git+https://github.com/allenai/olmo-core.git@main",
# ]
# ///

"""
Example Training script for training an OLMo 30M used in source mixing experimentation.

Launch this with torchrun:
    uv run torchrun --nproc-per-node=1 src/train.py run_name [OVERRIDES...]

Or for a dry run that prints the config:
    uv run src/train.py run_name --dry-run [OVERRIDES...]
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmupAndLinearDecay
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

SEQ_LENGTH = 2048
GLOBAL_BATCH_SIZE = 131_072
MAX_TOKENS = 2_910_233_600  # 2.9B tokens @ 5x(C)hinchilla
LR = 0.007276622186288963
SEED = 1337


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyFSLDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = SEED
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder."""
    load_trainer_state: bool = False
    """Whether to load the trainer state when loading from `load_path`."""


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if (
        not trainer.no_checkpoints
        and not trainer.maybe_load_checkpoint()
        and config.load_path
    ):
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(
            config.load_path, load_trainer_state=config.load_trainer_state
        )

    trainer.fit()


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    save_folder = opts.save_folder
    if not save_folder:
        save_folder = f"/tmp/{opts.run_name}"

    work_dir = opts.work_dir
    if not work_dir:
        work_dir = "/tmp/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")

    model_config = factory(vocab_size=tokenizer_config.padded_vocab_size())

    # Load example source mixture configuration
    source_list = SourceMixtureList.from_yaml(
        os.path.join(os.path.dirname(__file__), "sources.yaml")
    )
    source_list.validate()

    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix=SourceMixtureDatasetConfig(
            source_list=source_list,
            requested_tokens=opts.max_tokens,
            global_batch_size=GLOBAL_BATCH_SIZE,
            processes=16,
            seed=SEED,
        ),
        tokenizer=tokenizer_config,
        work_dir=work_dir,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE, seed=SEED, num_workers=4
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH * 2,
        max_sequence_length=SEQ_LENGTH,
        optim=AdamWConfig(
            lr=LR,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmupAndLinearDecay(t_max=opts.max_tokens, warmup_steps=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.tokens(opts.max_tokens),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    )

    config = config.merge(overrides)

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        usage=f"python {sys.argv[0]} RUN_NAME [OPTIONS...] [CONFIG_OVERRIDES...]",
        description="Train a transformer language model with source mixing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="""The name of the run.""")
    parser.add_argument(
        "--model-factory",
        type=str,
        default="olmo2_30M",
        help="""The name of the model factory to use.
        This can be any classmethod on the TransformerConfig class.""",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="""The maximum number of tokens to train on.""",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        help="""A local or remote directory to save checkpoints to.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="""A local working directory for dataset preprocessing.
        Defaults to a temporary directory if not provided.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="""Print the config and exit.""",
    )
    opts, overrides = parser.parse_known_args()
    return opts, overrides


def main():
    opts, overrides = parse_args()
    config = build_config(opts, overrides)

    if opts.dry_run:
        rich.print(config)
        return

    prepare_training_environment()
    train(config)
    teardown_training_environment()


if __name__ == "__main__":
    main()
