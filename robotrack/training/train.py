"""Train RoboTracker: fine-tune AllTracker on a mix of Kubric and RoboTrack-15M.

    torchrun --nproc_per_node=8 -m robotrack.training.train \
        kubric_root=/data/alltracker robotrack15m_root=/data/RoboTrack-15M exp_name=robotracker

Settings live in robotrack/training/config/robotracker.yaml (Hydra; override any of them on the command line). On 4
GPUs, add distributed.virtual_world_size=8 to reproduce the 8-GPU run exactly (gradient accumulation over virtual
ranks). A run resumes automatically from <exp_dir>/checkpoints/latest.pth.
"""
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from robotrack.training.trainer import create_trainer


@hydra.main(config_path="config", config_name="robotracker", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, force=True,
                        format=f"[rank {os.environ.get('RANK', '0')}] [%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    if int(os.environ.get("RANK", 0)) == 0:
        logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    trainer = create_trainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
