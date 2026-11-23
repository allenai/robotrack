"""
Optional Weights & Biases logger.

Wraps wandb so the trainer can call .log_scalar / .log_dict / .finish whether
or not wandb is enabled. Only the main rank should construct one (the trainer
already gates this by rank).
"""

import logging
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class WandbLogger:
    """Thin wrapper around wandb. Falls back to a no-op if wandb is disabled."""

    def __init__(self, cfg: DictConfig, run_config: Optional[DictConfig] = None):
        self.enabled = bool(cfg.get("enabled", False))
        self._run = None
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError:
            logging.warning(
                "wandb not installed, install with `pip install wandb`. Disabling WandbLogger."
            )
            self.enabled = False
            return

        self._wandb = wandb
        config_dict = (
            OmegaConf.to_container(run_config, resolve=True)
            if run_config is not None
            else None
        )
        self._run = wandb.init(
            project=cfg.get("project", None),
            entity=cfg.get("entity", None),
            name=cfg.get("name", None),
            tags=list(cfg.get("tags", [])) or None,
            notes=cfg.get("notes", None),
            group=cfg.get("group", None),
            mode=cfg.get("mode", "online"),
            dir=cfg.get("dir", None),
            id=cfg.get("id", None),
            resume=cfg.get("resume", "allow"),
            config=config_dict,
        )
        logging.info(
            f"WandB initialized: project={cfg.get('project')} "
            f"name={cfg.get('name')} mode={cfg.get('mode', 'online')}"
        )

    def log_scalar(self, name: str, value: Any, step: int):
        if not self.enabled:
            return
        self._wandb.log({name: value}, step=step)

    def log_dict(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        if not self.enabled:
            return
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self._wandb.log(metrics, step=step)

    def finish(self):
        if not self.enabled or self._run is None:
            return
        try:
            self._wandb.finish()
        except Exception as e:
            logging.warning(f"wandb.finish() failed: {e}")
