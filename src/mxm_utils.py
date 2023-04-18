import typing as t

import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig


def only_dino(cfg: DictConfig) -> bool:
    """Whether we only use the DINO backbone, without the STEGO segmentation head."""
    return "only_dino" in cfg and cfg.only_dino


def do_dimred(cfg: DictConfig) -> bool:
    """Whether we perform dimension reduction with PCA, HNNE, RP etc on the DINO output feats."""
    return "dimred_type" in cfg and cfg.dimred_type != None


def dimred_in_forward_pass(cfg: DictConfig) -> bool:
    """
    For PCA and RP, we can downproject the features in the forward pass of the model, since those operations are
    quite lightweight. However, for UMAP and HNNE, we load the cached and down-projected features, so we do not
    need to do this in the forward pass.
    By first checking if "dimred_type" is in the config, we stay compatible with the legacy configs.
    """
    return "dimred_type" in cfg and cfg.dimred_type in ["PCA", "RP"]


def safe_wandb_log(item: t.Any) -> None:
    if wandb.run is None:
        return
    wandb.log(item)


def setup_wandb(cfg: DictConfig) -> None:
    """
    Sets up wandb logging and saves the config to wandb.
    Make sure that the keys "wandb_dotenv_path" and "run_name" are contained in the config.

    Args:
        cfg (DictConfig): Hydra config.
    """
    if cfg.log_type != "wandb":
        return
    load_dotenv(cfg.wandb_dotenv_path)
    wandb.init(config=cfg, name=cfg.run_name, settings=wandb.Settings(code_dir="."))
