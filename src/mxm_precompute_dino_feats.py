"""
This script 
- pre-computes the DINO features for the train and val sets 
- saves them in a Squirrel dataset
- is based on the logic STEGO's original precompute_knns.py script
"""

import itertools
import os
import shutil
import typing as t
from enum import Enum

import hydra
import numpy as np
import psutil
import torch
from data import ContrastiveSegDataset
from modules import DinoFeaturizer, LambdaLayer
from mxm_utils import setup_wandb
from omegaconf import DictConfig
from pytorch_lightning.utilities.seed import seed_everything
from squirrel.driver import FileDriver, MessagepackDriver
from squirrel.iterstream import IterableSource
from squirrel.iterstream.torch_composables import SplitByWorker
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_transform

STATUS_FILE_NAME = "status.txt"
SHARD_SIZE = 500


class WriteStatus(Enum):
    WRITING = "WRITING"
    SUCCESS = "SUCCESS"


def write_status_file(directory: str, status: WriteStatus, status_file=STATUS_FILE_NAME) -> None:
    """
    Writes a status file to the given directory. Allows us to track whether a Squirrel dataset has been successfully written to disk.
    Background here is that if the run crashes (e.g. machines shuts down) or we are debugging our Squirrel dataloader a dataset may have been
    only partially written to disk, and a new iteration of the script will append to the existing shards, which can lead to
    duplicate data.
    """
    with FileDriver(f"{directory}/{status_file}").open(mode="w", create_if_not_exists=True) as f:
        f.write(status.value)


def status_is_success(directory: str, status_file=STATUS_FILE_NAME) -> bool:
    """
    Checks whether a dataset has been successfully written.
    """
    try:
        with FileDriver(f"{directory}/{status_file}").open(mode="r") as f:
            line = f.read()
    except FileNotFoundError:
        print(f"File {directory}/{status_file} not found.")
        return False

    return line == WriteStatus.SUCCESS.value


def get_only_dino_feats(
    loader: DataLoader,
    num_images_limit: int = None,
    mem_limit_percent: int = None,
) -> torch.Tensor:
    """
    Extracts the DINO features from a DataLoader and returns them as a single batched tensor. Note
    that for very large datasets (e.g. Cocostuff27 five-crop) we may run out of memory, so we added
    a mechanism that stops the extraction once the memory limit is reached. Since the training dataset
    is shuffled anyway, this is not a problem, and the validation datasets should all fit into memory.

    Args:
        loader (DataLoader): Dataloader yielding pre-computed dino features under key "dino_feats".
        num_images_limit (int, optional): If set, we only load DINO features from num_images_limit images,
            since sometimes we want to only load a fraction of the data, because we only need a subset of
            it anyway to fit e.g. PCA on the features. Defaults to None.
        mem_limit_percent (int, optional): If set, stops loading more images once a memory threshold has
            been reached. Expects percent, i.e. 70 not 0.7 for 70% system memory util. Defaults to None.

    Returns:
        torch.Tensor: A concatenation of all DINO features from the loader.
    """
    feats = []
    num_samples = 0
    for item in tqdm(loader):
        num_samples += item["dino_feats"].shape[0]
        if num_images_limit is not None and num_samples > num_images_limit:
            print(f"Number of samples limit of {num_images_limit} reached.")
            break

        if mem_limit_percent is not None and psutil.virtual_memory().percent > mem_limit_percent:
            print(f"Memory limit of {mem_limit_percent}% reached.")
            break

        feats.append(item["dino_feats"])
    return torch.cat(feats, dim=0)


def get_dino_feats_driver(cfg: DictConfig, image_set: str, prefix: str = "dino_feats") -> MessagepackDriver:
    """
    Returns MessagepackDriver over the DINO features for the given dataset.
    """
    crop_type = cfg.crop_type if image_set == "train" else None  # STEGO only uses five-crop for training
    res = cfg.res if image_set == "train" else cfg.val_res

    path = os.path.join(
        cfg.root_feat_store,
        "dino_feats",
        f"{prefix}_{cfg.model_type}_{cfg.dataset_name}_{image_set}_{crop_type}_{res}",
    )
    print(f"Creating MessagepackDriver at path {path}")
    return MessagepackDriver(path)


def _collate(batch: t.List[t.Dict[str, np.ndarray]]) -> t.Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.concatenate([item[k] for item in batch])) for k in batch[0].keys()}


def get_dataloader_from_driver(
    driver: MessagepackDriver,
    batch_size: int,
    num_workers: int,
    shuffle_key_buffer: int = 1,  # 1 means no shuffling
    shuffle_item_buffer: int = 1,  # 1 means no shuffling
) -> DataLoader:
    """Converts a squirrel driver into a pytorch dataloader."""

    it = (
        driver.get_iter(
            key_hooks=[SplitByWorker],
            shuffle_key_buffer=shuffle_key_buffer,
            shuffle_item_buffer=shuffle_item_buffer,
        )
        .batched(batch_size, _collate, drop_last_if_not_full=False)
        .to_torch_iterable()
    )
    return DataLoader(it, batch_size=None, num_workers=num_workers, pin_memory=True)


def get_dino_feats_loader(
    cfg: DictConfig,
    image_set: str,
    batch_size: int,
    num_workers: int,
    prefix: str = "dino_feats",
    shuffle_key_buffer: int = 1,  # 1 means no shuffling
    shuffle_item_buffer: int = 1,  # 1 means no shuffling
) -> DataLoader:
    """
    Returns a torch Dataloader over the DINO features for the given dataset. This is a drop-in replacement for
    Dataloader(ContrastiveSegDataset), which will directly yield the pre-computed DINO features from the Squirrel dataset.
    Note that the Dataloader also yields the images and the labels, so it can be used directly with the original code.
    I.e. the driver returns samples of the form {"img": ..., "label": ..., "dino_feats": ...}.
    """
    driver = get_dino_feats_driver(cfg, image_set, prefix=prefix)
    return get_dataloader_from_driver(driver, batch_size, num_workers, shuffle_key_buffer, shuffle_item_buffer)


def save_dino_feats(
    loader: DataLoader,
    driver: MessagepackDriver,
    model: torch.nn.Module,
    model_proc_key: str = "img",  # what the model processes
    shard_size: int = SHARD_SIZE,
    device: torch.device = None,
):
    """
    Loads images from original STEGO dataloder, computes forward pass of DINO model and saves DINO features to a Squirrel store.
    If the store already exists, we check if the previous computation succeeded based on a status file. If it did, we
    skip the computation. If it did not, we wipe the store and start over.
    """
    if os.path.isdir(driver.url):
        print(f"Directory {driver.url} exists.")
        if status_is_success(driver.url):
            print("Previous calculation succeeded, skipping computation of DINO features.")
            return
        else:
            print("Previous calculation did not succeed. Wiping directory and starting over.")
            shutil.rmtree(driver.url)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    def _gen_feats(item: t.Dict[str, torch.Tensor]):
        with torch.no_grad():
            feats = model(item[model_proc_key].to(device))
        item["dino_feats"] = feats

        return {k: v.cpu().numpy() for k, v in item.items()}

    print(f"Saving DINO features in Squirrel store with url {driver.url} ...")

    def _seperate_samples(it: t.Iterator[t.Dict[str, np.ndarray]]) -> t.Dict[str, np.ndarray]:
        """
        Splits the batched items into single items. This is necessary because we don't want
        to serialize the whole batch and let this determine our shard size, but rather we want
        to individually adjust the shard size. This allows us later to read an arbitrary number
        of items in a batch (also smaller than the batch size we fed the model with).
        """
        for batch in it:
            bs = batch["img"].shape[0]
            batch = {k: np.split(v, bs) for k, v in batch.items()}
            for i in range(bs):
                yield {k: v[i] for k, v in batch.items()}

    feats_it = IterableSource(loader).tqdm().map(_gen_feats)
    write_status_file(driver.url, WriteStatus.WRITING)
    (
        IterableSource(_seperate_samples(feats_it))
        .tqdm()
        .batched(shard_size, drop_last_if_not_full=False)
        .async_map(driver.store.set, buffer=5)
        .join()
    )
    write_status_file(driver.url, WriteStatus.SUCCESS)
    print(f"Done! Saved DINO features in Squirrel store with url {driver.url}.")


def proc_dataset(cfg: DictConfig, image_set: str):
    """Implements high-level logic for processing a dataset and saving its DINO features to a Squirrel store."""
    print(
        f"\nCreating Squirrel dataset with dataset_name={cfg.dataset_name}, "
        f"crop_type={cfg.crop_type}, image_set={image_set}."
    )

    # STEGO uses res=224 for training and res=320 for validation, let's do the same
    res = cfg.res if image_set == "train" else cfg.val_res

    # STEGO uses vit_small for Potsdam, for all other events it uses vit_base
    cfg.model_type = "vit_small" if cfg.dataset_name == "potsdam" else "vit_base"

    # STEGO only does five-crop for training, let's do the same
    if image_set == "val" and cfg.crop_type is not None:
        print("Skipping since crop_type must be None for val set, because we're only validating on non-cropped images.")
        return

    # STEGO only does crop_type=None for Potsdam, let's do the same
    if cfg.crop_type is not None and cfg.dataset_name == "potsdam":
        print("Skipping since crop_type must be None for Potsdam.")
        return

    # STEGO pre-trained models were only trained on five-crop for Cityscapes and Coco, let's do the same
    if cfg.crop_type != "five" and image_set == "train" and cfg.dataset_name in ["cityscapes", "cocostuff27"]:
        print("Skipping since crop_type must be five for Cityscapes and Coco for train set.")
        return

    # We want to stay as close as possible to the STEGO implementation, so we use the default shuffling in
    # the torch Dataloader as opposed to the Squirrel shuffling (which may be less comparable since here
    # we need to set a shuffle_buffer_size which we know impacts the shuffling quality). We save the already
    # shuffled training dataset to the disk.
    shuffle = image_set == "train"  # don't shuffle validation set

    # BEGIN - code taken from precompute_knns.py
    model = torch.nn.Sequential(
        DinoFeaturizer(20, cfg),  # dim doesn't matter
        LambdaLayer(lambda p: p[0]),
    )
    dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set=image_set,
        transform=get_transform(res, False, "center"),
        target_transform=get_transform(res, True, "center"),
        cfg=cfg,
    )
    loader = DataLoader(dataset, cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=False)
    # END

    driver = get_dino_feats_driver(cfg, image_set)
    save_dino_feats(loader, driver, model)


@hydra.main(config_path="configs", config_name="train_config.yml")
def main(cfg: DictConfig) -> None:
    setup_wandb(cfg)
    seed_everything(seed=0)

    image_sets = ["train", "val"]
    dataset_names = ["cocostuff27", "cityscapes", "potsdam"]
    crop_types = ["five", None]

    for crop_type, dataset_name, image_set in itertools.product(crop_types, dataset_names, image_sets):
        cfg.crop_type = crop_type
        cfg.dataset_name = dataset_name
        proc_dataset(cfg, image_set)

    print("Done! Have a nice day.")


if __name__ == "__main__":
    main()
