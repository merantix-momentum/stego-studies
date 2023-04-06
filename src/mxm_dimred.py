import os
import pickle
from copy import copy
from enum import Enum
from multiprocessing import Pool
from os.path import join

import hydra
import numpy as np
import torch
import umap
import wandb
from hnne import HNNE
from mxm_precompute_dino_feats import (
    SHARD_SIZE,
    get_dino_feats_driver,
    get_dino_feats_loader,
    get_only_dino_feats,
    save_dino_feats,
)
from mxm_utils import setup_wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from sklearn import random_projection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from squirrel.driver import FileDriver


class DimRedType(Enum):
    PCA = "PCA"
    UMAP = "UMAP"
    HNNE = "HNNE"
    RP = "RP"


def reshape_from_torch_to_sklearn(feats: torch.Tensor, to_numpy: bool = True, verbose: bool = False) -> np.ndarray:
    """
    Reshapes from torch tensor [B, VIT_DIM, P_H, P_W] to numpy array [B*P_H*P_W, VIT_DIM] for sklearn.
    """
    if verbose:
        print("Features shape:", feats.shape, "Reshaping ...")
    _, VIT_DIM, _, _ = feats.shape
    feats = feats.permute(0, 2, 3, 1)  # move feature dimension to last
    feats = feats.reshape((-1, VIT_DIM))  # reduce batch and patch dimensions
    if to_numpy:
        feats = feats.cpu().numpy()
    if verbose:
        print("New shape:", feats.shape)
    return feats


def reshape_from_sklearn_to_torch(
    feats: np.ndarray,
    batch_size: int,
    patched_h: int,
    patched_w: int,
    verbose: bool = False,
    from_numpy: bool = True,
) -> torch.Tensor:
    """
    Reshapes from numpy array [B*P_H*P_W, VIT_DIM] to torch tensor [B, VIT_DIM, P_H, P_W] for torch.
    Reverses the logic in reshape_from_torch_to_sklearn.
    """
    if verbose:
        print("Features shape:", feats.shape, "Reshaping ...")
    if from_numpy:
        feats = torch.from_numpy(feats)
    feats = feats.reshape((batch_size, patched_h, patched_w, -1))
    feats = feats.permute(0, 3, 1, 2)  # move feature dimension to first
    if verbose:
        print("New shape:", feats.shape)
    return feats


class DimRedRandomProjection(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        n_components: int = None,
        num_images_limit: int = None,
        mem_limit_percent: int = None,
    ) -> None:
        """Dimensionality reduction with Random Projection.

        Args:
            cfg (DictConfig): Hydra training config.
            n_components (int, optional): Number of components to keep. Defaults to None.
            num_images_limit (int, optional): Limit the number of train images to load for fitting. Defaults to None.
            mem_limit_percent (int, optional): Only load samples until mem_limit_percent reached. Defaults to 70.
        """
        super().__init__()

        self.cfg = cfg
        self.n_components = n_components
        self.num_images_limit = num_images_limit
        self.mem_limit_percent = mem_limit_percent

        self.image_set = "train"  # only fit on train set

        # we're not caching RandomProjection as it's so quick to calculate
        self.compute()

    def compute(self) -> None:
        """
        Computes Random Projection on DINO training features and saves results to disk.
        """
        loader = get_dino_feats_loader(self.cfg, self.image_set, batch_size=SHARD_SIZE, num_workers=0)
        print("Loading DINO features from disk ...")
        feats = get_only_dino_feats(loader, self.num_images_limit, self.mem_limit_percent)
        feats = reshape_from_torch_to_sklearn(feats, verbose=True)

        print("Normalizing DINO features ...")
        feats = StandardScaler().fit_transform(feats)

        print("Fitting full RP on DINO features of shape", feats.shape)
        self.random_proj = random_projection.GaussianRandomProjection(n_components=self.n_components).fit(feats)
        print("Fitting done.")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Projects feats onto the RP subspace.

        Args:
            feats (torch.Tensor): Features you want to project.

        Returns:
            torch.Tensor: RP projected features.
        """
        B, VIT_DIM, P_H, P_W = feats.shape
        dev = feats.device

        feats = reshape_from_torch_to_sklearn(feats)
        feats = StandardScaler().fit_transform(feats)
        feats = self.random_proj.transform(feats)
        feats = reshape_from_sklearn_to_torch(feats, B, P_H, P_W).to(dev)

        return feats


class DimRedPCA(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        n_components: int = None,
        num_images_limit: int = None,
        mem_limit_percent: int = None,
    ):
        """Dimensionality reduction with PCA.

        Args:
            cfg (DictConfig): Hydra training config.
            n_components (int, optional): Number of components to keep. Defaults to None.
            num_images_limit (int, optional): Limit the number of train images to load for fitting. Defaults to None.
            mem_limit_percent (int, optional): Only load samples until mem_limit_percent reached. Defaults to 70.
        """
        super().__init__()
        self.cfg = cfg
        self.n_components = n_components
        self.num_images_limit = num_images_limit
        self.mem_limit_percent = mem_limit_percent

        self.image_set = "train"  # only fit on train set
        self.dimred_cache = get_dimred_cache_path(cfg, cfg.pca_conf, DimRedType.PCA, self.image_set)

        try:
            with FileDriver(self.dimred_cache).open(mode="rb") as f:
                self.pca = pickle.load(f)
            print(f"Loaded PCA from cache {self.dimred_cache}.")
        except FileNotFoundError:
            print(f"Could not find cached PCA {self.dimred_cache}. Computing PCA.")
            self.pca = self.compute()

        self.V = torch.from_numpy(self.pca.components_.T)

        if wandb.run is not None:
            # save results for Figure 3 in paper
            res = {
                "pca_explained_variance_ratio": self.pca.explained_variance_ratio_,
                "pca_explained_variance_ratio_cumsum": self.pca.explained_variance_ratio_.cumsum(),
            }
            wandb.log(res)
            print("Logged PCA results to wandb.")

    def compute(self) -> PCA:
        """
        Computes PCA on DINO training features and saves PCA results to disk.
        """
        loader = get_dino_feats_loader(self.cfg, self.image_set, batch_size=SHARD_SIZE, num_workers=0)
        print("Loading DINO features from disk ...")
        feats = get_only_dino_feats(loader, self.num_images_limit, self.mem_limit_percent)
        feats = reshape_from_torch_to_sklearn(feats, verbose=True)

        print("Normalizing DINO features ...")
        feats = StandardScaler().fit_transform(feats)

        print("Fitting full PCA on DINO features of shape", feats.shape)
        pca = PCA().fit(feats)
        print("Fitting done.")

        with FileDriver(self.dimred_cache).open(mode="wb", create_if_not_exists=True) as f:
            pickle.dump(pca, f)

        print(f"Saved PCA under {self.dimred_cache}.")
        return pca

    def forward(self, feats: torch.Tensor, stay_torch: bool = True) -> torch.Tensor:
        """
        Projects feats onto the PCA subspace.

        Args:
            feats (torch.Tensor): Features you want to project.

        Returns:
            torch.Tensor: PCA projected features.
        """
        B, VIT_DIM, P_H, P_W = feats.shape
        dev = feats.device

        if stay_torch:
            # below code is equivalent to the numpy and sklearn solution below, but is faster since it can be executed in the GPU.
            # torch.allclose(np_feats, th_feats, atol=1e-5) returns True.

            th_feats = reshape_from_torch_to_sklearn(feats, to_numpy=False)
            # equiv. to StandardScaler().fit_transform(feats), see https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/7
            th_feats = (th_feats - th_feats.mean(dim=0, keepdim=True)) / th_feats.std(
                dim=0, unbiased=False, keepdim=True
            )
            th_feats = torch.matmul(th_feats, self.V[:, : self.n_components].to(dev))
            th_feats = reshape_from_sklearn_to_torch(th_feats, B, P_H, P_W, from_numpy=False)
            return th_feats

        np_feats = reshape_from_torch_to_sklearn(feats)
        np_feats = StandardScaler().fit_transform(np_feats)
        np_feats = self.pca.transform(np_feats)
        np_feats = np_feats[:, : self.n_components]
        np_feats = reshape_from_sklearn_to_torch(np_feats, B, P_H, P_W).to(dev)
        return feats


class DimRedUMAP(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        num_tokens_limit: int,
        n_components: int = None,
        num_images_limit: int = None,
        mem_limit_percent: int = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ):
        """Dimensionality reduction with UMAP.

        Args:
            cfg (DictConfig): Hydra training config.
            n_components (int, optional): Number of components to keep. Defaults to None.
            num_images_limit (int, optional): Limit the number of train images to load for fitting. Defaults to None.
            mem_limit_percent (int, optional): Only load samples until mem_limit_percent reached. Defaults to 70.
            n_neighbors (int, optional): UMAP parameter. How many neighbors to consider. Defaults to 15.
            min_dist (float, optional): UMAP parameter. Effective minimum distance between embedded points. Defaults to 0.1.
            metric (str, optional): UMAP parameter. Distance metric to use. Defaults to "euclidean".
        """

        super().__init__()

        self.cfg = cfg
        self.num_tokens_limit = num_tokens_limit
        self.n_components = n_components
        self.num_images_limit = num_images_limit
        self.mem_limit_percent = mem_limit_percent
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

        self.image_set = "train"  # only fit on train set
        self.dimred_cache = get_dimred_cache_path(cfg, cfg.umap_conf, DimRedType.UMAP, self.image_set)

        try:
            with FileDriver(self.dimred_cache).open(mode="rb") as f:
                self.reducer = pickle.load(f)
            print(f"Loaded UMAP from cache {self.dimred_cache}.")
        except FileNotFoundError:
            print(f"Could not find cached UMAP {self.dimred_cache}. Computing UMAP.")
            self.reducer = self.compute()

    def compute(self, **kwargs):
        """
        Computes UMAP on DINO training features and saves UMAP results to disk.

        Args:
            feats (torch.Tensor): Training DINO features you want to compute UMAP for.
        """
        loader = get_dino_feats_loader(self.cfg, self.image_set, batch_size=SHARD_SIZE, num_workers=0)
        print("Loading DINO features from disk ...")
        feats = get_only_dino_feats(loader, self.num_images_limit, self.mem_limit_percent)
        feats = reshape_from_torch_to_sklearn(feats, verbose=True)

        # subsample only num_tokens_limit from feats
        print("Loaded DINO feats shape:", feats.shape, "Subsampling ...")
        indices = torch.randperm(feats.shape[0])[: self.num_tokens_limit]
        feats = feats[indices]
        print(f"Subsampled {self.num_tokens_limit} tokens. New shape:", feats.shape)

        # UMAP features should be normalized https://umap-learn.readthedocs.io/en/latest/faq.html
        print("Normalizing DINO features ...")
        feats = StandardScaler().fit_transform(feats)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            **kwargs,
        )

        print(f"Fitting UMAP with {self.n_components} components on data of shape", feats.shape, "...")
        reducer.fit(feats)
        print("Done fitting UMAP.")

        with FileDriver(self.dimred_cache).open(mode="wb", create_if_not_exists=True) as f:
            pickle.dump(reducer, f)
        print(f"Saved UMAP under {self.dimred_cache}.")
        return reducer

    def forward(self, feats: torch.Tensor, num_workers: int = -1):
        """
        Projects feats onto the UMAP subspace.

        Args:
            feats (torch.Tensor): Features you want to project.
            num_workers (int): Number of workers to use for reducer.transform, which doesn't use multiproc by default.
                Defaults to -1, which uses all available CPUs.

        Returns:
            torch.Tensor: UMAP projected features.
        """

        dev = feats.device
        B, VIT_DIM, P_H, P_W = feats.shape

        feats = reshape_from_torch_to_sklearn(feats)
        feats = StandardScaler().fit_transform(feats)
        print("Projecting feats of shape", feats.shape, "onto UMAP subspace ...")

        if num_workers == 0:
            feats = self.reducer.transform(feats)
        else:
            if num_workers == -1:
                num_workers = os.cpu_count()

            with Pool(num_workers) as p:
                feats = p.map(self.reducer.transform, np.array_split(feats, num_workers))
            feats = np.concatenate(feats, axis=0)

        return reshape_from_sklearn_to_torch(feats, B, P_H, P_W).to(dev)


class DimRedHNNE(torch.nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        num_tokens_limit: int,
        n_components: int = None,
        num_images_limit: int = None,
        mem_limit_percent: int = None,
    ):
        """Dimensionality reduction with HNNE.

        Args:
            cfg (DictConfig): Hydra training config.
            num_tokens_limit (int): Upper limit on how many ViT-tokens to fit HNNE on. This ultimately determines the number of training tokens, not num_images_limit.
            n_components (int, optional): Number of components to keep. Defaults to None.
            num_images_limit (int, optional): Limit the number of train images to load for fitting. Defaults to None.
            mem_limit_percent (int, optional): Only load samples until mem_limit_percent reached. Defaults to 70.
        """
        super().__init__()

        self.cfg = cfg
        self.num_tokens_limit = num_tokens_limit
        self.n_components = n_components
        self.num_images_limit = num_images_limit
        self.mem_limit_percent = mem_limit_percent

        self.image_set = "train"  # only fit on train set
        self.compute()

    def compute(self, **kwargs):
        """
        Computes HNNE on DINO training features.
        """
        loader = get_dino_feats_loader(self.cfg, self.image_set, batch_size=SHARD_SIZE, num_workers=0)
        print("Loading DINO features from disk ...")
        feats = get_only_dino_feats(loader, self.num_images_limit, self.mem_limit_percent)
        feats = reshape_from_torch_to_sklearn(feats, verbose=True)

        # subsample only num_tokens_limit from feats
        print("Loaded DINO feats shape:", feats.shape, "Subsampling ...")
        indices = torch.randperm(feats.shape[0])[: self.num_tokens_limit]
        feats = feats[indices]
        print(f"Subsampled {self.num_tokens_limit} tokens. New shape:", feats.shape)

        print("Normalizing DINO features ...")
        feats = StandardScaler().fit_transform(feats)

        self.hnne = HNNE(dim=self.n_components, metric="cosine", **kwargs)
        print(f"Fitting HNNE with {self.n_components} components on data of shape", feats.shape, "...")
        self.hnne.fit(feats, dim=self.n_components)
        print("Done fitting HNNE.")

    def forward(self, feats: torch.Tensor):
        """
        Projects feats onto the HNNE subspace.
        The HNNE transform has good CPU util (unlike the UMAP implementation), so we don't need to use multiprocessing explicitly here.

        Args:
            feats (torch.Tensor): Features you want to project.

        Returns:
            torch.Tensor: HNNE projected features.
        """

        dev = feats.device
        dtype = feats.dtype
        B, VIT_DIM, P_H, P_W = feats.shape

        feats = reshape_from_torch_to_sklearn(feats)
        feats = StandardScaler().fit_transform(feats)
        print("Projecting feats of shape", feats.shape, "onto HNNE subspace ...")

        feats = self.hnne.transform(feats)
        return reshape_from_sklearn_to_torch(feats, B, P_H, P_W).to(device=dev, dtype=dtype)


def get_dimred_cache_path(
    cfg: DictConfig,
    dimred_conf: DictConfig,
    dimred_type: DimRedType,
    image_set: str,
) -> str:
    """Returns path where to save/load the dimensionality reduction model."""

    # for PCA we do not need to save the number of components, because we do a full PCA first,
    # then select the number of components in the forward pass
    if dimred_type == DimRedType.PCA:
        dimred_conf = copy(dimred_conf)
        dimred_conf.pop("n_components")

    suffix = "_".join([f"{k}{v}" for k, v in dimred_conf.items()])
    res = cfg.res if image_set == "train" else cfg.val_res
    dim_red_name = (
        f"dimred_{cfg.model_type}_{cfg.dataset_name}_{image_set}_{cfg.crop_type}_{res}_{dimred_type.value}_{suffix}"
    )
    return join(cfg.root_dimred_store, "dimred", f"{dim_red_name}.pkl")


def get_dimred(cfg: DictConfig) -> torch.nn.Module:
    """Returns instance of dimensionality reduction module."""

    dt = cfg.dimred_type
    if type(dt) == str:
        dt = DimRedType(dt)

    if dt == DimRedType.PCA:
        return DimRedPCA(cfg, **cfg.pca_conf)
    elif dt == DimRedType.RP:
        return DimRedRandomProjection(cfg, **cfg.random_projection_conf)
    elif dt == DimRedType.UMAP:
        return DimRedUMAP(cfg, **cfg.umap_conf)
    elif dt == DimRedType.HNNE:
        return DimRedHNNE(cfg, **cfg.hnne_conf)


def prepare_projected_dataset(cfg: DictConfig, batch_size: int = 512) -> None:
    """Saves pre-computed and down-projected features onto disk.

    Args:
        cfg (DictConfig): Training config.
        batch_size (int, optional): Batch size. This is rather larger, because empirically large batch sizes
            seemed to compute faster (per sample) - at least for UMAP.
    """
    dimred = get_dimred(cfg)

    # get regular squirrel loader over dino features
    for image_set in ["train", "val"]:
        print(f"Preparing {cfg.dimred_type}  {image_set} dataset ...")
        dimred_driver = get_dino_feats_driver(cfg, image_set, f"{cfg.dimred_type}_feats_dim{cfg.dim}")
        # 0 workers in loader to not interfer with multiproc of down-projection
        feats_loader = get_dino_feats_loader(cfg, image_set, batch_size, num_workers=0)

        save_dino_feats(
            feats_loader,
            dimred_driver,
            dimred,  # feed dimred as the model (was previously DINO backbone)
            model_proc_key="dino_feats",
            device="cpu",
        )

    print("Saved pre-computed dimension reduction features.")


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    setup_wandb(cfg)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0)

    # pre-computing the HNNE down-projected features, hard-coding this here for now
    for dim in [100, 768, 384, 192, 100, 96, 48, 24, 12, 6, 3]:
        cfg.dim = dim
        cfg.dimred_type = "HNNE"
        cfg.hnne_conf.n_components = dim
        print("DIM is", dim)
        prepare_projected_dataset(cfg)
    print("Done.")
