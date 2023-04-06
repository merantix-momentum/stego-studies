import itertools
import os
import pickle
import typing as t
from collections import defaultdict
from datetime import datetime
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from dotenv import load_dotenv

sns.set_theme()

# TODO CHANGE THESE
WANDB_DOTENV_PATH = "wandb.env"
WANDB_ROOT = "your-wandb-root"

DIMS = 3 * 2 ** np.arange(0, 9)
METRICS = ["cluster/mIoU", "cluster/Accuracy", "linear/mIoU", "linear/Accuracy"]
COLORS = sns.color_palette("colorblind")
MAPPING_DS_TITLE = {
    "Cocostuff27": "Cocostuff\n" + r"$D_\mathregular{STEGO}=90$, $D_\mathregular{ViT}=768$",
    "Cityscapes": "Cityscapes\n" + r"$D_\mathregular{STEGO}=100$, $D_\mathregular{ViT}=768$",
    "Potsdam": "Potsdam\n" + r"$D_\mathregular{STEGO}=70$, $D_\mathregular{ViT}=384$",
}
MAPPING_METRIC_YLABEL = {
    "cluster/mIoU": "Unsupervised\nCluster Probe mIoU",
    "linear/mIoU": "Supervised\nLinear Probe mIoU",
    "cluster/Accuracy": "Unsupervised\nCluster Probe Accuracy",
    "linear/Accuracy": "Supervised\nLinear Probe Accuracy",
}

final_coco_results = {
    "run_names": {
        "stego": "jan12_repro_coco",
        "stego_crf": "jan12_repro_coco_crf",
        "stego_repro": "STEGO_repro_coco_dim*_date02-20_12:57:01",
        "dino": "final_only_dino_coco_01-23_14:55:24",
        "dino_crf": "jan24_dino_coco_crf",
        "dino_pca": "PCA_coco_dim*_date02-24_18:19:38",
        "dino_rp": "RP_coco_dim*_date02-26_10:53:44",
        "dino_hnne": None,
    },
    "dims": [768, 384, 192, 96, 90, 48, 24, 12, 6, 3],
    "dataset": "Cocostuff27",
}

final_city_results = {
    "run_names": {
        "stego": "jan12_repro_city",
        "stego_crf": "jan12_repro_city_crf",
        "stego_repro": "STEGO_repro_city_dim*_date02-21_10:01:01",
        "dino": "final_only_dino_city_01-23_14:55:24",
        "dino_crf": "jan24_dino_city_crf",
        "dino_pca": "PCA_city_dim*_date02-14_16:51:58",
        "dino_rp": "RP_city_dim*_date02-26_10:53:44",
        "dino_hnne": None,
    },
    "dims": [768, 384, 192, 100, 96, 48, 24, 12, 6, 3],
    "dataset": "Cityscapes",
}

final_pots_results = {
    "run_names": {
        "stego": "jan12_repro_pots",
        "stego_crf": "jan12_repro_pots_crf",
        "stego_repro": "STEGO_repro_pots_dim*_date02-22_11:45:06",
        "dino": "final_only_dino_pots_01-24_15:37:40",
        "dino_crf": "jan24_dino_pots_crf",
        "dino_pca": "PCA_pots_dim*_date02-23_18:57:20",
        "dino_rp": "RP_pots_dim*_date02-23_18:57:20",
        "dino_hnne": None,
    },
    "dims": [384, 192, 96, 70, 48, 24, 12, 6, 3],
    "dataset": "Potsdam",
}


def get_stego_dim(name):
    """Returns the embedding dimension of the STEGO layer for a given experiment name."""
    name = name.lower()
    if "coco" in name:
        return 90
    elif "city" in name:
        return 100
    elif "pots" in name:
        return 70
    raise ValueError(f"Unknown name {name}")


def get_dino_dim(name):
    """Returns the embedding dimension of the DINO layer for a given experiment name."""
    name = name.lower()
    if "coco" in name or "city" in name:
        return 768  # ViT-B
    elif "pots" in name:
        return 384  # ViT-S
    raise ValueError(f"Unknown name {name}")


def get_wandb_run_by_name(name, wandb_root: str = WANDB_ROOT) -> wandb.run:
    """Fetches wandb runs by name (as opposed to id), which is more convenient for humans."""
    print("Fetching results for", name)
    api = wandb.Api()
    runs = api.runs(path=wandb_root, filters={"display_name": name})
    if len(runs) == 0:
        warn(f"No run with name {name} found.")
        return None
    elif len(runs) > 1:
        warn(f"Multiple runs with name {name} found. Choosing first one...")
    return runs[0]


def get_wandb_metrics_single_run(run: wandb.run, filter_keys=METRICS):
    def _filter_keys(d: t.Dict[str, t.Any]):
        """
        Filters out all keys in d that do not contain any of the filter_keys. Note that the keys
        can be either of style "test/cluster/mIoU" or "final/cluster/mIoU" and both pass the filter.
        """
        return {k: v for k, v in d.items() if any(f in k for f in filter_keys)}

    # like the original STEGO code here https://github.com/mhamilton723/STEGO/blob/eb4d6b521740bd4265681b353547f2ffca65d673/src/train_segmentation.py#L491
    # we look at test/cluster/mIoU and filter out the highest scoring model for reporting results
    # we know it's a train run if it has a "test" prefix, because STEGO val steps log to "test" and not "val" prefix
    max_metric = "test/cluster/mIoU"
    if max_metric in run.summary.keys():
        # for the training runs, let's double check if we trained for at least X steps, because some things may have ended runs early (e.g. preemptible machines shut down early)
        steps = run.summary["trainer/global_step"]
        raise_pots = "pots" in run.name.lower() and steps < 4500  # Should be 5k
        raise_city_coco = ("coco" in run.name.lower() or "city" in run.name.lower()) and steps < 6500  # Should be 7k
        if raise_pots or raise_city_coco:
            warn(f"Run {run.name} has only {run.summary['trainer/global_step']} steps. ")

        hist = run.history()
        max_idx = hist[max_metric].idxmax()
        keys = [f"test/{k}" for k in filter_keys]
        res = {}  # one-liner gives me "hist" is not defined
        for k in keys:
            res[k] = hist[k][max_idx]
    else:
        # these results come from eval_segmentation.py, runs and we can directly use the summary, because only one value is logged
        res = _filter_keys(run.summary)
    return res


def get_wandb_metrics_all_runs(
    run_names,
    dims=DIMS,
    wandb_dotenv_path=WANDB_DOTENV_PATH,
):
    load_dotenv(wandb_dotenv_path)
    date = datetime.now().strftime("%m-%d_%H:%M")
    wandb.init(name=f"plot_figs_{date}", settings=wandb.Settings(code_dir="."))
    results = {}

    for name in ["stego", "stego_crf", "dino", "dino_crf"]:
        run = get_wandb_run_by_name(run_names[name])
        if run is None:
            continue
        results[name] = get_wandb_metrics_single_run(run)
        results[name]["dims"] = (
            get_stego_dim(run_names[name]) if name.startswith("stego") else get_dino_dim(run_names[name])
        )

    def _get_wand_results_for_dims(name: str, dims=DIMS):
        out = defaultdict(list)
        for d in dims:
            run = get_wandb_run_by_name(name.replace("*", str(d)))
            if run is None:
                continue
            res = get_wandb_metrics_single_run(run)
            for k, v in res.items():
                out[k].append(v)
            out["dims"].append(d)
        return out

    for name in ["dino_pca", "dino_rp", "stego_repro", "dino_hnne"]:
        results[name] = _get_wand_results_for_dims(run_names[name], dims=dims) if run_names[name] is not None else None

    return results


def plot_single_dataset(
    run_names: t.Dict[str, str],
    dims=DIMS,
    dataset=str,
    metrics=METRICS,
):
    results = get_wandb_metrics_all_runs(run_names, dims=dims)
    for m in metrics:
        title = f"Validation {m} over embedding dimension for {dataset}"
        plot_metric_over_dims(metric=m, title=title, **results)


def plot_metric_over_dims(
    stego: t.Dict[str, t.Any],
    stego_crf: t.Dict[str, t.Any],
    stego_repro: t.Dict[str, t.Any],
    dino: t.Dict[str, t.Any],
    dino_crf: t.Dict[str, t.Any],
    dino_pca: t.Dict[str, t.Any],
    dino_rp: t.Dict[str, t.Any],
    dino_hnne: t.Dict[str, t.Any],
    metric: str,
    title: str = "",
    ax: matplotlib.axes.Axes = None,
    save_fig: bool = True,
):

    eval_metric = f"final/{metric}"  # STEGO results from eval_segmentation.py (slightly different from validation_epoch_end, images are flipped and double forward passed)
    val_metric = f"test/{metric}"  # Results from validation_epoch_end in train_segmentation.py

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        stego_crf["dims"], stego_crf[eval_metric], marker="v", linestyle="", color=COLORS[3], label="STEGO (theirs)"
    )
    ax.axhline(y=stego[eval_metric], color=COLORS[3], linestyle="--")
    ax.plot(stego["dims"], stego[eval_metric], "o--", color=COLORS[3], label="STEGO (theirs) w/o CRF")
    ax.plot(stego_repro["dims"], stego_repro[val_metric], "x-", color=COLORS[3], label="STEGO")

    ax.plot(dino_crf["dims"], dino_crf[eval_metric], marker="v", linestyle="", color=COLORS[0], label="DINO (ours)")
    ax.axhline(y=dino[val_metric], color=COLORS[0], linestyle="--")
    ax.plot(dino["dims"], dino[val_metric], "o--", color=COLORS[0], label="DINO (ours) w/o CRF")

    if dino_pca is not None:
        ax.plot(dino_pca["dims"], dino_pca[val_metric], "x-", color=COLORS[2], label="PCA")
    if dino_rp is not None:
        ax.plot(dino_rp["dims"], dino_rp[val_metric], "x-", color=COLORS[6], label="RP")
    if dino_hnne is not None:
        ax.plot(dino_hnne["dims"], dino_hnne[val_metric], "x-", color=COLORS[8], label="h-NNE")

    # log scale and set ticks, ScalarFormatter required to make xticks work with logscale
    ax.set_xscale("log")
    ax.set_xticks(DIMS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if not save_fig:
        return

    ax.set_xlabel("embedding dimension")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()

    fig_title = title.replace("/", "_")
    fig_title = "_".join(fig_title.split(" ")).lower()

    if wandb.run is not None:
        wandb.log({fig_title: wandb.Image(plt)})
        print("logged figure to wandb as", fig_title)

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{fig_title}.png", dpi=500)

    plt.clf()


def get_cached_results(results: t.List[t.Dict[str, t.Any]], cache_path: str) -> t.Dict[str, t.Any]:
    # cache results so we don't need to download them from wandb every time
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_results = pickle.load(f)
        print("loaded cached results from", cache_path)
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        all_results = {r["dataset"]: get_wandb_metrics_all_runs(r["run_names"], r["dims"]) for r in results}
        with open(cache_path, "wb") as f:
            pickle.dump(all_results, f)
        print("saved results to", cache_path)
    return all_results


def plot_all_datasets(
    results: t.List[t.Dict[str, t.Any]],
    metrics: t.List[str],
    fig_name: str = "all_datasets.pdf",
    cache_path: str = "figures/cache_results.pkl",
) -> None:

    all_results = get_cached_results(results, cache_path)

    fig, axs = plt.subplots(nrows=len(metrics), ncols=len(results), sharex=True, figsize=(10, 7))

    counter = 0
    subplots = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for i, dataset in enumerate(all_results.keys()):
        # set title in first and second row
        title = MAPPING_DS_TITLE[dataset]
        axs[0, i].set_title(title + "\n\n" + subplots[counter])
        axs[1, i].set_title(subplots[counter + 3])

        # set xlabel in last row
        axs[-1, i].set_xlabel("\n" + r"Embedding Dimension $D$")
        counter += 1

    # set ylabel in first column
    for i, metric in enumerate(metrics):
        axs[i, 0].set_ylabel(MAPPING_METRIC_YLABEL[metric])

    # fill subplots
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(all_results.keys()):
            counter += 1
            plot_metric_over_dims(ax=axs[i, j], metric=metric, save_fig=False, **all_results[dataset])

    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.13, 1, 1], h_pad=1, w_pad=1)
    fig.legend(handles, labels, loc="lower center", ncol=3)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{fig_name}")


def plot_pca_cum_variance(
    fig_dir: str = "figures",
    file_name: str = "pca_cum_variance.pdf",
    root_dimred_store: str = "/data/datasets/",
) -> None:
    from sklearn.decomposition import PCA
    from squirrel.driver import FileDriver

    data = {
        "Cityscapes": "dimred_vit_base_cityscapes_train_five_224_PCA_num_images_limit5000_mem_limit_percentNone.pkl",
        "Potsdam": "dimred_vit_small_potsdam_train_None_224_PCA_num_images_limit5000_mem_limit_percentNone.pkl",
        "Cocostuff": "dimred_vit_base_cocostuff27_train_five_224_PCA_num_images_limit5000_mem_limit_percentNone.pkl",
    }
    max_dim = 768

    fig, ax = plt.subplots(figsize=(4.2, 3))

    for i, (name, url) in enumerate(data.items()):
        path = os.path.join(root_dimred_store, url)
        with FileDriver(path).open("rb") as f:
            pca: PCA = pickle.load(f)

        x = np.linspace(0, 1, len(pca.explained_variance_ratio_.cumsum()))
        new_x = np.linspace(0, 1, max_dim)
        y = np.interp(new_x, x, pca.explained_variance_ratio_.cumsum())
        ax.plot(new_x, y, color=COLORS[i], label=name)

    plt.xticks(
        [0, 1 / 16, 0.25, 0.5, 0.75, 1],
        [
            0,
            r"$\frac{D_\mathregular{ViT}}{16}$",
            r"$\frac{D_\mathregular{ViT}}{4}$",
            r"$\frac{D_\mathregular{ViT}}{2}$",
            r"$\frac{3D_\mathregular{ViT}}{4}$",
            r"$D_\mathregular{ViT}$",
        ],
    )

    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.tight_layout()
    plt.legend()
    plt.show()

    fig.savefig(os.path.join(fig_dir, file_name))
    print(f"Saved figure to {os.path.join(fig_dir, file_name)}")


def plot_hnne_appendix(
    results: t.List[t.Dict[str, t.Any]],
    metrics: t.List[str],
    fig_name: str = "hnne_appendix.pdf",
    cache_path: str = "figures/cache_results_hnne.pkl",
) -> None:

    # this is a bit ugly, but makes sure we can use the other code as is - let's not over-engineer this
    assert len(results) == 1, "This func is intended for one dataset only."
    dataset_name = results[0]["dataset"]

    nrows = 2
    ncols = 2
    all_results = get_cached_results(results, cache_path)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(10, 5.4))

    subplots = ["(a)", "(b)", "(c)", "(d)"]

    for counter, (i, j) in enumerate(itertools.product(range(nrows), range(ncols))):
        axs[i, j].set_title(subplots[counter])
        axs[i, j].set_ylabel(MAPPING_METRIC_YLABEL[metrics[counter]])
        axs[-1, i].set_xlabel("\n" + r"Embedding Dimension $D$")
        plot_metric_over_dims(ax=axs[i, j], metric=metrics[counter], save_fig=False, **all_results[dataset_name])

    handles, labels = axs[0, 1].get_legend_handles_labels()

    # fig.suptitle(MAPPING_DS_TITLE[dataset_name], fontsize="medium")
    fig.tight_layout(rect=[0, 0, 0.7192, 1], h_pad=1, w_pad=1)
    fig.legend(handles, labels, loc="center right", ncol=1)
    os.makedirs("figures", exist_ok=True)

    fig_name = f"{dataset_name}_{fig_name}"
    plt.savefig(f"figures/{fig_name}")


if __name__ == "__main__":

    # plot_single_dataset(**final_coco_results)
    # plot_single_dataset(**final_city_results)
    # plot_single_dataset(**final_pots_results)

    plot_all_datasets(
        results=[final_coco_results, final_city_results, final_pots_results],
        metrics=["linear/mIoU", "cluster/mIoU"],
        fig_name="all_datasets_miou.pdf",
    )
    plot_all_datasets(
        results=[final_coco_results, final_city_results, final_pots_results],
        metrics=["linear/Accuracy", "cluster/Accuracy"],
        fig_name="all_datasets_acc.pdf",
    )

    # add our HNNE experiments to existing results
    final_city_results["run_names"]["dino_hnne"] = "HNNE_city_dim*_date02-26_10:53:44"
    plot_hnne_appendix(
        [final_city_results],
        ["linear/mIoU", "linear/Accuracy", "cluster/mIoU", "cluster/Accuracy"],
        "city_hnne.pdf",
    )
    plot_pca_cum_variance()
