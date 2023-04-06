# Uncovering the Inner Workings of STEGO for Safe Unsupervised Semantic Segmentation

This repository contains the code accompanying the publication "Uncovering the Inner Workings of STEGO for Safe Unsupervised Semantic Segmentation".
## Citation

```
@inproceedings{koenig2023stego_studies,
    title={Uncovering the Inner Workings of STEGO for Safe Unsupervised Semantic Segmentation},
    author={Alexander Koenig and Maximilian Schambach and Johannes Otterbach},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    year={2023},
}
```

## About

Our paper is a follow-up study on [STEGO](https://arxiv.org/abs/2203.08414). The high-level objective of this code is to stay as close to the original implementation of the method as to ensure comparability and reproducibility. Hence, the code is forked from the original [STEGO repository](https://github.com/mhamilton723/STEGO), and our edits only implement minimal changes to the original code. You can see the main diffs to the original code in [this PR](https://github.com/merantix-momentum/stego-studies/pull/1). There are three types of additions.
1. Minor self-explanatory edits (e.g., formatting or adjusted paths) are not specially indicated.
2. Other edits (e.g., implementing new logging functionality) are wrapped in blocks starting with `# MXM EDIT BEGIN - Description` and ending with `# MXM EDIT END`.
3. New files (e.g., python modules) have the prefix `mxm_` added to their file path to be better distinguishable from the original code.

## Usage

### 1. Getting Started

Clone this repo and install the Python dependencies.

```bash
git clone https://github.com/merantix-momentum/stego-studies
cd stego-studies
pip install -r requirements.txt
```

Now go through the steps in the original [STEGO repository](https://github.com/mhamilton723/STEGO) for downloading pre-trained models, preparing datasets, and computing the k-NN indices.

We extended the code for WandB logging and heavily used it (especially for storing metrics that will later be turned into plots). If you'd also like to use this feature, add a config file at the path `wandb.env`. This file should contain your WandB credentials, as detailed below. If you don't want to use WandB logging, you should be able to fall back to the default Tensorboard logger by changing `log_type` to `tensorboard` in `src/configs/train_config.yaml` and `src/configs/eval_config.yaml`.
```
WANDB_API_KEY=local-abc123
WANDB_BASE_URL=https://...
WANDB_ENTITY=example-entity
WANDB_PROJECT=example-project
```
### 2. Reproducing Table 2

To generate the STEGO results, we followed the evaluation protocol from the original implementation. Simply run inference on the pre-trained models you downloaded.
```bash
MODELS_DIR="/data/datasets/models/saved_models/"
python eval_segmentation.py run_crf=True model_paths=[$MODELS_DIR"cocostuff27_vit_base_5.ckpt"] run_name="jan12_repro_coco_crf"
python eval_segmentation.py run_crf=True model_paths=[$MODELS_DIR"cityscapes_vit_base_1.ckpt"] run_name="jan12_repro_city_crf"
python eval_segmentation.py run_crf=True model_paths=[$MODELS_DIR"potsdam_test.ckpt"] run_name="jan12_repro_pots_crf"
```

To generate the DINO baseline results, you must train the linear layer on the DINO outputs and then evaluate the saved model. Note that we use the `max_steps`, `batch_size`, and `crop_type`, as found in the configuration of the pre-trained STEGO models, to stay comparable. For more information on this, see Table 1 in our paper. We implemented an `only_dino` flag, which removes the STEGO segmentation head from the architecture, and leaves everything else untouched.

```bash
DATE=$(date +"%m-%d_%T")
python train_segmentation.py run_name="final_only_dino_coco_"$DATE only_dino=True correspondence_weight=0 dataset_name="cocostuff27" model_type="vit_base" max_steps=7000 batch_size=32 crop_type="five"
python train_segmentation.py run_name="final_only_dino_city_"$DATE only_dino=True correspondence_weight=0 dataset_name="cityscapes" model_type="vit_base" max_steps=7000 batch_size=32 crop_type="five"
python train_segmentation.py run_name="final_only_dino_pots_"$DATE only_dino=True correspondence_weight=0 dataset_name="potsdam" model_type="vit_small" max_steps=5000 batch_size=16 crop_type=null

CHKPTS_ROOT="/data/output/checkpoints/"
python eval_segmentation.py run_crf=True model_paths=[$CHKPTS_ROOT"final_only_dino_coco_01-23_14:55:24/epoch0-step6399.ckpt"] run_name="jan24_dino_coco_crf"
python eval_segmentation.py run_crf=True model_paths=[$CHKPTS_ROOT"final_only_dino_city_01-23_14:55:24/epoch2-step1199.ckpt"] run_name="jan24_dino_city_crf"
python eval_segmentation.py run_crf=True model_paths=[$CHKPTS_ROOT"final_only_dino_pots_01-24_15:37:40/1809-cluster_miou46.57.ckpt"] run_name="jan24_dino_pots_crf"
```
### 3. Reproducing Figures 3 and 4

In section 4 of the paper, we benchmark the performance of STEGO and traditional dimensionality reduction techniques across various target dimensions. Figure 4 mainly shows the performances without applying CRF to paint an undistorted picture of the underlying feature projection method. Hence, we must first generate the non-CRF results for STEGO and the DINO baseline.

```bash
python eval_segmentation.py run_crf=False model_paths=[$MODELS_DIR"cocostuff27_vit_base_5.ckpt"] run_name="jan12_repro_coco"
python eval_segmentation.py run_crf=False model_paths=[$MODELS_DIR"cityscapes_vit_base_1.ckpt"] run_name="jan12_repro_city"
python eval_segmentation.py run_crf=False model_paths=[$MODELS_DIR"potsdam_test.ckpt"] run_name="jan12_repro_pots"
python eval_segmentation.py run_crf=False model_paths=[$CHKPTS_ROOT"final_only_dino_coco_01-23_14:55:24/epoch0-step6399.ckpt"] run_name="jan24_dino_coco"
python eval_segmentation.py run_crf=False model_paths=[$CHKPTS_ROOT"final_only_dino_city_01-23_14:55:24/epoch2-step1199.ckpt"] run_name="jan24_dino_city"
python eval_segmentation.py run_crf=False model_paths=[$CHKPTS_ROOT"final_only_dino_pots_01-24_15:37:40/1809-cluster_miou46.57.ckpt"] run_name="jan24_dino_pots"
```

Now we train the STEGO segmentation head and the other dimensionality reduction techniques for each dataset. For STEGO, we simply execute the provided `train_segmentation.py` script with different target dimensions to ensure maximum reproducibility.

```bash
# COCOSTUFF
for dim in 90 768 384 192 100 96 48 24 12 6 3:
do
    train_segmentation.py run_name="STEGO_repro_coco_dim"$dim"_date"$DATE only_dino=False dimred_type=null pointwise=True dim=$dim correspondence_weight=1.0 dataset_name="cocostuff27" model_type="vit_base" max_steps=7000 batch_size=32 crop_type=five neg_inter_weight=0.1538476246415498 pos_inter_weight=1 pos_intra_weight=0.1 neg_inter_shift=1 pos_inter_shift=0.2 pos_intra_shift=0.12

# CITYSCAPES
for dim in 100 768 384 192 100 96 48 24 12 6 3:
do
    train_segmentation.py run_name="STEGO_repro_city_dim"$dim"_date"$DATE only_dino=False dimred_type=null pointwise=False dim=$dim correspondence_weight=1.0 dataset_name="cityscapes" model_type="vit_base" max_steps=7000 batch_size=32 crop_type="five" neg_inter_weight=0.9058762625226623 pos_inter_weight=0.577453483136995 pos_intra_weight=1 neg_inter_shift=0.31361241889448443 pos_inter_shift=0.1754346515479633 pos_intra_shift=0.45828472207
done

# POTSDAM
for dim in 70 384 192 100 96 48 24 12 6 3:
do
    train_segmentation.py run_name="STEGO_repro_pots_dim"$dim"_date"$DATE only_dino=False dimred_type=null pointwise=True dim=$dim correspondence_weight=1.0 dataset_name="potsdam" model_type="vit_small" max_steps=5000 batch_size=16 crop_type=null neg_inter_weight=0.63 pos_inter_weight=0.25 pos_intra_weight=0.67 neg_inter_shift=0.76 pos_inter_shift=0.02 pos_intra_shift=0.08
done
```

Note that in Figure 4, we mainly use validation results generated from the `train_segmentation.py` rather than from the `eval_segmentation.py` script. The evaluation code is largely the same except for two things. 
1. First, `train_segmentation.py` performs a single ViT forward pass [here](https://github.com/mhamilton723/STEGO/blob/eb4d6b521740bd4265681b353547f2ffca65d673/src/train_segmentation.py#L260), while the `eval_segmentation.py` script runs a double-forward pass with one regular and one horizontally flipped image and uses the average ViT token embeddings [here](https://github.com/mhamilton723/STEGO/blob/e20df22cf17c41ac78e3c8c75a3118ea87ff0a4c/src/eval_segmentation.py#L125).
2. Second, `train_segmentation.py` first computes linear and cluster predictions and then spatially upsamples predictions [here](https://github.com/mhamilton723/STEGO/blob/eb4d6b521740bd4265681b353547f2ffca65d673/src/train_segmentation.py#L215), while `eval_segmentation.py` first upsamples the feature map and then calculates the predictions [here](https://github.com/mhamilton723/STEGO/blob/eb4d6b521740bd4265681b353547f2ffca65d673/src/eval_segmentation.py#L128). 

In a small benchmark, we found that the validation results coming out of `eval_segmentation.py` are marginally better than those from the `train_segmentation.py` script, but the difference of ~0.5% in mIoU is negligible for our analysis and argumentation. Hence, we directly report the validation results from the `train_segmentation.py` script, which is more convenient and efficient than going through another validation procedure in `eval_segmentation.py`.

For PCA and RP (and HNNE in the appendix), we implemented a mechanism that caches the DINO features using a [Squirrel](https://github.com/merantix-momentum/squirrel-core) data loader. This approach saves lots of compute because the DINO features stay the same regardless of the target dimension we project the features into. Having pre-processed DINO features also speeds the ablations up by a large margin. The pre-processed datasets may become very large, so you might want to save your data to a remote data storage (e.g., GCS bucket or AWS S3) by setting `root_feat_store` to a remote path that [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) can handle. Now, execute the below script, which will cache the DINO features of all datasets under `root_feat_store`.

```bash
python mxm_precompute_dino_feats.py batch_size=64 # better system util with higher batch size
```

We wrote a variant `mxm_train_segmentation.py` of the original `train_segmentation.py` script that uses the pre-computed DINO features instead of running the usual DINO forward pass. It was easier to copy the original `train_segmentation.py` and create a new `mxm_train_segmentation.py` with these more invasive additions instead of editing the original file and keeping it backward compatible. 

Now you can run the remaining benchmarks on the other dimension reduction baselines. Once you start the script, the dimension reduction techniques (i.e., PCA, RP) will be automatically fitted on a subset of the training dataset. For HNNE, we need not only fit the method but also pre-compute its down-projected features because calculating them on the fly in the model's forward pass is prohibitively expensive. Hence, to reproduce the HNNE results from the appendix, run `python mxm_dimred.py dataset_name="cityscapes"`. Consequently, the training script will directly fetch the pre-computed, lower-dimensional HNNE features.

```bash
# COCOSTUFF
for dimred in "PCA" "RP":
do
    for dim in 90 768 384 192 100 96 48 24 12 6 3:
    do
        python mxm_train_segmentation.py run_name=$dimred"_coco_dim"$dim"_date"$DATE only_dino=True dimred_type=$dimred dim=$dim correspondence_weight=0.0 dataset_name="cocostuff27" model_type="vit_base" max_steps=7000 batch_size=32 crop_type="five"
    done
done
    
# CITYSCAPES
for dimred in "PCA" "RP" "HNNE":
do
    for dim in 100 768 384 192 100 96 48 24 12 6 3:
    do
        python mxm_train_segmentation.py run_name=$dimred"_city_dim"$dim"_date"$DATE only_dino=True dimred_type=$dimred dim=$dim correspondence_weight=0.0 dataset_name="cityscapes" model_type="vit_base" max_steps=7000 batch_size=32 crop_type="five" 
    done
done

# POTSDAM
for dimred in "PCA" "RP" "HNNE":
do
    for dim in 70 384 192 100 96 48 24 12 6 3:
    do
        python mxm_train_segmentation.py run_name=$dimred"_pots_dim"$dim"_date"$DATE only_dino=True dimred_type=$dimred dim=$dim correspondence_weight=0.0 dataset_name="potsdam" model_type="vit_small" max_steps=5000 batch_size=16 crop_type=null
    done
done
```

After completing all experiments, you can generate Figures 3 and 4 from the paper and Figures 1 and 2 from the appendix with the below command. Depending on the' DATE' environment variable, you'll need to adapt some experiment names. Also, you'll need to change some WandB settings at the top of the file.

```bash
python figures.py
```