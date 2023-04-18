import sys
from datetime import datetime

import hydra
import pytorch_lightning as pl
import seaborn as sns
import torch.multiprocessing
import torch.nn.functional as F
from data import *
from modules import *
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

# MXM EDIT BEGIN - Imports
import wandb
from mxm_utils import dimred_in_forward_pass, do_dimred, only_dino, setup_wandb
from pytorch_lightning.loggers import WandbLogger

# MXM EDIT END

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # MXM EDIT BEGIN - Adjust input dimension for cluster lookup and linear probe to number of output ViT channels of the "only_dino" baseline
        # if dino baseline, unreduced dimension "n_feats", if dimred (including STEGO) target dimension is dim
        target_dim = self.net.n_feats if only_dino(cfg) and not do_dimred(cfg) else dim
        self.cluster_probe = ClusterLookup(target_dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(target_dim, n_classes, (1, 1))
        # MXM EDIT END
        
        # MXM EDIT BEGIN - For the DINO only baseline, we don't need the STEGO loss calculation and can avoid forward pass of "KNN" and "Random" images
        if only_dino(cfg) and cfg.correspondence_weight != 0:
            raise ValueError("Training DINO baseline expects correspondence_weight=0.")
        # MXM EDIT END
        
        # MXM EDIT BEGIN - Instead of the STEGO clustering layer, we use a simple dimensionality reduction layer.
        if dimred_in_forward_pass(cfg):
            from mxm_dimred import get_dimred
            self.dimred = get_dimred(cfg)
        # MXM EDIT END
        
        # MXM EDIT BEGIN - Without changing original implementation too much, we can only safely plot batch_size of validation images.
        if "val_batch_size" in cfg and cfg.val_batch_size < cfg.n_images:
            print(f"Reducing number of validation images {cfg.n_images} to batch size {cfg.val_batch_size}.")
            cfg.n_images = cfg.val_batch_size
        # MXM EDIT END
        
        # MXM EDIT BEGIN - Decoder is only trained when rec_weight > 0, hence we also check for this here to avoid loading unnecessary weights into GPU memory
        if self.cfg.rec_weight > 0:
            self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))
        # MXM EDIT END

        # MXM EDIT BEGIN - Pass human-readable class labels into metrics calculator to later log class-specific metrics. 
        # Also note that "test/cluster" actually refers to validation results not test! This is the original implementation and we won't change it here.
        class_labels = get_class_labels(cfg.dataset_name)
        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", class_labels, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", class_labels, 0, False)
        # MXM EDIT END
        
        # MXM EDIT BEGIN - Since we want to log the training cluster metrics to more clearly idenfity overfitting, we add "train_cluster_metrics" and "train_linear_metrics".
        self.train_cluster_metrics = UnsupervisedMetrics(
            "train/cluster/", class_labels, cfg.extra_clusters, True)
        self.train_linear_metrics = UnsupervisedMetrics(
            "train/linear/", class_labels, 0, False)
        # MXM EDIT END
        
        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", class_labels, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", class_labels, 0, False)
                
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

        feats, code = self.net(img)
        # MXM EDIT BEGIN - Dimensionality reduction if requested.
        if dimred_in_forward_pass(self.cfg):
            code = self.dimred(code)
        # MXM EDIT END
        
        if self.cfg.correspondence_weight > 0:
            feats_pos, code_pos = self.net(img_pos)
        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            # MXM EDIT BEGIN - Avoid "'feats_pos' referenced before assignment" error when correspondence_weight=0.
            if self.cfg.correspondence_weight > 0:
                signal = feats
                signal_pos = feats_pos
            # MXM EDIT END

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)
        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        if self.cfg.correspondence_weight > 0:
            (
                pos_intra_loss, pos_intra_cd,
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd,
            ) = self.contrastive_corr_loss_fn(
                signal, signal_pos,
                salience, salience_pos,
                code, code_pos,
            )

            if should_log_hist:
                if self.cfg.log_type == "wandb":
                    wandb.log({"histograms/intra_cd": wandb.Histogram(pos_intra_cd.detach().cpu())}, step=self.global_step)
                    wandb.log({"histograms/inter_cd": wandb.Histogram(pos_inter_cd.detach().cpu())}, step=self.global_step)
                    wandb.log({"histograms/neg_cd": wandb.Histogram(neg_inter_cd.detach().cpu())}, step=self.global_step)
                elif self.cfg.log_type == "tensorboard":
                    self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                    self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                    self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)
            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            loss += self.cfg.aug_alignment_weight * aug_alignment

        if self.cfg.crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            loss += self.cfg.crf_weight * crf

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        # MXM EDIT BEGIN - Log cluster and linear metrics for training set. We also have to bilinearly upsample the cluster probs to match the label size. 
        self.train_linear_metrics.update(linear_logits.argmax(1), label)
        cluster_probs = F.interpolate(cluster_probs, label.shape[-2:], mode='bilinear', align_corners=False)
        self.train_cluster_metrics.update(cluster_probs.argmax(1), label)
        # MXM EDIT END
        
        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        if self.cfg.log_type == "tensorboard" and self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        # MXM EDIT BEGIN - Log metrics for training set. This follows the same logic in as in on_validation_epoch_end, and copies some code, to avoid having to change the original code.
        if self.global_step % self.cfg.train_metrics_log_interval == 0:
            metrics = {
                **self.train_linear_metrics.compute(),
                **self.train_cluster_metrics.compute(),
            }
            self.log_dict(metrics)
            self.train_linear_metrics.reset()
            self.train_cluster_metrics.reset()
        # MXM EDIT END
        return loss
    
    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute(),
            # MXM EDIT BEGIN - Not sure why STEGO computes metrics here already, but let's do the same for our added train metrics.
            **self.train_linear_metrics.compute(),
            **self.train_cluster_metrics.compute(),
            # MXM EDIT END
        }
        if self.cfg.log_type == "tensorboard": 
            self.logger.log_hyperparams(self.cfg, tb_metrics)
        else:
            self.logger.log_hyperparams(self.cfg)
            
    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)
            # MXM EDIT BEGIN - Dimensionality reduction if requested.
            if dimred_in_forward_pass(self.cfg):
                code = self.dimred(code)
            # MXM EDIT END
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            # MXM EDIT BEGIN - Only return the validation results from the first batch. The original implementation 
            # returns the validation results every epoch here. Since PyTorch Lightning saves all intermediate validation
            # results this OOMs our system as the intermediate results are rather large images. Only returning the results
            # from the first batch here will only influence the visualization in validation_epoch_end, but not the final 
            # metrics as their data is stored in self.linear_metrics and self.cluster_metrics, respectively.
            if batch_idx == 0:
                return {
                    'img': img[:self.cfg.n_images].detach().cpu(),
                    'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                    "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                    "label": label[:self.cfg.n_images].detach().cpu()}
            # MXM EDIT END

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]])
                    ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                    ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
                ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.cfg.log_type, self.logger.experiment, "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    names = get_class_labels(self.cfg.dataset_name)
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=14)
                    ax.yaxis.set_ticklabels(names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    # ax.yaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    # ax.xaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.cfg.log_type, self.logger.experiment, "conf_matrix", self.global_step)

                    all_bars = torch.cat([
                        self.cluster_metrics.histogram.sum(0).cpu(),
                        self.cluster_metrics.histogram.sum(1).cpu()
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    ax[0].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(0).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    ax[1].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(1).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    add_plot(self.cfg.log_type, self.logger.experiment, "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return net_optim, linear_probe_optim, cluster_probe_optim


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    setup_wandb(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    # MXM EDIT BEGIN - Use new val_batch_size key from config here. Validation batch size should be lower than training batch size, 
    # since validation images are 320*320 by default and take up more memory than the 244*244 training images.
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, cfg.val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    # MXM EDIT END

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    if cfg.log_type == "tensorboard":
        logger = TensorBoardLogger(
            join(log_dir, name),
            default_hp_metric=False
        )
    elif cfg.log_type == "wandb":
        logger = WandbLogger(log_model=cfg.log_model)
        if cfg.wandb_watch:
            logger.watch(model, log="all", log_freq=20)
    ckpt_pth = join(checkpoint_dir, cfg.run_name)
    os.makedirs(ckpt_pth, exist_ok=True)

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=logger,
        max_steps=cfg.max_steps,
        # MXM EDIT BEGIN - Custom arguments, replacing the "gpu_args" dictionary, which was modifying
        # val_check_interval depending on training dataset size. This did not make sense for us, because we're
        # training on one GPU.
        gpus=-1,
        val_check_interval=cfg.val_freq,       
        # MXM EDIT END
        callbacks=[
            ModelCheckpoint(
                dirpath=None if cfg.log_type == "wandb" else ckpt_pth, # log model to wandb instead if using wandb
                # MXM EDIT BEGIN - Comment out every_n_train_steps=400 because val_check_interval=100 by default, which 
                # means that not every validation step is taken into account in the model checkpointing but only every 4th.
                # every_n_train_steps=400,
                # MXM EDIT END
                save_top_k=2,
                monitor="test/cluster/mIoU",
                mode="max",
                # MXM EDIT BEGIN - Adding a custom filename to the checkpoint, which includes the cluster mIoU.
                save_last=True,
                filename="step{step}-cluster_miou{test/cluster/mIoU:.2f}",
                auto_insert_metric_name=False,
                # MXM EDIT END
            )
        ],
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
