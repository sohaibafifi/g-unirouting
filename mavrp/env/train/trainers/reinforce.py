import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, Timer, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins import AsyncCheckpointIO
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from scipy.stats import ttest_rel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.baselines import RolloutBaseline
from mavrp.env.datasets import MTVRPDataset
from mavrp.env.mixins import InfoMixin
from mavrp.env.models import TransformerModel
from mavrp.env.normalization import CostNormalization
from mavrp.env.train.logging import Stats
from mavrp.env.train.lr_scheduler import MultiStepWithWarmupLR


class ReinforceTrainer(L.LightningModule, InfoMixin):

    def __init__(self, p_config: Config):
        super(ReinforceTrainer, self).__init__()

        self.timer : Timer | None = None
        self.config = p_config

        L.seed_everything(self.config.seed, verbose=False, workers=True)

        warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

        self.stats = Stats(self.config.device)
        self.disable_logger = self.config.disable_logger
        self.wandb_logger = None  # Initialize to None (will be set in get_loggers if needed)

        self.validation_dataloader : DataLoader | None = None
        self.validation_dataset : MTVRPDataset | None = None
        self.test_dataset : MTVRPDataset | None = None
        self.t_dataloader : DataLoader | None = None

        self.model : TransformerModel = TransformerModel(self.config)

        self.baseline = RolloutBaseline(self.config, model=self.model)
        self.normalization = CostNormalization()
        self.name = repr(self.config)
        self.working_dir = str(os.path.join(self.config.working_dir,
                                            self.config.problem,
                                            str(self.config.graph_size),
                                            self.get_name()))

        self.hyper_params = self.config.to_dict()
        # Track whether the baseline has changed since we last cached
        self.baseline_updated = True  # Force a compute the first time
        # Cache of baseline costs for the current validation set: {batch_idx: (baseline, normalized_baseline)}
        self.cached_val_baselines = {}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,
                                      weight_decay=self.config.weight_decay
                                      # eps = 1e-7 if mixed_precision else 1e-8
                                      )
        # milestones at 90% and 95% of the total number of epochs
        milestones = [int(self.config.n_epochs * 0.9), int(self.config.n_epochs * 0.95)]
        scheduler = MultiStepWithWarmupLR(optimizer, milestones=milestones, gamma=0.1,
                                          num_warmup_epochs=self.config.warmup_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def val_dataloader(self):
        if self.validation_dataloader is None:
            self.validation_dataset = self.config.get_problem().dataset(graph_size=self.config.graph_size,
                                                                        num_samples=self.config.nb_val_samples,
                                                                        device=self.config.device)
            self.validation_dataloader = DataLoader(self.validation_dataset,
                                                    batch_size=self.config.batch_size,
                                                    shuffle=False,
                                                    collate_fn=self.validation_dataset.collate_fn
                                                    )
        return self.validation_dataloader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset_type = 'test'
        variant = self.config.problem.lower()
        data_folder = 'data/mtvrp/'
        folder = os.path.join(data_folder, str(self.config.graph_size), variant)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f'{dataset_type}.npz')
        if os.path.exists(file_path):
            self.test_dataset = MTVRPDataset.load(file_path).to(self.config.device)

        else:
            self.test_dataset = self.config.get_problem().dataset(graph_size=self.config.graph_size,
                                                                  num_samples=self.config.nb_test_samples,
                                                                  device=self.config.device)
            self.test_dataset.save(file_path)
        self.t_dataloader = DataLoader(self.test_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=False,
                                       collate_fn=self.test_dataset.collate_fn
                                       )
        return self.t_dataloader

    def test_variant_dataloader(self, variant='mtvrp'):
        dataset_type = 'test'
        data_folder = 'data/mtvrp/'
        folder = os.path.join(data_folder, str(self.config.graph_size), variant)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f'{dataset_type}.npz')
        if os.path.exists(file_path):
            test_dataset = MTVRPDataset.load(file_path).to(self.config.device)

        else:
            test_dataset = self.config.get_problem().dataset_cls()(graph_size=self.config.graph_size,
                                                                   variant=variant,
                                                                   num_samples=self.config.nb_test_samples,
                                                                   device=self.config.device)
            test_dataset.save(file_path)
        self.t_dataloader = DataLoader(test_dataset,
                                       batch_size=self.config.batch_size,
                                       shuffle=False,
                                       collate_fn=test_dataset.collate_fn
                                       )
        return self.t_dataloader, test_dataset

    def train_dataloader(self):
        train_dataset = self.config.get_problem().dataset(graph_size=self.config.graph_size,
                                                          num_samples=self.config.nb_train_samples,
                                                          device=self.config.device)
        return DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )

    def forward(self, inputs: Tuple[Tensor, Tensor], decode_mode: str = "sample",
                actions: Optional[Tensor] = None) -> Any:
        return self.model(inputs, decode_mode, actions)

    def on_train_epoch_start(self) -> None:
        self.stats.reset_validation_epoch()
        self.stats.reset_training_epoch()
        self.model.train()

    def training_step(self, batch, batch_idx):
        log_prob, tours, tour_costs = self.model(batch, decode_mode="sample")

        # Calculate tour costs and advantages
        with torch.no_grad():
            baseline_tour_costs = self.baseline.evaluate(batch, tour_costs)
            if self.config.cost_norm:
                normalized_tour_costs, _ = self.normalization(batch, tour_costs)
                normalized_baseline_tour_costs, _ = self.normalization(batch, baseline_tour_costs)
                cost_advantage = normalized_tour_costs - normalized_baseline_tour_costs
            else:
                cost_advantage = tour_costs - baseline_tour_costs

        # REINFORCE loss on cost
        loss = (cost_advantage * log_prob).mean()

        self.stats.log_training_step(tour_costs, baseline_tour_costs, loss.item())

        # Log metrics including penalty head stats if enabled
        metrics = {
            "_train_cost": tour_costs.mean().item(),
            "_train_baseline_cost": baseline_tour_costs.mean().item(),
            "_train_loss": loss.item(),
        }

        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        _, tours, tour_costs = self.model(batch, decode_mode="greedy")

        if (not self.baseline_updated) and (batch_idx in self.cached_val_baselines):
            baseline_tour_costs = self.cached_val_baselines[batch_idx]
        else:
            with torch.no_grad():
                baseline_tour_costs = self.baseline.validate(batch, tour_costs)
            # Update our cache
            self.cached_val_baselines[batch_idx] = baseline_tour_costs

        if self.config.cost_norm:
            normalized_tour_costs, _ = self.normalization(batch, tour_costs)
            normalized_baseline_tour_costs, _ = self.normalization(batch, baseline_tour_costs)
        else:
            normalized_tour_costs = tour_costs
            normalized_baseline_tour_costs = baseline_tour_costs

        self.stats.log_validation_step(tour_costs, baseline_tour_costs, normalized_tour_costs,
                                       normalized_baseline_tour_costs)


        if self.wandb_logger is not None:
            metrics = {
                "_val_cost": tour_costs.mean().item(),
                "_val_baseline_cost": baseline_tour_costs.mean().item(),
            }

            self.wandb_logger.log_metrics(metrics, step=self.global_step)

    def test_step(self, batch, batch_idx):
        _, _, tour_costs = self.baseline.model.inference(batch, decode_mode="greedy")
        return tour_costs

    def on_validation_epoch_end(self) -> None:
        self.baseline_updated = False

        val_costs = self.stats.epoch_validation_tour_costs_normalized.cpu().numpy()
        val_baseline_costs = self.stats.validation_baseline_tour_costs_normalized.cpu().numpy()
        validation_avg_cost = val_costs.mean()
        validation_avg_baseline_cost = val_baseline_costs.mean()

        if validation_avg_cost < validation_avg_baseline_cost:
            # Perform a paired t-test on costs
            _, pvalue_cost = ttest_rel(val_costs, val_baseline_costs)
            pvalue_cost /= 2  # One-sided t-test

            if pvalue_cost < 0.05:
                # Cost has significantly improved
                # Update the baseline model
                self.baseline.update(self.model)

                # Invalidate the validation data
                self.validation_dataloader = None
                self.baseline_updated = True

    def on_train_epoch_end(self) -> None:
        log_dict : dict[str, float] =  {
            "train_avg_cost": self.stats.epoch_training_tour_costs.cpu().numpy().mean(),
            "train_avg_bcost": self.stats.epoch_training_baseline_tour_costs.cpu().numpy().mean(),

            "valid_avg_cost": self.stats.epoch_validation_tour_costs.cpu().numpy().mean(),
            "valid_avg_bcost": self.stats.validation_baseline_tour_costs.cpu().numpy().mean(),

            "loss": self.stats.epoch_training_loss.cpu().numpy().mean(),

            "train_time": self.timer.time_elapsed("train") / 3600.0 if self.timer else 0.0,
            "val_time": self.timer.time_elapsed("validate") / 3600.0  if self.timer else 0.0,

            "avg_epoch_time": int(
                (self.timer.time_elapsed("train") + self.timer.time_elapsed("validate")) / (self.current_epoch + 1))
            if self.timer else 0.0,
        }


        self.log_dict(log_dict, sync_dist=True)

        self.stats.reset_training_epoch()
        self.stats.reset_validation_epoch()

        if self.current_epoch % (self.config.factor * 50) == 1 or self.current_epoch >= self.config.n_epochs - 1:
            torch.save(self.baseline.state(), os.path.join(self.working_dir, f"baseline.pt"))
            self.test()

    def test(self):
        variants = ['mtvrp'] if self.config.problem == 'MTVRP' else [self.config.problem.lower()]
        if self.current_epoch >= self.config.n_epochs - 1:
            variants = self.config.get_problem().get_variants()
        for variant in variants:
            all_costs = []
            if self.config.cost_norm:
                all_normalized_costs = []
            loader, dataset = self.test_variant_dataloader(variant)
            for idx, batch in tqdm(enumerate(loader), desc=f"Testing {variant}", total=len(loader)):
                batch = [x.to(self.device) for x in batch]
                costs = self.test_step(batch, idx)
                if self.config.cost_norm:
                    normalized_costs, _ = self.normalization(batch, costs)
                all_costs += list(costs.cpu().numpy())
                if self.config.cost_norm:
                    all_normalized_costs += list(normalized_costs.cpu().numpy())
            avg_cost = np.mean(all_costs).item()
            if dataset.pyvrp['cost'] is not None:
                avg_pyvrp_cost = dataset.pyvrp['cost'].mean().cpu().item()
            else:
                avg_pyvrp_cost = float('inf')

            gap = (avg_cost - avg_pyvrp_cost) / avg_cost * 100
            if self.config.cost_norm:
                avg_normalized_cost = np.mean(all_normalized_costs).item()

            if variant == 'mtvrp':
                self.log_dict({
                    f"test_avg_cost": avg_cost,
                }, sync_dist=True)
                self.log_dict({
                    f"test_avg_cost_neg": -avg_cost,
                }, sync_dist=True)
                if avg_pyvrp_cost < float('inf'):
                    self.log_dict({
                        f"test_avg_gap": gap,
                    }, sync_dist=True)
                    self.log_dict({
                        f"test_avg_gap_neg": -gap,
                    }, sync_dist=True)

                if self.config.cost_norm:
                    self.log_dict({
                        f"test_avg_normalized_cost": avg_normalized_cost,
                    }, sync_dist=True)

            self.log_dict({
                f"test_avg_cost_{variant}": avg_cost,
            }, sync_dist=True)
            self.log_dict({
                f"test_avg_cost_{variant}_neg": -avg_cost,
            }, sync_dist=True)
            if avg_pyvrp_cost < float('inf'):
                self.log_dict({
                    f"test_avg_gap_{variant}": gap,
                }, sync_dist=True)
                self.log_dict({
                    f"test_avg_gap_{variant}_neg": -gap,
                }, sync_dist=True)

            if self.config.cost_norm:
                self.log_dict({
                    f"test_avg_normalized_cost_{variant}": avg_normalized_cost,
                }, sync_dist=True)

    def get_loggers(self):
        self.wandb_logger = None
        if self.disable_logger:
            return []
        loggers = [CSVLogger(save_dir=self.working_dir)]
        try:
            import configs.wandb
            path = configs.wandb.wandb['project']
            import wandb
            wandb.login(key=configs.wandb.wandb['api_key'], verify=True)
            api = wandb.Api(api_key=configs.wandb.wandb['api_key'])
            runs = api.runs(path=configs.wandb.wandb['project'], filters={
                "config.problem": self.config.problem,
                "config.graph_size": self.config.graph_size,
                "display_name": self.get_name()
            })
            run = None if len(runs) == 0 else runs[0]
            project_name = path.split('/')[-1]
            from lightning.pytorch.loggers import WandbLogger
            self.wandb_logger = WandbLogger(project=project_name, name=self.get_name(), log_model=False,
                                            dir=self.working_dir,
                                            id=run.id if run is not None else None,
                                            resume="must" if run is not None else 'allow')
            self.wandb_logger.log_hyperparams(self.hyper_params)
            loggers.append(self.wandb_logger)
        except ImportError:
            self.wandb_logger = None

        return loggers

    def get_trainer(self, resume=True):

        self.timer = Timer()
        callbacks = [
            RichModelSummary(max_depth=7),
            TQDMProgressBar(leave=True),
            self.timer
        ]
        if not self.config.disable_logger:
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))

        if resume:
            checkpoint_callback = ModelCheckpoint(dirpath=self.working_dir,
                                                  save_top_k=1,
                                                  enable_version_counter=False,
                                                  save_last=True,
                                                  filename='checkpoint',
                                                  monitor=self.get_monitor(),
                                                  every_n_epochs=1)
            callbacks.append(checkpoint_callback)
        loggers = self.get_loggers()

        return L.Trainer(max_epochs=self.config.n_epochs,
                         enable_model_summary=True,
                         enable_checkpointing=resume,
                         num_sanity_val_steps=0,
                         logger=loggers,
                         default_root_dir=self.working_dir,
                         accelerator='gpu' if self.config.device in ['cuda', 'mps'] else 'cpu',
                         reload_dataloaders_every_n_epochs=1,
                         callbacks=callbacks,
                         plugins=[AsyncCheckpointIO(), ],
                         gradient_clip_val=1,
                         deterministic=self.config.deterministic_train,  # Ensure reproducibility
                         benchmark=not self.config.deterministic_train
                         )

    def get_monitor(self):
        return 'valid_avg_bcost'

    def fit(self, resume=True):
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.trainer = self.get_trainer(resume=resume)
        return self.trainer.fit(self, ckpt_path="last")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['configs'] = self.config
        checkpoint['stats'] = self.stats.__dict__
        checkpoint['baseline'] = self.baseline.state()

        checkpoint['baseline_updated'] = self.baseline_updated
        checkpoint['cached_val_baselines'] = self.cached_val_baselines



    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.baseline.load_state(checkpoint['baseline'])
        self.stats.__dict__ = checkpoint['stats']
        if 'baseline_updated' in checkpoint and 'cached_val_baselines' in checkpoint:
            self.baseline_updated = checkpoint['baseline_updated']
            self.cached_val_baselines = checkpoint['cached_val_baselines']
        else:
            self.baseline_updated = True
            self.cached_val_baselines = {}

