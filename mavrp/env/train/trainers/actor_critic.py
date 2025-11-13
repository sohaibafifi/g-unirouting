import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from mavrp.configs.config import Config
from mavrp.env.models import CriticModel
from mavrp.env.train.lr_scheduler import MultiStepWithWarmupLR
from mavrp.env.train.trainers.reinforce import ReinforceTrainer


class ActorCriticTrainer(ReinforceTrainer):
    """
    Actor–Critic trainer inheriting from ReinforceTrainer.
    Overrides only optimizer config and training_step to include a critic.
    """
    def __init__(self, p_config: Config):
        super().__init__(p_config)
        # Critic network to estimate state-value (expected tour cost)
        self.critic = CriticModel(self.config)
        # Remove unused baseline fields inherited from ReinforceTrainer
        for attr in ("baseline", "normalization", "baseline_updated", "cached_val_baselines"):
            if hasattr(self, attr):
                delattr(self, attr)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        # same LR schedule as base
        milestones = [int(self.config.n_epochs * 0.9), int(self.config.n_epochs * 0.95)]
        scheduler = MultiStepWithWarmupLR(
            optimizer, milestones=milestones, gamma=0.1,
            num_warmup_epochs=self.config.warmup_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }

    def training_step(self, batch, batch_idx):
        # Actor: sample a tour and get log-probabilities
        log_prob, _, tour_costs = self.model(batch, decode_mode="sample")

        # Critic: predict state-value
        values = self.critic(batch)  # shape [batch_size]
        # Advantage: actual cost minus estimated value
        advantages = tour_costs - values.detach()

        # Compute actor and critic losses
        actor_loss = (log_prob * advantages).mean()
        critic_loss = F.mse_loss(values, tour_costs)
        loss = actor_loss + self.config.critic_coef * critic_loss

        # Add penalty scale regularization if trainable
        penalty_reg_loss = self.model.decoder.get_penalty_scale_regularization_loss()
        if penalty_reg_loss is not None:
            loss = loss + penalty_reg_loss

        # Update penalty scale EMA
        if hasattr(self.model.decoder, 'update_penalty_scale_ema'):
            self.model.decoder.update_penalty_scale_ema()

        # Logging (reuse stats infrastructure)
        self.stats.log_training_step(tour_costs, values.detach(), loss.item())

        # Log metrics including penalty head stats if enabled
        metrics = {
            "_train_cost": tour_costs.mean().item(),
            "_train_baseline_cost": values.mean().item(),
            "_train_loss": loss.item(),
            "_actor_loss": actor_loss.item(),
            "_critic_loss": critic_loss.item()
        }

        if penalty_reg_loss is not None:
            metrics["_train_penalty_reg_loss"] = penalty_reg_loss.item()

        # Add penalty head statistics if available
        if self.penalty_trainer is not None:
            penalty_stats = self.penalty_trainer.get_statistics()
            if penalty_stats:
                metrics.update({f"_penalty_{k}": v for k, v in penalty_stats.items()})
                # Log penalty scale
                penalty_scale_val = self.model.decoder.get_penalty_scale()
                if penalty_scale_val is not None:
                    metrics["_penalty_scale"] = penalty_scale_val

                # Log EMA if available
                if hasattr(self.model.decoder, 'penalty_scale_ema'):
                    metrics["_penalty_scale_ema"] = self.model.decoder.penalty_scale_ema.item()

        if self.wandb_logger is not None:
            self.wandb_logger.log_metrics(metrics, step=self.global_step)

        return loss


    def validation_step(self, batch, batch_idx):
        _, _, tour_costs = self.model(batch, decode_mode="greedy")
        # log only the model cost (we repeat it to satisfy Stats API)
        self.stats.log_validation_step(tour_costs, tour_costs, tour_costs, tour_costs)
        if self.wandb_logger:
            self.wandb_logger.log_metrics({"_val_cost": tour_costs.mean().item()},
                                          step=self.global_step)

    def on_validation_epoch_end(self):
        # no baseline comparison — just log the avg cost
        self.log_dict({
            "valid_avg_cost": self.stats.epoch_validation_tour_costs.cpu().mean().item(),
        }, sync_dist=True)
        self.stats.reset_validation_epoch()

    def test_step(self, batch, batch_idx):
        _, _, tour_costs = self.model.inference(batch, decode_mode="greedy")
        return tour_costs

    def on_train_epoch_end(self):
        # log only the actor cost and loss
        log_dict = {
            "train_avg_cost": self.stats.epoch_training_tour_costs.cpu().mean().item(),
            "loss": self.stats.epoch_training_loss.cpu().mean().item(),

            "train_time": self.timer.time_elapsed("train") / 3600.0,

            "avg_epoch_time": int(
                (self.timer.time_elapsed("train") + self.timer.time_elapsed("validate")) / (self.current_epoch + 1)),
        }

        # Add penalty head statistics if available
        if self.penalty_trainer is not None:
            penalty_stats = self.penalty_trainer.get_statistics()
            if penalty_stats:
                log_dict.update({f"penalty_{k}": v for k, v in penalty_stats.items()})
                # Log penalty scale
                penalty_scale_val = self.model.decoder.get_penalty_scale()
                if penalty_scale_val is not None:
                    log_dict["penalty_scale"] = penalty_scale_val

        self.log_dict(log_dict, sync_dist=True)
        self.stats.reset_training_epoch()

        if self.current_epoch % (self.config.factor * 50) == 1 or self.current_epoch >= self.config.n_epochs - 1:
            # Save penalty trainer state if enabled
            if self.penalty_trainer is not None:
                penalty_state = {
                    'buffer': self.penalty_trainer.experience_buffer,
                    'optimizer': self.penalty_trainer.optimizer.state_dict() if self.penalty_trainer.optimizer else None
                }
                torch.save(penalty_state, os.path.join(self.working_dir, f"penalty_trainer.pt"))
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
                all_costs += list(costs.cpu().numpy())

            avg_cost = np.mean(all_costs).item()
            if dataset.pyvrp['cost'] is not None:
                avg_pyvrp_cost = dataset.pyvrp['cost'].mean().cpu().item()
            else:
                avg_pyvrp_cost = float('inf')

            gap = (avg_cost - avg_pyvrp_cost) / avg_cost * 100

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



    def on_save_checkpoint(self, checkpoint):
        # save only config & stats
        checkpoint['configs'] = self.config
        checkpoint['stats'] = self.stats.__dict__

    def on_load_checkpoint(self, checkpoint):
        self.stats.__dict__ = checkpoint['stats']

    def get_monitor(self):
        return 'train_avg_cost'
