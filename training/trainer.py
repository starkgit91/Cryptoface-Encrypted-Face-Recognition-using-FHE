"""
CryptoFace Training Loop.

Implements:
- SGD optimizer with multi-step LR schedule (paper Section 4, Parameters)
- ArcFace loss + Jigsaw puzzle auxiliary loss (Equation 6)
- Gradient clipping (clip norm=1.0 from paper)
- Standard 10-fold cross-validation evaluation

Paper training params:
  lr=0.05, epochs=26, batch_size=256, momentum=0.9, weight_decay=0.0005
  LR decay ×0.1 at epochs [12, 20, 24]
  grad_clip=1.0, α_jigsaw=0.005
"""

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from models.cryptofacenet import CryptoFaceNet
from losses.arcface import ArcFace
from losses.jigsaw import JigsawLoss, CombinedLoss


class AverageMeter:
    """Track running average of a scalar."""

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class CryptoFaceTrainer:
    """
    Trainer for CryptoFaceNet.

    Args:
        model:        CryptoFaceNet model
        num_classes:  Number of training identities
        device:       Torch device
        config:       Training configuration dict
        log_dir:      TensorBoard log directory
        checkpoint_dir: Directory to save checkpoints
    """

    def __init__(self, model: CryptoFaceNet, num_classes: int,
                 device: torch.device, config: Dict[str, Any],
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints'):
        self.model = model
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Losses ────────────────────────────────────────────────────────────
        self.arcface = ArcFace(
            feature_dim=config.get('feature_dim', 256),
            num_classes=num_classes,
            s=config.get('arcface_s', 64.0),
            m=config.get('arcface_m', 0.5),
        ).to(device)

        self.jigsaw_loss = JigsawLoss(
            n_patches=model.n_patches,
            feature_dim=config.get('feature_dim', 256),
            alpha=config.get('jigsaw_alpha', 0.005),
        )

        # ── Optimizer: SGD with momentum (paper params) ───────────────────────
        all_params = list(model.parameters()) + list(self.arcface.parameters())
        self.optimizer = SGD(
            all_params,
            lr=config.get('lr', 0.05),
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4),
        )

        # ── LR Schedule: MultiStep ×0.1 at [12, 20, 24] ──────────────────────
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.get('lr_steps', [12, 20, 24]),
            gamma=config.get('lr_gamma', 0.1),
        )

        # ── Misc ──────────────────────────────────────────────────────────────
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_amp = config.get('amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.best_avg_acc = 0.0

        print(f'\nTrainer initialized:')
        print(f'  Model: {model.get_model_name()}')
        print(f'  Identities: {num_classes:,}')
        print(f'  Device: {device}')
        print(f'  AMP: {self.use_amp}')

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns:
            Dict with 'loss_total', 'loss_arcface', 'loss_jigsaw'
        """
        self.model.train()
        self.arcface.train()

        meters = {
            'total': AverageMeter('Loss/Total'),
            'arcface': AverageMeter('Loss/ArcFace'),
            'jigsaw': AverageMeter('Loss/Jigsaw'),
        }
        t0 = time.time()
        log_freq = self.config.get('log_freq', 100)

        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            B = images.size(0)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Forward: global feature + local features for jigsaw
                global_feat, local_feats = self.model(
                    images, return_local=True)

                # ArcFace loss
                arc_loss = self.arcface(global_feat, labels)

                # Jigsaw loss
                pos_logits = self.model.jigsaw_head(local_feats)
                pos_labels = self.jigsaw_loss.create_position_labels(
                    B, self.device)
                jig_loss = self.jigsaw_loss(pos_logits, pos_labels)

                total_loss = arc_loss + jig_loss

            # Backward
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.arcface.parameters()),
                    self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.arcface.parameters()),
                    self.grad_clip)
                self.optimizer.step()

            # Update meters
            meters['total'].update(total_loss.item(), B)
            meters['arcface'].update(arc_loss.item(), B)
            meters['jigsaw'].update(jig_loss.item(), B)
            self.global_step += 1

            # Log to TensorBoard
            if self.global_step % log_freq == 0:
                self.writer.add_scalar('Train/Loss_Total',
                                       meters['total'].avg, self.global_step)
                self.writer.add_scalar('Train/Loss_ArcFace',
                                       meters['arcface'].avg, self.global_step)
                self.writer.add_scalar('Train/Loss_Jigsaw',
                                       meters['jigsaw'].avg, self.global_step)
                elapsed = time.time() - t0
                print(f'  Epoch {epoch:3d} | Step {step:5d}/{len(loader)} | '
                      f'Loss: {meters["total"].avg:.4f} '
                      f'(arc={meters["arcface"].avg:.4f}, '
                      f'jig={meters["jigsaw"].avg:.4f}) | '
                      f'LR: {self.scheduler.get_last_lr()[0]:.6f} | '
                      f'{elapsed:.1f}s')
                t0 = time.time()

        return {
            'loss_total': meters['total'].avg,
            'loss_arcface': meters['arcface'].avg,
            'loss_jigsaw': meters['jigsaw'].avg,
        }

    def train(self, train_loader: DataLoader, epochs: int,
              eval_fn: Optional[Any] = None,
              start_epoch: int = 0) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            epochs:       Total number of epochs
            eval_fn:      Callable(model) → Dict[dataset → accuracy]
            start_epoch:  Starting epoch (for resuming)

        Returns:
            History dict with losses and accuracies per epoch
        """
        history = {
            'epoch': [],
            'loss_total': [],
            'loss_arcface': [],
            'loss_jigsaw': [],
            'lr': [],
        }

        print(f'\n{"="*60}')
        print(f'Starting training for {epochs} epochs')
        print(f'{"="*60}\n')

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            print(f'\nEpoch {epoch+1}/{epochs} | '
                  f'LR: {self.scheduler.get_last_lr()[0]:.6f}')
            print('-' * 50)

            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch + 1)

            # Step LR scheduler
            self.scheduler.step()

            # Log
            history['epoch'].append(epoch + 1)
            history['loss_total'].append(train_metrics['loss_total'])
            history['loss_arcface'].append(train_metrics['loss_arcface'])
            history['loss_jigsaw'].append(train_metrics['loss_jigsaw'])
            history['lr'].append(self.scheduler.get_last_lr()[0])

            self.writer.add_scalar('Train/LR',
                                   self.scheduler.get_last_lr()[0], epoch + 1)

            # Optional evaluation
            if eval_fn is not None:
                accs = eval_fn(self.model)
                avg_acc = sum(accs.values()) / len(accs)

                for ds, acc in accs.items():
                    self.writer.add_scalar(f'Eval/{ds}', acc, epoch + 1)
                    if ds not in history:
                        history[ds] = []
                    history[ds].append(acc)

                print(f'\n  Eval: ' +
                      ' | '.join(f'{k}={v:.2f}%' for k, v in accs.items()))
                print(f'  Average: {avg_acc:.2f}%')

                # Save best checkpoint
                if avg_acc > self.best_avg_acc:
                    self.best_avg_acc = avg_acc
                    self.save_checkpoint(epoch + 1, accs, is_best=True)

            epoch_time = time.time() - epoch_start
            print(f'\n  Epoch time: {epoch_time:.1f}s')

            # Regular checkpoint
            save_freq = self.config.get('save_freq', 5)
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1, {})

        print(f'\n{"="*60}')
        print(f'Training complete. Best avg accuracy: {self.best_avg_acc:.2f}%')
        print(f'{"="*60}')

        self.writer.close()
        return history

    def save_checkpoint(self, epoch: int, metrics: dict,
                        is_best: bool = False):
        """Save model checkpoint."""
        model_name = self.model.get_model_name()
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'arcface_state': self.arcface.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_avg_acc': self.best_avg_acc,
            'config': self.config,
        }

        fname = f'{model_name}_epoch{epoch:03d}.pth'
        path = self.checkpoint_dir / fname
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / f'{model_name}_best.pth'
            torch.save(state, best_path)
            print(f'  [★] New best checkpoint saved: {best_path}')
        else:
            print(f'  Checkpoint saved: {path}')

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint. Returns starting epoch."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.arcface.load_state_dict(state['arcface_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        self.best_avg_acc = state.get('best_avg_acc', 0.0)
        print(f'Loaded checkpoint from {path} (epoch {state["epoch"]})')
        return state['epoch']
