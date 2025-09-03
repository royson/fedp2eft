import torch
from transformers import Trainer, Seq2SeqTrainer
from trl import SFTTrainer
from src.models.btlora.layer import sparsity_loss, significance_loss
from copy import deepcopy
from src.models.btlora.layer import fedbt_get_bts
from src.models.model_utils import clampSTE_max
from torch.nn import CrossEntropyLoss


class FedP2EFTTrainer(Trainer):
    '''
    Wrapper around transformers.Trainer
    Includes Bayestune losses
    '''

    def __init__(self, *args,
                 loss_weights,
                 bts_masks=None,
                 bts_max_clamp=1e+4,
                 debug=False,
                 result_fp='',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.init_model = deepcopy(self.model)
        self.loss_weights = loss_weights
        self.bts_masks = bts_masks
        # self.task_loss_reduction = task_loss_reduction
        self.bts_max_clamp = bts_max_clamp
        self.bts_eps = 0 if bts_masks is None else 1e-9

        # for debugging
        self.cus_debug = debug
        self.result_fp = result_fp
        if debug:
            self.save_losses = {}
            self.save_step = 0


class FedP2EFTSFTTrainer(FedP2EFTTrainer, SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        bts = fedbt_get_bts(model)
        if self.bts_masks:
            for idx, bt in enumerate(bts):
                bts[idx] = clampSTE_max(
                    bt, min_limit=1e-4, max_limit=self.bts_max_clamp) * self.bts_masks[idx]
        else:
            for idx, bt in enumerate(bts):
                bts[idx] = clampSTE_max(
                    bt, min_limit=1e-4, max_limit=self.bts_max_clamp)

        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs)

        loss = self.loss_weights['task'] * loss
        if self.cus_debug:
            task_loss = loss.item()

        if self.loss_weights['significance']:
            sig_loss = self.loss_weights['significance'] * significance_loss(
                model, self.init_model, bts, eps=self.bts_eps) / len(self.train_dataset)
            loss = loss + sig_loss

        if self.loss_weights['sparsity']:
            sparse_loss = self.loss_weights['sparsity'] * sparsity_loss(
                bts, eps=self.bts_eps) / len(self.train_dataset)
            loss = loss + sparse_loss

        # for debugging
        if self.cus_debug:
            # print(f'Mean BTS: {torch.stack(bts).mean().item()}')
            # print(f'Mean BTS: {torch.stack(bts).mean().item()}. Max BTS: {torch.stack(bts).max().item()}')
            # print(f"[Losses] task ({self.loss_weights['task']}): {task_loss}")
            # print(f"[Losses] sig ({self.loss_weights['significance']}): {sig_loss}")
            # print(f"[Losses] sparse ({self.loss_weights['sparsity']}): {sparse_loss}")

            self.save_losses[self.save_step] = {
                'task': task_loss,
                'sig': sig_loss if type(sig_loss) == float else sig_loss.item(),
                'sparsity': sparse_loss.item(),
                'bts': torch.stack(bts).mean().item(),
                'qr': torch.quantile(torch.stack(bts).flatten().float().cpu(), torch.tensor([0.25, 0.5, 0.75])).tolist(),
                'min_bts': torch.stack(bts).min().item(),
                'max_bts': torch.stack(bts).max().item(),
            }
            torch.save(self.save_losses, f'{self.result_fp}')
            self.save_step += 1

        return (loss, outputs) if return_outputs else loss


class FedP2EFTSeqClsTrainer(FedP2EFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        bts = fedbt_get_bts(model)
        if self.bts_masks:
            for idx, bt in enumerate(bts):
                bts[idx] = clampSTE_max(
                    bt, min_limit=1e-4, max_limit=self.bts_max_clamp) * self.bts_masks[idx]
        else:
            for idx, bt in enumerate(bts):
                bts[idx] = clampSTE_max(
                    bt, min_limit=1e-4, max_limit=self.bts_max_clamp)

        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
        else:
            loss = super().compute_loss(model, inputs)

        loss = self.loss_weights['task'] * loss
        if self.cus_debug:
            task_loss = loss.item()

        if self.loss_weights['significance']:
            sig_loss = self.loss_weights['significance'] * significance_loss(
                model, self.init_model, bts, eps=self.bts_eps) / len(self.train_dataset)
            loss = loss + sig_loss

        if self.loss_weights['sparsity']:
            sparse_loss = self.loss_weights['sparsity'] * sparsity_loss(
                bts, eps=self.bts_eps) / len(self.train_dataset)
            loss = loss + sparse_loss

        # for debugging
        if self.cus_debug:
            # print(f'Mean BTS: {torch.stack(bts).mean().item()}')
            # print(f'Mean BTS: {torch.stack(bts).mean().item()}. Max BTS: {torch.stack(bts).max().item()}')
            # print(f"[Losses] task ({self.loss_weights['task']}): {loss.item()}")
            # print(f"[Losses] sig ({self.loss_weights['significance']}): {sig_loss.item()}")
            # print(f"[Losses] sparse ({self.loss_weights['sparsity']}): {sparse_loss.item()}")
            self.save_losses[self.save_step] = {
                'task': task_loss,
                'sig': sig_loss.item(),
                'sparsity': sparse_loss.item(),
                'bts': torch.stack(bts).mean().item(),
                'qr': torch.quantile(torch.stack(bts).flatten().float().cpu(), torch.tensor([0.25, 0.5, 0.75])).tolist(),
                'min_bts': torch.min(torch.stack(bts)).item(),
                'max_bts': torch.max(torch.stack(bts)).item(),
            }
            torch.save(self.save_losses, f'{self.result_fp}')
            self.save_step += 1

        return (loss, outputs) if return_outputs else loss
