"""Faithful single-GPU SSL objectives adapted from the pinned official repos.

The formulas in this file follow USB commit 1ef4cbebcc0b368158315aeb425053858cf6c845
for FixMatch, SoftMatch, FlexMatch, and FreeMatch, and CGMatch commit
e77d020c6bedb13a1aac2882d1d1a9b513a0f71a for CGMatch.  The surrounding
network, data budget, and optimizer belong to the AttrVR experiment protocol.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Iterable

import torch
import torch.nn.functional as F


METHODS = ("fixmatch", "softmatch", "cgmatch", "flexmatch", "freematch")


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """USB consistency loss: masked per-sample CE followed by batch mean."""
    return (F.cross_entropy(logits, targets, reduction="none") * mask.float()).mean()


def generalized_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    q: float = 0.7,
) -> torch.Tensor:
    """CGMatch generalized cross entropy, including the official batch mean."""
    # Do not clamp in place: the probabilities remain in the autograd graph for
    # the ambiguous strong branch.
    probs = logits.softmax(dim=1).clamp(1e-7, 1.0)
    selected = probs.gather(1, targets[:, None]).squeeze(1)
    loss = (1.0 - selected.pow(q)) / q
    return (loss * mask.float()).mean()


def _replace_inf_with_zero(value: torch.Tensor) -> torch.Tensor:
    value = value.clone()
    value[torch.isinf(value)] = 0.0
    return value


def freematch_entropy_loss(
    mask: torch.Tensor,
    logits_s: torch.Tensor,
    p_model: torch.Tensor,
    label_hist: torch.Tensor,
) -> torch.Tensor:
    """SAF loss copied algebraically from USB FreeMatch."""
    selected = logits_s[mask.bool()]
    if selected.numel() == 0:
        return logits_s.new_zeros(())
    prob_s = selected.softmax(dim=-1)
    pred_s = prob_s.argmax(dim=-1)
    hist_s = torch.bincount(pred_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    model_scaler = _replace_inf_with_zero(1.0 / label_hist).detach()
    mod_model = p_model * model_scaler
    mod_model = mod_model / mod_model.sum()

    mean_scaler = _replace_inf_with_zero(1.0 / hist_s).detach()
    mod_mean = prob_s.mean(dim=0) * mean_scaler
    mod_mean = mod_mean / mod_mean.sum()
    return (mod_model * torch.log(mod_mean + 1e-12)).sum()


class SSLController:
    """State and loss construction for one official SSL selection rule."""

    def __init__(
        self,
        method: str,
        num_classes: int,
        num_unlabeled: int,
        device: torch.device,
        total_steps: int,
        *,
        p_cutoff: float = 0.95,
        ema_p: float = 0.999,
        n_sigma: float = 2.0,
        entropy_ratio: float = 0.001,
        cg_queue_size: int = 1000,
        cg_warmup_steps: int = 0,
        cg_lambda_h: float = 1.0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown SSL method: {method}")
        self.method = method
        self.num_classes = num_classes
        self.num_unlabeled = num_unlabeled
        self.device = device
        self.total_steps = total_steps
        self.p_cutoff = p_cutoff
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.entropy_ratio = entropy_ratio

        uniform = torch.ones(num_classes, device=device) / num_classes
        self.p_model = uniform.clone()
        self.p_target = uniform.clone()
        self.label_hist = uniform.clone()
        self.time_p = uniform.mean()

        self.soft_mu = torch.tensor(1.0 / num_classes, device=device)
        self.soft_var = torch.tensor(1.0, device=device)

        self.selected_label = torch.full(
            (num_unlabeled,), -1, dtype=torch.long, device=device
        )
        self.classwise_acc = torch.zeros(num_classes, device=device)

        self.cg_queue_size = int(cg_queue_size)
        self.cg_warmup_steps = int(cg_warmup_steps)
        self.cg_lambda_h = float(cg_lambda_h)
        self.cg_history: Dict[int, deque] = {}
        self.cg_count_gap: Dict[int, int] = {}
        self.cg_count_gap_mu = None

    @torch.no_grad()
    def _distribution_align(self, probs: torch.Tensor) -> torch.Tensor:
        batch_mean = probs.mean(dim=0)
        self.p_model.mul_(self.ema_p).add_(batch_mean, alpha=1.0 - self.ema_p)
        aligned = probs * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        return aligned / aligned.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def _softmatch_weight(self, probs: torch.Tensor) -> torch.Tensor:
        max_probs = probs.max(dim=-1).values
        batch_mu = max_probs.mean()
        batch_var = max_probs.var(unbiased=max_probs.numel() > 1)
        self.soft_mu.mul_(self.ema_p).add_(batch_mu, alpha=1.0 - self.ema_p)
        self.soft_var.mul_(self.ema_p).add_(batch_var, alpha=1.0 - self.ema_p)
        numerator = torch.clamp(max_probs - self.soft_mu, max=0.0).pow(2)
        denominator = (2.0 * self.soft_var / (self.n_sigma**2)).clamp_min(1e-12)
        return torch.exp(-numerator / denominator)

    @torch.no_grad()
    def _flexmatch_mask(self, probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        max_probs, pseudo = probs.max(dim=-1)
        threshold = self.p_cutoff * (
            self.classwise_acc[pseudo] / (2.0 - self.classwise_acc[pseudo])
        )
        mask = max_probs.ge(threshold).float()
        select = max_probs.ge(self.p_cutoff)
        if select.any():
            self.selected_label[indices[select]] = pseudo[select]

        valid = self.selected_label.ge(0)
        counts = torch.bincount(
            self.selected_label[valid], minlength=self.num_classes
        ).float()
        unused = (~valid).sum().float()
        denominator = torch.maximum(counts.max(), unused).clamp_min(1.0)
        self.classwise_acc.copy_(counts / denominator)
        return mask

    @torch.no_grad()
    def _freematch_mask(self, probs: torch.Tensor) -> torch.Tensor:
        max_probs, pseudo = probs.max(dim=-1)
        quantile = torch.quantile(max_probs[:, None], 0.8)
        self.time_p.mul_(self.ema_p).add_(quantile, alpha=1.0 - self.ema_p)
        self.p_model.mul_(self.ema_p).add_(probs.mean(dim=0), alpha=1.0 - self.ema_p)
        hist = torch.bincount(pseudo, minlength=self.num_classes).to(probs.dtype)
        hist = hist / hist.sum()
        self.label_hist.mul_(self.ema_p).add_(hist, alpha=1.0 - self.ema_p)
        modifier = self.p_model / self.p_model.max()
        return max_probs.ge(self.time_p * modifier[pseudo]).float()

    def _cg_row(self, index: int) -> Iterable[int]:
        history = self.cg_history.get(index)
        used = [] if history is None else list(history)
        return [0] * (self.cg_queue_size - len(used)) + used

    @torch.no_grad()
    def _cg_update_and_mask(
        self,
        probs: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_probs, pseudo = probs.max(dim=-1)
        for index, label in zip(indices.tolist(), pseudo.tolist()):
            history = self.cg_history.setdefault(
                int(index), deque(maxlen=self.cg_queue_size)
            )
            history.append(int(label))

        gaps = []
        for index in indices.tolist():
            counts = Counter(self._cg_row(int(index)))
            ordered = sorted(counts.values(), reverse=True)
            gap = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
            self.cg_count_gap[int(index)] = gap
            gaps.append(gap)
        gap_tensor = torch.tensor(gaps, dtype=probs.dtype, device=probs.device)

        batch_mu = max_probs.mean()
        self.soft_mu.mul_(self.ema_p).add_(batch_mu, alpha=1.0 - self.ema_p)
        if self.cg_count_gap_mu is None:
            self.cg_count_gap_mu = torch.tensor(
                sum(self.cg_count_gap.values()) / len(self.cg_count_gap),
                dtype=probs.dtype,
                device=probs.device,
            )
        else:
            self.cg_count_gap_mu.mul_(self.ema_p).add_(
                gap_tensor.mean(), alpha=1.0 - self.ema_p
            )
        easy = max_probs.ge(self.soft_mu).float()
        ambiguous = gap_tensor.ge(self.cg_count_gap_mu).logical_and(
            max_probs.lt(self.soft_mu)
        ).float()
        return easy, ambiguous, pseudo, gap_tensor

    def loss(
        self,
        step: int,
        logits_l: torch.Tensor,
        labels_l: torch.Tensor,
        logits_w: torch.Tensor | None,
        logits_s: torch.Tensor | None,
        indices: torch.Tensor | None,
        lambda_u: float,
    ) -> tuple[torch.Tensor, dict]:
        supervised = F.cross_entropy(logits_l, labels_l)
        zero = supervised.new_zeros(())
        metrics = {
            "supervised_loss": float(supervised.detach()),
            "unsupervised_loss": 0.0,
            "mask": 0.0,
            "ambiguous_mask": 0.0,
            "entropy_loss": 0.0,
        }

        if self.method == "cgmatch" and step < self.cg_warmup_steps:
            return supervised, metrics
        if logits_w is None or logits_s is None:
            raise ValueError("Unlabeled logits are required after warmup")

        raw_probs = logits_w.detach().softmax(dim=-1)
        if self.method == "fixmatch":
            confidence, pseudo = raw_probs.max(dim=-1)
            mask = confidence.ge(self.p_cutoff).float()
            unsupervised = masked_ce(logits_s, pseudo, mask)
            total = supervised + lambda_u * unsupervised

        elif self.method == "softmatch":
            aligned = self._distribution_align(raw_probs)
            mask = self._softmatch_weight(aligned)
            pseudo = raw_probs.argmax(dim=-1)
            unsupervised = masked_ce(logits_s, pseudo, mask)
            total = supervised + lambda_u * unsupervised

        elif self.method == "flexmatch":
            if indices is None:
                raise ValueError("FlexMatch needs persistent unlabeled indices")
            pseudo = raw_probs.argmax(dim=-1)
            mask = self._flexmatch_mask(raw_probs, indices)
            unsupervised = masked_ce(logits_s, pseudo, mask)
            total = supervised + lambda_u * unsupervised

        elif self.method == "freematch":
            pseudo = raw_probs.argmax(dim=-1)
            mask = self._freematch_mask(raw_probs)
            unsupervised = masked_ce(logits_s, pseudo, mask)
            entropy = freematch_entropy_loss(
                mask, logits_s, self.p_model, self.label_hist
            )
            total = supervised + lambda_u * unsupervised + self.entropy_ratio * entropy
            metrics["entropy_loss"] = float(entropy.detach())

        else:
            if indices is None:
                raise ValueError("CGMatch needs persistent unlabeled indices")
            aligned = self._distribution_align(raw_probs)
            easy, ambiguous, pseudo, gaps = self._cg_update_and_mask(aligned, indices)
            easy_loss = masked_ce(logits_s, pseudo, easy)
            ambiguous_loss = generalized_ce(logits_w, pseudo, ambiguous)
            ambiguous_loss = ambiguous_loss + generalized_ce(logits_s, pseudo, ambiguous)
            progress = (step - self.cg_warmup_steps) / max(
                1, self.total_steps - self.cg_warmup_steps
            )
            hard_weight = self.cg_lambda_h * progress**2
            unsupervised = easy_loss + hard_weight * ambiguous_loss
            total = supervised + lambda_u * easy_loss + hard_weight * ambiguous_loss
            mask = easy
            metrics["ambiguous_mask"] = float(ambiguous.mean())
            metrics["count_gap"] = float(gaps.mean())
            metrics["hard_weight"] = float(hard_weight)

        metrics["unsupervised_loss"] = float(unsupervised.detach())
        metrics["mask"] = float(mask.float().mean())
        return total, metrics

    def state_dict(self) -> dict:
        state = {
            "p_model": self.p_model.detach().cpu(),
            "p_target": self.p_target.detach().cpu(),
            "label_hist": self.label_hist.detach().cpu(),
            "time_p": self.time_p.detach().cpu(),
            "soft_mu": self.soft_mu.detach().cpu(),
            "soft_var": self.soft_var.detach().cpu(),
            "selected_label": self.selected_label.detach().cpu(),
            "classwise_acc": self.classwise_acc.detach().cpu(),
            "cg_count_gap_mu": None
            if self.cg_count_gap_mu is None
            else self.cg_count_gap_mu.detach().cpu(),
            "cg_history": {key: list(value) for key, value in self.cg_history.items()},
            "cg_count_gap": self.cg_count_gap,
        }
        return state
