"""Query weighted, identity budgeted image ranking on the original CLS space."""

import torch
from torch import nn
from torch.nn import functional as F


class BBRankingLoss(nn.Module):
    def __init__(self, pos_mode='adaptive', margin=0.05, risk_temperature=0.05,
                 weight_temperature=0.10, loss_temperature=0.05,
                 alpha_max=0.5, beta=0.5, rho=0.5, num_modalities=3):
        super().__init__()
        if pos_mode not in ('uniform', 'adaptive'):
            raise ValueError('BB_POS_MODE must be uniform or adaptive')
        if not all(0 < t < float('inf') for t in
                   (risk_temperature, weight_temperature, loss_temperature)):
            raise ValueError('B-B temperatures must be positive')
        if not 0 <= margin < float('inf'):
            raise ValueError('B-B margin must be finite and nonnegative')
        if not (0 <= alpha_max <= 1 and 0 <= beta <= 1 and 0 <= rho <= 1):
            raise ValueError('B-B budget fractions must be in [0, 1]')
        self.pos_mode = pos_mode
        self.margin = margin
        self.risk_temperature = risk_temperature
        self.weight_temperature = weight_temperature
        self.loss_temperature = loss_temperature
        self.alpha_max = alpha_max
        self.beta = beta
        self.rho = rho
        self.num_modalities = num_modalities

    def forward(self, features, pids, cameras):
        if features.ndim != 2 or pids.ndim != 1 or cameras.ndim != 1:
            raise ValueError('Expected features [N,D], pids [N], cameras [N]')
        if features.size(0) != pids.numel() or pids.numel() != cameras.numel():
            raise ValueError('B-B features, pids and cameras must have equal lengths')
        # Autocast is disabled only for ranking arithmetic. The float conversion
        # remains on the autograd path to the original BN-before CLS tensor.
        with torch.cuda.amp.autocast(enabled=False):
            z = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
            similarities = z @ z.t()
            same_camera = cameras[:, None].eq(cameras[None, :])
            same_pid = pids[:, None].eq(pids[None, :])
            positive = same_pid & ~same_camera
            negative = ~same_pid & ~same_camera
            pos_counts = positive.sum(1)
            valid = (pos_counts > 0) & negative.any(1)

            count = int(valid.sum().item())
            # Only the budget branch is detached. Scores in the pairwise branch
            # retain their path through normalization and the shared ViT.
            with torch.no_grad():
                scores = similarities.detach()
                masked_pos = scores.masked_fill(~positive, -1e9)
                masked_neg = scores.masked_fill(~negative, -1e9)
                best_pos_score, best_pos_index = masked_pos.max(dim=1)
                best_neg_score = masked_neg.max(dim=1).values
                gap = best_pos_score - best_neg_score
                if self.pos_mode == 'adaptive':
                    alpha = self.alpha_max * torch.sigmoid(-gap / self.risk_temperature)
                else:
                    alpha = torch.zeros_like(gap)
                alpha = alpha * valid.float()
                a = ((1 - alpha[:, None]) * positive.float()
                     / pos_counts.clamp(min=1)[:, None].float())
                a.scatter_add_(1, best_pos_index[:, None], alpha[:, None])

                # Group negatives by training PID. All tensors here are local to
                # the rank; no prototype bank or feature gather is used.
                ids = torch.unique(pids)
                group = negative[:, None, :] & pids[None, None, :].eq(ids[None, :, None])
                group_count = group.sum(dim=2)
                present = group_count > 0
                id_count = present.sum(dim=1)
                threats = scores[:, None, :].masked_fill(~group, -1e9).max(dim=2).values
                identity_probs = F.softmax(
                    (threats / self.weight_temperature).masked_fill(~present, -1e9), dim=1)
                c = ((1 - self.beta) * present.float()
                     / id_count.clamp(min=1)[:, None].float()
                     + self.beta * identity_probs * present.float())
                image_probs = F.softmax(
                    (scores[:, None, :] / self.weight_temperature).masked_fill(~group, -1e9),
                    dim=2)
                d = ((1 - self.rho) * group.float()
                     / group_count.clamp(min=1)[:, :, None].float()
                     + self.rho * image_probs * group.float())
                b = (c[:, :, None] * d).sum(dim=1)

            # Chunk the query axis so a 192-image local batch does not materialize
            # the full [N,N,N] pairwise autograd tensor at once.
            total = similarities.sum() * 0.0
            if count:
                for start in range(0, features.size(0), 16):
                    stop = min(start + 16, features.size(0))
                    error = (self.margin + similarities[start:stop, None, :]
                             - similarities[start:stop, :, None])
                    pair_loss = self.loss_temperature * F.softplus(
                        error / self.loss_temperature)
                    total = total + (pair_loss * a[start:stop, :, None]
                                     * b[start:stop, None, :]).sum()

            stats = {
                'valid_query_ratio': count / max(1, features.size(0)),
                'positive_count_mean': pos_counts[valid].float().mean().item() if count else 0.0,
                'negative_identity_count_mean': id_count[valid].float().mean().item() if count else 0.0,
                'top_gap_mean': gap[valid].mean().item() if count else 0.0,
                'top_error_ratio': (gap[valid] < 0).float().mean().item() if count else 0.0,
                'alpha_mean': alpha[valid].mean().item() if count else 0.0,
                'alpha_near_max_ratio': (alpha[valid] > self.alpha_max * 0.9).float().mean().item() if count and self.alpha_max else 0.0,
                'negative_identity_budget_max_mean': c[valid].max(dim=1).values.mean().item() if count else 0.0,
                'valid_query_count': count,
            }
            if self.num_modalities > 0 and features.size(0) % self.num_modalities == 0:
                for modality, block in enumerate(valid.chunk(self.num_modalities)):
                    stats['valid_query_ratio_modality_{}'.format(modality)] = block.float().mean().item()
            return total, count, stats
