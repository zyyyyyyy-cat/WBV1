# encoding: utf-8

import torch.nn.functional as F
import torch
import torch.distributed as dist
import math
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .wrt_loss import TripletLoss_WRT, TripletLoss_WRT_Fixed
from .bb_ranking_loss import BBRankingLoss
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    metric_type = cfg.MODEL.METRIC_LOSS_TYPE.lower()
    if metric_type == 'triplet':
        triplet = TripletLoss()
        print("using soft triplet loss for training")
    elif metric_type == 'wrt':
        triplet = TripletLoss_WRT()
        print("using WRT loss for training")
    elif metric_type == 'wrt_fixed':
        triplet = TripletLoss_WRT_Fixed()
        print('using B-A stable WRT loss for training')
    elif metric_type == 'bb':
        triplet = BBRankingLoss(
            pos_mode=cfg.MODEL.BB_POS_MODE,
            margin=cfg.MODEL.BB_MARGIN,
            risk_temperature=cfg.MODEL.BB_RISK_TAU,
            weight_temperature=cfg.MODEL.BB_WEIGHT_TAU,
            loss_temperature=cfg.MODEL.BB_LOSS_TAU,
            alpha_max=cfg.MODEL.BB_ALPHA_MAX,
            beta=cfg.MODEL.BB_BETA,
            rho=cfg.MODEL.BB_RHO,
            num_modalities=len(cfg.DATASETS.MODALITIES),
        )
        if not math.isfinite(cfg.MODEL.BB_LOSS_WEIGHT) or cfg.MODEL.BB_LOSS_WEIGHT <= 0:
            raise ValueError('Set MODEL.BB_LOSS_WEIGHT to the calibrated positive value before B-B training')
        print('using B-B ranking loss for training')
    else:
        raise ValueError('unknown METRIC_LOSS_TYPE: {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler.upper() != 'PKM':
        raise ValueError('expected PKM sampler, but got {}'.format(sampler))

    def loss_func(score, feat, target, cameras=None):
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            ID_LOSS = xent(score, target)
        else:
            ID_LOSS = F.cross_entropy(score, target)

        stats = {}
        if metric_type == 'bb':
            if cameras is None:
                raise ValueError('B-B requires camera IDs in feature order')
            local_sum, local_count, stats = triplet(feat, target, cameras)
            count = torch.tensor(local_count, device=feat.device, dtype=torch.long)
            ddp_on = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
            if ddp_on:
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            total_count = int(count.item())
            # DDP averages gradients, so multiply each local sum by world size.
            world_size = dist.get_world_size() if ddp_on else 1
            TRI_LOSS = local_sum * (world_size / total_count) if total_count else local_sum
            metric_weight = cfg.MODEL.BB_LOSS_WEIGHT
            stats['bb_raw'] = (local_sum.detach() / local_count).item() if local_count else 0.0
            stats['bb_weighted'] = (TRI_LOSS.detach() * metric_weight).item()
        else:
            TRI_LOSS = triplet(feat, target)[0]
            metric_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        return (cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + metric_weight * TRI_LOSS,
                ID_LOSS, TRI_LOSS, stats)

    return loss_func, center_criterion
