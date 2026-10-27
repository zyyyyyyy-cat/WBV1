"""Measure one fixed B-B coefficient on real PKM batches without training."""

import argparse
import torch
import torch.distributed as dist
from torch.cuda import amp

from config import cfg
from datasets import make_dataloader
from loss.bb_ranking_loss import BBRankingLoss
from loss.wrt_loss import TripletLoss_WRT, TripletLoss_WRT_Fixed
from model import make_model
from processor.processor import GPD, PCA, PrototypeBank
from train import init_distributed, set_seed, setup_cuda_visible_devices


def grad_norm(loss, features):
    grad = torch.autograd.grad(loss, features, retain_graph=True)[0]
    return grad.float().norm().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_file', default='configs/BB0.yml')
    parser.add_argument('--batches', type=int, default=12)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError('--batches must be positive')
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_cuda_visible_devices(cfg)
    local_rank, rank, world_size = init_distributed(cfg, args)
    torch.cuda.set_device(local_rank)
    set_seed(cfg.SOLVER.SEED)
    loader, _, _, _, classes, cameras, views = make_dataloader(cfg)
    model = make_model(cfg, classes, cameras, views).to(local_rank).train()
    bb = BBRankingLoss(
        pos_mode='uniform', margin=cfg.MODEL.BB_MARGIN,
        risk_temperature=cfg.MODEL.BB_RISK_TAU,
        weight_temperature=cfg.MODEL.BB_WEIGHT_TAU,
        loss_temperature=cfg.MODEL.BB_LOSS_TAU,
        alpha_max=cfg.MODEL.BB_ALPHA_MAX,
        beta=cfg.MODEL.BB_BETA, rho=cfg.MODEL.BB_RHO,
        num_modalities=len(cfg.DATASETS.MODALITIES))
    wrt_original, wrt_fixed = TripletLoss_WRT(), TripletLoss_WRT_Fixed()
    bank = PrototypeBank(classes, 768, momentum=cfg.MODEL.GPD_MOMENTUM, device='cuda')
    rows = []
    for step, (imgs, target, camids) in enumerate(loader):
        if step >= args.batches:
            break
        imgs = [x.to(local_rank) for x in imgs]
        target = target.to(local_rank)
        camids = [x.to(local_rank) for x in camids]
        target_rep = target.repeat(len(imgs))
        camera_rep = torch.cat(camids)
        with torch.no_grad():
            with amp.autocast():
                _, raw_features, _ = model(imgs, target, camids)
        # Measure each objective against the same FP32 BN-before CLS leaf.
        # The actual trainer's GradScaler handles model-parameter gradients;
        # FP32 here avoids unscaled FP16 leaf-gradient underflow in calibration.
        features = raw_features.detach().float().requires_grad_(True)
        with amp.autocast(enabled=False):
            loss_original = wrt_original(features, target_rep)[0]
            loss_fixed = wrt_fixed(features, target_rep)[0]
            _, ids, centers = PCA(features, target, len(imgs), cfg.DATALOADER.NUM_INSTANCE)
            loss_gpd = GPD(features, target, bank, ids, centers, len(imgs))
        bb_sum, valid_count, stats = bb(features, target_rep, camera_rep)
        global_count = torch.tensor(valid_count, device=features.device)
        if world_size > 1:
            dist.all_reduce(global_count)
        if global_count.item() == 0:
            raise RuntimeError('No valid cross-camera query in calibration batch')
        bb_loss = bb_sum * (world_size / global_count.item())
        row = [grad_norm(x, features) for x in
               (loss_original, loss_fixed, bb_loss, loss_gpd)]
        if not all(torch.isfinite(torch.tensor(row[1:]))):
            raise RuntimeError('Non-finite fixed WRT/B-B/GPD gradient at batch {}'.format(step))
        coverage = [stats['valid_query_ratio'], stats['positive_count_mean']]
        coverage += [stats['valid_query_ratio_modality_{}'.format(i)]
                     for i in range(len(imgs))]
        rows.append(row + coverage)

    if len(rows) != args.batches:
        raise RuntimeError('Expected {} calibration batches, received {}'.format(args.batches, len(rows)))
    rows_tensor = torch.tensor(rows, device='cuda', dtype=torch.float32)
    if world_size > 1:
        size = torch.tensor([len(rows)], device='cuda')
        sizes = [torch.empty_like(size) for _ in range(world_size)]
        dist.all_gather(sizes, size)
        if any(int(x.item()) != len(rows) for x in sizes):
            raise RuntimeError('Ranks received different calibration batch counts')
        gathered = [torch.empty_like(rows_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, rows_tensor)
        rows_tensor = torch.cat(gathered)
    if rank == 0:
        medians = rows_tensor[:, 1:].median(dim=0).values.tolist()
        original_finite = rows_tensor[:, 0][torch.isfinite(rows_tensor[:, 0])]
        original_median = original_finite.median().item() if len(original_finite) else float('nan')
        labels = (['WRT_original', 'WRT_fixed', 'BB0', 'GPD',
                   'valid_query_ratio', 'positive_count_mean']
                  + ['valid_query_ratio_modality_{}'.format(i)
                     for i in range(len(cfg.DATASETS.MODALITIES))])
        for name, value in zip(labels, [original_median] + medians):
            print('{} median: {:.8g}'.format(name, value))
        if medians[0] <= 1e-12 or medians[1] <= 1e-12 or medians[3] < 0.5:
            raise RuntimeError('Invalid WRT/B-B gradient or insufficient valid queries; inspect sampling')
        print('Original WRT finite gradients: {}/{}'.format(len(original_finite), len(rows_tensor)))
        print('Suggested MODEL.BB_LOSS_WEIGHT: {:.8g}'.format(medians[0] / medians[1]))
        print('Use the same measured coefficient for BB0 and BB1; keep it fixed during training.')
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
