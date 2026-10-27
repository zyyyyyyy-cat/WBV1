# B-B V1.0 implementation and review notes

## Experiment matrix

| Config | Local metric objective | Positive budget |
|---|---|---|
| `configs/UAD.yml` | Original WRT | Original WRT |
| `configs/BA.yml` | Stable-negative WRT | Original WRT |
| `configs/BB0.yml` | B-B pairwise ranking | Uniform |
| `configs/BB1.yml` | B-B pairwise ranking | Query-risk adaptive |

The BA/BB configs fix the documented four-GPU effective sampler batch of 256 and base learning rate of 0.032. For a single GPU, override these with `SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008`. All configs retain the UAD backbone, ProCA, GPD, 120 epochs, preprocessing, and inference settings. `UAD.yml` is untouched.

## Model and loss

The model remains one ViT-B/16 followed by a BNNeck and identity classifier. Training uses the original BN-before 768D CLS for WRT or B-B, ProCA and GPD. Inference still returns the BN-before CLS and the existing evaluator normalizes it. No trainable layers, teacher, extra bank, or inference computation were added.

For each local batch, B-B L2-normalizes the FP32 CLS and computes its cosine similarity matrix. Positive and negative candidates have a different camera ID from the query; the query is excluded. Invalid queries contribute zero. Positive weights sum to one. BB0 assigns uniform positive weights; BB1 gives the best positive an extra budget up to `alpha_max * sigmoid(-(best_positive - best_negative)/risk_tau)`. Negative images are grouped by PID. Each negative PID receives a uniform/risk softmax mixture based on its maximum image similarity; its images receive a uniform/hardness softmax mixture. The resulting negative weights sum to one. All budgets are detached. The differentiable objective is the weighted sum of `loss_tau * softplus((margin + negative_similarity - positive_similarity)/loss_tau)` over every valid positive-negative image pair. B-B replaces WRT exactly once in `CE + lambda_B * B-B + 0.01 ProCA + GPD`.

Each rank computes a sum over its valid queries. The global valid query count is all-reduced; the per-rank sum is multiplied by `world_size / global_count` to account for DDP gradient averaging. Empty-query ranks keep a graph-connected zero. No image features are gathered across GPUs.

## What was carried over from the references

- **AGW:** Its WRT connects identity CE to the normalized classification feature and metric learning to the feature before BNNeck. Its component comparison shows WRT can help retrieval, especially coverage-sensitive mINP, without changing the classifier's job. B-B therefore occupies the existing WRT position on UAD's BN-before CLS, while CE remains on BN-after features. B-A changes only WRT's negative weight normalization so its result can be interpreted separately from a new ranking objective.
- **MINER:** Its multi-positive query/gallery objective and negative-identity mining act together; its ablation shows negative mining is a reliable contribution across its tested datasets, while hard-positive gains depend on intra-person variation. Its deterministic hardest-case ablation also cautions against giving a single image all of the training signal. B-B uses the maximal negative image only to estimate each identity's risk, then mixes identity and within-identity softmax budgets with uniform coverage. BB0 versus BB1 isolates the additional top-positive budget. The original PKM sampler is retained so sampling changes cannot confound this first comparison.
- **SupRank:** Its rank surrogate and decomposability objective have separately measurable benefits, especially when a small batch poorly represents the full gallery. B-B is a lighter pairwise surrogate, not SupRank or an AP guarantee. UAD's existing GPD already supplies a global prototype-based objective, so this first experiment does not add another proxy/decomposability loss. Report full-gallery mAP, mINP and directed protocols before claiming retrieval gains.

These observations motivate the placement and comparison design; results reported in those papers are from different datasets and do not establish B-B's performance on WHU-MARS.

## Mandatory calibration on the training machine

### PyTorch 1.12.1+cu113 environment

The original `requirements.txt` pins torch 2.2.2 and torchvision 0.17.2, so do not install it into the stated server environment. PyTorch's official version table pairs torch 1.12.1+cu113 with torchvision 0.13.1+cu113. Verify the server's Python is 3.8–3.10 for the versions in this ancillary list, then verify `torch`, `torchvision` and CUDA imports before installing `requirements-torch112.txt`. The custom ViT does not depend on modern timm model APIs; this list selects timm 0.6.13 for the existing `timm.data.random_erasing` import. Run `python -m pip install -r requirements-torch112.txt` only after the torch/torchvision pair is correct, and verify the pair again afterward. The new B-B code uses the existing `torch.cuda.amp` API rather than the newer `torch.amp` API. Actual CUDA execution on torch 1.12.1 still needs verification on the server.

```bash
python -c "import torch, torchvision, timm; from timm.data.random_erasing import RandomErasing; print(torch.__version__, torchvision.__version__, timm.__version__, torch.cuda.is_available())"
```

No WHU-MARS data, ImageNet initialization weights, or completed training weights were present in this workspace. Therefore `BB0.yml` and `BB1.yml` deliberately have `BB_LOSS_WEIGHT: 0.0`; `make_loss` refuses to train B-B until a measured positive value is supplied. The calibration script uses 12 actual PKM batches with the same initialization and augmentation, does not step an optimizer, and compares BN-before CLS gradients from original WRT, stable WRT, BB0 and GPD. It proposes `median(stable_WRT_gradient) / median(BB0_gradient)`. Inspect its valid-query coverage and original-versus-stable WRT results before entering the same number in both BB configs.

Four GPU example, from the `WHU` directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  calibrate_bb.py --config_file configs/BB0.yml MODEL.DIST_TRAIN True
```

After entering the measured coefficient into both configs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/BB0.yml MODEL.DIST_TRAIN True
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/BB1.yml MODEL.DIST_TRAIN True
```

For WHU-MARS-2337, additionally set `DATASETS.DATASET_DIR WHU-MARS-2337` and a separate `OUTPUT_DIR`. Use the 1000 checkpoint for 1000-GD evaluation with `DATASETS.FILTER_GD True`; do not train a separate GD model. Keep seed, batch, initialization and coefficient the same across BB0 and BB1.

## File changes

| File | Change and purpose |
|---|---|
| `loss/bb_ranking_loss.py` | Adds the complete training-only B-B candidate masks, budgets, query loss and diagnostics. |
| `loss/wrt_loss.py` | Adds a separate BA class with a stable valid-negative softmax. Original WRT remains callable. |
| `loss/make_loss.py` | Selects WRT, BA or B-B, applies B-B weight once, and globally normalizes valid queries. |
| `processor/processor.py` | Aligns concatenated camera IDs with CLS/PIDs and logs B-B diagnostics. |
| `config/defaults.py` | Declares B-B hyperparameters and an uncalibrated zero default coefficient. |
| `configs/BA.yml`, `BB0.yml`, `BB1.yml` | Separate four-GPU comparison configurations. |
| `calibrate_bb.py` | Computes the fixed coefficient from real unmodified batches. |
| `requirements-torch112.txt` | Ancillary dependency set for the stated PyTorch 1.12.1 CUDA environment. |
| `requirements.txt` | Labels the pre-existing PyTorch 2.2 dependency set to prevent mixing it into the 1.12 environment. |
| `BB_IMPLEMENTATION.md` | Records the implementation, experiment sequence and review details. |

No files were deleted. `model/make_model.py`, datasets, inference and evaluator were not changed.

## Verification and limitations

Syntax and configuration parsing passed on Python 3.9.13. On local PyTorch 2.8 CPU, synthetic tests covered analytic one-positive/one-negative loss, camera masks, empty queries, budget normalization on equal-score inputs, 192-image forward/backward, and separate CE/B-B/ProCA/GPD gradient routing through a toy backbone and BNNeck. A batch-vectorized B-B implementation matched the original per-query implementation's loss, CLS gradient and diagnostics within floating-point tolerance for both BB0 and BB1. The DDP scaling formula was reviewed algebraically; a local multi-process runtime test could not start because this Windows PyTorch build's rendezvous requested unavailable libuv. Full CUDA/NCCL execution on torch 1.12.1, actual valid-query coverage, calibration and retrieval metrics require the server's real dataset and environment. The proposed temperature and budget values are first-run settings from the design, not measured optima. A pairwise surrogate encourages the desired order but does not guarantee every retrieval metric improves.
