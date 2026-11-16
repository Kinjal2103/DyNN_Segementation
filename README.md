# Does Dynamic Convolution Help Polyp Segmentation?
### A Controlled Study on PraNet's Reverse-Attention Modules
**Course:** SEM 5 Machine Learning Project (Dynamic Neural Networks) | **Institution:** IIT Patna | **Student:** Kinjal Agarwal | **Roll No:** 2401CS95

---

## Summary

This project tests whether replacing four 3×3 convolutions in PraNet's reverse-attention (RA) modules (stages 2 and 3: `ra3_conv2`, `ra3_conv3`, `ra2_conv2`, `ra2_conv3`) with **Dynamic Convolution** (K = 4) improves polyp segmentation. Both models were trained with seeds 0, 1, 2 under identical settings and evaluated on five benchmarks.

**Finding: no consistent improvement.**
- 3-seed mean Dice is **lower on 4 of 5 datasets** and only marginally higher on ETIS (+0.0048).
- The sign depends on the seed: seed 2 is worse on every dataset (e.g. -0.0849 on ColonDB, -0.1173 on ETIS).
- The dynamic model has larger seed-to-seed variance.
- Attention weights of the inspected layer stay in a narrow range (about 0.18 to 0.31), so the layer adapts little to individual inputs.
- Cost: +1.4% parameters, +0.005% MACs, about +7% latency (T4, single measurement).

---

## Project Structure

```
2401CS95_MLProject/
├── README.md
├── requirements.txt
├── configs/            # baseline_config.json, dynamic_conv_config.json, dataset_config.json
├── data/               # download_data.sh, download_data.py
├── notebooks/
│   └── 2401CS95_MLProject.ipynb   # Full pipeline: EDA, training, evaluation, plots
├── results/
│   ├── plots/          # Figure_4_1 to Figure_4_4
│   ├── tables/         # eval_summary.csv, detailed_image_metrics.csv
│   └── attention_weights/         # Exported attention matrices (.npy)
└── models/             # Checkpoints (.pth) and training logs
```

---

## Methodology

**Baseline:** PraNet with ImageNet-pretrained Res2Net-50, a Parallel Partial Decoder and three RA modules (stages 4, 3, 2). 32,547,319 parameters.

**Dynamic variant:** the four 3×3 convolutions (64 → 64 channels) are replaced by

$$W(x) = \sum_{k=1}^{4} \pi_k(x)\, W_k, \qquad \pi(x) = \text{Softmax}\big(W_2\,\text{ReLU}(W_1\, g(x)) / \tau\big), \quad \tau = 1$$

where $g(x)$ is global average pooling and $W_1, W_2$ are 1×1 convolutions ($C \to C/4 \to K$). Stage 4 is unchanged to keep the comparison controlled. This adds 446,736 parameters (32,994,055 total, +1.4%).

**Setup (identical for both models):** Kaggle, one Tesla T4 per run; PyTorch 2.11.0; 352×352 input; batch size 16; 19 epochs; Adam, lr 1e-4; weighted IoU + BCE loss; seeds 0, 1, 2; final checkpoint used; about 65 min per run. No hyperparameters were tuned (no validation split in PraNet's protocol).

**Data:** PolypDataset from Zenodo (record 5579392), MD5 verified.

| Dataset | Split | Images | Role |
| :--- | :--- | :---: | :--- |
| Kvasir-SEG (900) + CVC-ClinicDB (550) | Train | 1,450 | Training |
| Kvasir | Test | 100 | Seen source |
| CVC-ClinicDB | Test | 62 | Seen source |
| CVC-300 | Test | 60 | Unseen |
| CVC-ColonDB | Test | 380 | Unseen |
| ETIS-LaribPolypDB | Test | 196 | Unseen |

A similarity check shows **49 of 62 ClinicDB test images have a near-copy (cosine > 0.9) in the training set**, so ClinicDB is a weak generalization test.

---

## Results

Per-image metrics at threshold 0.5, averaged per dataset. Mean ± std (n − 1) is recomputed from the six final runs in `eval_summary.csv`.


### 3-seed mean ± std

| Dataset | Baseline Dice | Dynamic Dice | Δ Dice | Baseline IoU | Dynamic IoU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Kvasir | 0.9033 ± 0.0082 | 0.8996 ± 0.0075 | -0.0037 | 0.8501 ± 0.0079 | 0.8462 ± 0.0081 |
| CVC-ClinicDB | 0.9179 ± 0.0012 | 0.9132 ± 0.0223 | -0.0046 | 0.8690 ± 0.0039 | 0.8649 ± 0.0198 |
| CVC-300 | 0.9006 ± 0.0132 | 0.8841 ± 0.0136 | -0.0165 | 0.8302 ± 0.0171 | 0.8131 ± 0.0166 |
| CVC-ColonDB | 0.7043 ± 0.0093 | 0.6884 ± 0.0511 | -0.0160 | 0.6324 ± 0.0097 | 0.6194 ± 0.0487 |
| ETIS | 0.6455 ± 0.0428 | 0.6502 ± 0.0928 | +0.0048 | 0.5793 ± 0.0355 | 0.5870 ± 0.0810 |


### Comparison with published PraNet (mean Dice)

| Dataset | Published | Our baseline (seed 0) | Our baseline (3-seed) |
| :--- | :---: | :---: | :---: |
| Kvasir | 0.898 | 0.9022 | 0.9033 ± 0.0082 |
| CVC-ClinicDB | 0.899 | 0.9165 | 0.9179 ± 0.0012 |
| CVC-300 | 0.871 | 0.9048 | 0.9006 ± 0.0132 |
| CVC-ColonDB | 0.709 | 0.6976 | 0.7043 ± 0.0093 |
| ETIS | 0.628 | 0.5970 | 0.6455 ± 0.0428 |

Evaluation differences limit direct comparability. Recall, precision and polyp-size breakdowns show no consistent pattern; see the report (Tables 4.5 and 4.6).

---

## Key Findings

1. **No reliable gain:** dynamic Dice is lower on 4 of 5 datasets, with gaps comparable to or smaller than its standard deviation.
2. **Seed-dependent:** seeds 0 and 1 mostly favour the dynamic model, seed 2 is worse everywhere. Three seeds show a pattern, not a proven effect.
3. **Near-static attention:** weights for `ra3_conv2` (seed 0, Kvasir) stay within about 0.18 to 0.31.
4. **High noise on hard datasets:** the baseline alone ranges from 0.597 to 0.678 on ETIS across seeds, so single-seed comparisons can mislead.
5. **Cost is small but there is no benefit:** +1.4% parameters, about +7% latency.

## Limitations

Only three seeds and one dynamic configuration; no parameter-matched static baseline or extra ablations; no validation split; ClinicDB train/test overlap; attention analyzed for one layer, one seed and one dataset; polyp-size sample counts not reported; single latency measurement.

## Future Work

Vary K (2, 4, 8) and temperature annealing, compare against parameter-matched static models, extend attention analysis to all layers and seeds, and add validation-based model selection.

---

## How to Run

```bash
pip install -r requirements.txt
```

Open `notebooks/2401CS95_MLProject.ipynb` in Jupyter / Kaggle / Colab and run the cells in order. The notebook downloads and verifies the data (or use `data/download_data.sh` / `data/download_data.py`), runs EDA, defines `DynamicConv2d` with unit tests, trains both models for seeds 0, 1, 2, evaluates on all five test sets, profiles efficiency, computes bootstrap intervals, and writes figures to `results/plots/` and tables to `results/tables/`.

## Figures

- **Figure 4.1:** mean Dice per seed, baseline vs dynamic.
- **Figure 4.2:** qualitative example on CVC-ColonDB (seed 0).
- **Figure 4.3:** softmax attention weights of `ra3_conv2` (seed 0, Kvasir).
- **Figure 4.4:** Dice by polyp size, pooled over all five datasets (the title says "unseen" and should be corrected when regenerated).

