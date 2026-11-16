# PraNet vs. PraNet + Dynamic Convolution
### Benchmark Study on Polyp Segmentation in Colonoscopy Images
**Course:** SEM 5 Machine Learning Project | **Institution:** IIT Patna | **Roll No:** 2401CS95  

---

## 📌 Executive Summary

Accurate automatic segmentation of polyps in colonoscopy imagery is crucial for the early detection and prevention of colorectal cancer. State-of-the-art architectures like **PraNet** (Parallel Reverse Attention Network) capture global context and refine boundary regions using Reverse Attention (RA) blocks. However, standard convolutions in these attention modules employ static filter weights that cannot dynamically adapt to extreme variations in lesion shape, texture, size, and endoscopic illumination.

In this project, we augment PraNet by replacing static convolutions in the critical Reverse Attention stages (stages 2 and 3) with **Dynamic Convolution ($K=4$)**. Dynamic Convolution aggregates $K$ parallel convolutional kernels weighted by sample-dependent softmax attention logits generated via a lightweight squeeze-and-excitation network.

Across three independent seeds ($0, 1, 2$) and five clinical benchmark datasets, Dynamic PraNet delivers substantial generalization gains on unseen, challenging datasets:
- **$+2.47\%$ Mean Dice** on **CVC-ColonDB** ($0.6984 \to 0.7231$)
- **$+2.28\%$ Mean Dice** on **ETIS-LaribPolypDB** ($0.5982 \to 0.6210$)
- With **$<1.5\%$** computational overhead in FLOPs and parameter count.

---

## 🗂️ Project Directory Structure

The project follows a modular, standardized structure keeping all core pipelines in `.ipynb` notebook format:

```
2401CS95_MLProject/
├── README.md                      # Comprehensive project documentation (this file)
├── requirements.txt               # Python package dependencies
│
├── configs/                       # Hyperparameter and dataset configurations
│   ├── baseline_config.json       # PraNet baseline training and evaluation hyperparameters
│   ├── dynamic_conv_config.json   # Dynamic convolution parameters (K=4, stages 2 & 3, temperature)
│   └── dataset_config.json        # Dataset definitions, paths, and normalization constants
│
├── data/                          # Dataset & backbone weight download scripts
│   ├── download_data.sh           # Bash download script (Linux / Mac / Kaggle)
│   └── download_data.py           # Cross-platform Python download script
│
├── notebooks/                     # Jupyter Notebooks
│   └── 2401CS95_MLProject.ipynb   # Complete pipeline: EDA, Data Prep, Multi-Seed Train, Test, Eval & Plots
│
├── results/                       # Experimental results, figures, tables, and weights
│   ├── plots/                     # High-resolution publication figures
│   │   ├── Figure_4_1_Dice_Per_Seed.png      # Dice score strip plot per seed across 5 test sets
│   │   ├── Figure_4_2_Qualitative.png          # Side-by-side segmentation visual comparison
│   │   ├── Figure_4_3_Attention_Weights.png    # Dynamic kernel softmax routing weights (pi) distribution
│   │   └── Figure_4_4_Error_By_Size.png        # Error analysis grouped by polyp size category
│   ├── tables/                    # CSV quantitative tables
│   │   ├── eval_summary.csv                    # Dataset-level Mean Dice and Mean IoU
│   │   └── detailed_image_metrics.csv          # Per-image metrics (Dice, IoU, Recall, Precision, Size)
│   └── attention_weights/         # Exported attention matrices (.npy)
│       ├── dyn_s0_Kvasir_attention.npy
│       ├── dyn_s1_Kvasir_attention.npy
│       └── dyn_s2_Kvasir_attention.npy
│
└── models/                        # Model checkpoints and training logs
    └── pranet_backup_s2/          # Checkpoint weights (.pth) and logs across all seeds
```

---

## 🔬 Methodology

### 1. Baseline: PraNet (Parallel Reverse Attention Network)
- **Backbone:** Res2Net-50 (`res2net50_v1b_26w_4s`), extracting multi-scale hierarchical feature maps ($f_2, f_3, f_4, f_5$).
- **Partial Decoder Component (PDC):** Aggregates high-level feature maps ($f_3, f_4, f_5$) into a global context saliency map $S_g$.
- **Reverse Attention (RA):** Progressively mines boundary details by subtracting intermediate predictions from foreground areas.

### 2. Proposed: PraNet + Dynamic Convolution
Standard convolution applies a fixed kernel $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$.  
In our dynamic model, we replace static convolutions in RA stages 2 and 3 with **DynamicConv2d**:
$$W(x) = \sum_{k=1}^K \pi_k(x) W_k$$
where:
- $\{W_1, \dots, W_K\}$ are $K=4$ learnable candidate convolution kernels.
- $\pi(x) = \text{Softmax}\left(\frac{\text{MLP}(\text{GAP}(x))}{\tau}\right)$ is an input-dependent routing weight vector computed via Global Average Pooling followed by a two-layer bottleneck MLP ($r=4$) with temperature $\tau=1.0$.
- The dynamic blend is evaluated efficiently in mini-batches via grouped convolution.

---

## 📊 Benchmark Datasets

| Dataset | Split | Images | Masks | Domain Type | Characteristics |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **TrainDataset** | Train | 1,450 | 1,450 | Mixed (Kvasir + ClinicDB) | Diverse colonoscopy images with binary masks |
| **Kvasir-SEG** | Test | 100 | 100 | In-Distribution / Seen | Standard polyp resolutions from Norwegian endoscopy |
| **CVC-ClinicDB** | Test | 62 | 62 | In-Distribution / Seen | Standard colonoscopy video frames |
| **CVC-300** | Test | 60 | 60 | Out-of-Distribution / Unseen | Clean frames from non-overlapping video sequences |
| **CVC-ColonDB** | Test | 380 | 380 | Out-of-Distribution / Unseen | 15 different colonoscopy sequences, variable lesions |
| **ETIS-LaribPolypDB**| Test | 196 | 196 | Out-of-Distribution / Unseen | Challenging early-stage, flat, and diminutive polyps |

---

## 📈 Quantitative Results

### 1. Multi-Seed Mean Dice ($\pm$ Std Dev)

Averaged across 3 independent runs (Seed 0, Seed 1, Seed 2):

| Dataset | Baseline Dice | Dynamic ($K=4$) Dice | $\Delta$ Dice | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: |
| **Kvasir** | $0.9015 \pm 0.0012$ | $0.8988 \pm 0.0021$ | $-0.0027$ | Neutral |
| **CVC-ClinicDB** | $0.9168 \pm 0.0015$ | $0.9234 \pm 0.0036$ | $+0.0066$ | Neutral |
| **CVC-300** | $0.9021 \pm 0.0025$ | $0.8974 \pm 0.0023$ | $-0.0047$ | Neutral |
| **CVC-ColonDB** | $0.6984 \pm 0.0031$ | **$0.7231 \pm 0.0028$** | **$+0.0247$** | **Yes ($p < 0.01$)** |
| **ETIS-LaribPolypDB**| $0.5982 \pm 0.0044$ | **$0.6210 \pm 0.0051$** | **$+0.0228$** | **Yes ($p < 0.01$)** |

### 2. Computational Complexity & Efficiency

| Model | Parameters | GFLOPs ($352 \times 352$) | Inference FPS (V100) | Relative Overhead |
| :--- | :---: | :---: | :---: | :---: |
| PraNet (Baseline) | 32.55 M | 13.12 | 62.4 FPS | Baseline |
| PraNet + DynamicConv ($K=4$) | 32.99 M | 13.18 | 58.1 FPS | **+1.35% Params / +0.46% FLOPs** |

---

## 🚀 How to Run the Project

### Prerequisites
Install all dependencies in your Python environment:
```bash
pip install -r requirements.txt
```

---

### Running the End-to-End Pipeline
The notebook **[`notebooks/2401CS95_MLProject.ipynb`](notebooks/2401CS95_MLProject.ipynb)** is the complete, self-contained project notebook that handles everything end-to-end:

1. **Environment & GPU Setup:** Checks GPU access, CUDA availability, and PraNet repository status.
2. **Data Download & Extraction:** Automatically downloads and unpacks Zenodo `PolypDataset.zip` with MD5 checksum verification (can also be downloaded via `bash data/download_data.sh` or `python data/download_data.py`).
3. **Exploratory Data Analysis (EDA):** Verifies file pairings, checks data dimensions across training and 5 test datasets, and checks for train/test data leakage.
4. **Backbone Acquisition:** Downloads and validates pretrained Res2Net-50 weights (`res2net50_v1b_26w_4s-3cf99910.pth`).
5. **Dynamic Convolution Architecture:** Defines `DynamicConv2d` ($K=4$, attention gating, temperature scaling) and verifies layer equivalence and gradient backprop.
6. **Multi-Seed Training:** Runs training for both Baseline PraNet and Dynamic PraNet across 3 random seeds ($0, 1, 2$).
7. **Inference & Scoring:** Evaluates models across all 5 benchmark test datasets, computing per-dataset and per-image Dice and IoU metrics.
8. **Complexity Profiling:** Computes FLOPs, parameter count, and inference FPS using `thop`.
9. **Statistical Significance Testing:** Computes 10,000-iteration paired bootstrap 95% confidence intervals.
10. **Report & Figure Generation:** Generates all qualitative and quantitative visualizations (`Figure_4_1` through `Figure_4_4`) saved in `results/plots/` and outputs metric tables (`eval_summary.csv` and `detailed_image_metrics.csv`) in `results/tables/`.

To run: Open **[`notebooks/2401CS95_MLProject.ipynb`](notebooks/2401CS95_MLProject.ipynb)** in Jupyter Notebook / JupyterLab / Kaggle / Google Colab and execute the cells sequentially.

---

## 📑 Key Figures

- **[Figure 4.1](results/plots/Figure_4_1_Dice_Per_Seed.png)**: Run-to-run consistency across seeds 0, 1, 2.
- **[Figure 4.2](results/plots/Figure_4_2_Qualitative.png)**: Visual segmentation comparison showing reduced false positives and sharper boundaries on CVC-ColonDB.
- **[Figure 4.3](results/plots/Figure_4_3_Attention_Weights.png)**: Dynamic attention weight boxplots confirming non-degenerate kernel utilization.
- **[Figure 4.4](results/plots/Figure_4_4_Error_By_Size.png)**: Error breakdown demonstrating maximum Dice gains on small and medium polyps.

---

## 📚 References
1. Fan, D. P., Ji, G. P., Zhou, T., Chen, G., Fu, H., Shen, J., & Shao, L. (2020). *PraNet: Parallel Reverse Attention Network for Polyp Segmentation*. MICCAI 2020.
2. Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., & Liu, Z. (2020). *Dynamic Convolution: Attention over Convolution Kernels*. CVPR 2020.
3. Gao, S. H., Cheng, M. M., Zhao, K., Zhang, X. Y., Yang, M. H., & Torr, P. (2021). *Res2Net: A new multi-scale backbone architecture*. IEEE TPAMI.
