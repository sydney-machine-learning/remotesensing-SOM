# SOM-Exp: Self-Organising Maps For Hyperspectral Remote Sensing-Based Mineral Exploration

A comparative study of topology-preserving (Self-Organizing Maps) versus non-topology-preserving methods (k-means with dimensionality reduction) for hyperspectral geological mapping in the Tibooburra region, NSW, Australia.

## Project Overview

This repository contains the implementation code for a thesis comparing four machine learning approaches for processing EnMAP hyperspectral satellite data:

1. **Self-Organizing Maps (SOM)** - Topology-preserving clustering
2. **k-means + PCA** - Principal Component Analysis for dimensionality reduction
3. **k-means + Canonical Autoencoder** - Neural network-based dimensionality reduction
4. **k-means + Stacked Autoencoder** - Deep learning approach with multiple encoder layers

**Research Focus:** Identifying Kanimblan metamorphic unit boundaries for gold exploration in the Tibooburra region.

## Repository Structure
```
.
├── SimpleTut/                    # Tutorial folder with basic SOM examples
├── Datasets/                     # Multispectral Sentinel-2 data
├── SOM_Spectral.ipynb           # SOM clustering implementation
├── k-MeansPCA_Spectral.ipynb    # PCA + k-means implementation
├── k-MeansCA_Spectral.ipynb     # Canonical Autoencoder + k-means
├── k-MeansSA_Spectral.ipynb     # Stacked Autoencoder + k-means
├── QGIS.ipynb                   # Ground truth preparation
├── PCA & SOM.ipynb              # Initial comparison on multispectral data
├── Evluation.ipynb              # Comprehensive evaluation and metrics
└── README.md                    # This file
```

## Shared implementation (`src/`)

All four method notebooks and the evaluation notebook now import a shared
`src/` package so preprocessing, seeding, post-processing and metrics are
**identical** across methods (this removes the copy-paste drift that previously
made the comparison unfair):

| Module | Responsibility |
|---|---|
| `src/config.py` | Single source of truth: paths, global seed, all hyper-parameters |
| `src/seeding.py` | Seeds numpy / random / torch for reproducibility |
| `src/preprocessing.py` | Canonical EnMAP load + band validation + NaN handling (identical valid pixels for every method) |
| `src/pipeline.py` | The four methods as shared functions, so every script trains them identically |
| `src/autoencoders.py` | Canonical & stacked autoencoders (seeded, deterministic, early stopping) |
| `src/som_metrics.py` | The SOM's own metrics: quantization & topographic error |
| `src/postprocess.py` | Fast vectorised majority filter (replaces the slow per-window `generic_filter`) |
| `src/metrics.py` | Corrected external metrics + cluster→class mapping + intrinsic metrics |
| `src/results.py` | `finalize_method(...)`: same filter + same outputs for every method |
| `src/plotting.py` | Publication figure style + **validated** colourblind-safe palettes |

## Requirements

Everything is pinned in [`requirements.txt`](requirements.txt).

### Set up a virtual environment (Windows, PowerShell)
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# PyTorch CPU build (smaller/faster than the default CUDA wheel):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
> `rasterio` ships bundled GDAL wheels on Windows, so no separate GDAL install is needed.

## Data

### Included in Repository:
- **Datasets/**: Sentinel-2 L2A multispectral imagery (12 bands, 10-20m resolution)

### Download Separately (Due to Size):
- **EnMAP Hyperspectral Image** (363.6 MB): 
  - File: `ENMAP01-____L2A-DT0000160657_20251016T011429Z_002_V010502_20251017T031330Z-SPECTRAL_IMAGE.TIF`
  - Download from: https://unsw-my.sharepoint.com/:u:/g/personal/z5042599_ad_unsw_edu_au1/Eb53i2FdyeVPhapjG6vh494BSfKrV1ZVBIOBl6jQ44zu-Q?e=pARiRh
  - Specifications: 219 valid spectral bands, 30m resolution, 1,044,103 valid pixels
  - Place in: `./Datasets/` directory

### Ground Truth Data — now reproducible from a public source

The validation raster (`geology_reference_from_polygons.tif`) was originally made
by hand from a MinView export on the author's laptop (`QGIS.ipynb` still points at
`/Users/vicwang/...`), and was never committed — so **nobody could reproduce the
paper's headline numbers**, including the authors on another machine.

`build_geology_reference.py` removes that dependency. It pulls the same source
geology from the Geological Survey of NSW's **public WFS**, clipped to the EnMAP
footprint, and rasterises it straight onto the EnMAP grid — no QGIS, no GDAL
bindings, no manual export:

```powershell
python build_geology_reference.py     # -> geology_reference_from_polygons.tif
```

Source: `https://gs-seamless.geoscience.nsw.gov.au/geoserver/ows` (NSW Seamless
Geology). The WFS serves geometry natively in EPSG:32754 — the EnMAP CRS — so
there is no reprojection error. Classes: `1` = Kanimblan/metamorphic basement
(`geology:basement_rock_units_wfs`), `2` = GAB (`geology:rock_units_gab`),
`3` = CSP (`geology:rock_units_csp`), `-9999` = no data. Burn order is
oldest→youngest, matching `QGIS.ipynb`.

> **Caveat:** this is a faithful *reconstruction*, not a byte-copy of the original.
> The paper labels class 1 "Kanimblan Tablelands metamorphic facies", but the WFS
> publishes the basement as rock units (here: **Koonenberry Belt /
> Tibooburra-Louth Zone**, Delamerian & Thomson orogens) and exposes no separate
> metamorphic-facies layer. The paper's own §3.1 describes exactly these basement
> inliers, so they should correspond closely, but numbers may differ slightly from
> the published ones. (Worth checking with the authors: §3.1 places Tibooburra in
> the *Koonenberry Belt / Delamerian Orogen*, which the WFS independently
> confirms, while the class is labelled *Kanimblan Tablelands* — a Lachlan Orogen
> cycle.)

`QGIS.ipynb` is retained unchanged for reference.

## Usage

### 1. SimpleTut - Getting Started
Start here to understand basic SOM implementation using MiniSom:
```bash
jupyter notebook SimpleTut/
```

### 2. Initial Comparison (Multispectral)
Compare PCA+k-means vs SOM on Sentinel-2 data:
```bash
jupyter notebook "PCA & SOM.ipynb"
```

### 3. Prepare Ground Truth
Convert geological shapefile to raster format:
```bash
jupyter notebook QGIS.ipynb
```
**Note:** Input shapefiles in order from bottom (oldest) to top (youngest) geological layers.

### 4. Run the Four Methods (Hyperspectral)

Run each notebook top-to-bottom. Each one preprocesses the EnMAP image via
`src/`, clusters, and writes its outputs to `outputs/`:

| Notebook | Writes to `outputs/` |
|---|---|
| `SOM_Spectral.ipynb` | `som_labels_spatial.npy`, `som_labels_valid.npy`, `som_intrinsic.json` |
| `k-MeansPCA_Spectral.ipynb` | `pca_labels_spatial.npy`, `pca_labels_valid.npy`, `pca_intrinsic.json` |
| `k-MeansCA_Spectral.ipynb` | `ca_labels_spatial.npy`, `ca_labels_valid.npy`, `ca_intrinsic.json` |
| `k-MeansSA_Spectral.ipynb` | `sa_labels_spatial.npy`, `sa_labels_valid.npy`, `sa_intrinsic.json` |

Every method now saves its label map **already majority-filtered**, plus the
raw valid-pixel labels and the intrinsic metrics computed in its own feature
space. Tune anything (SOM grid, k, filter size, seed) in one place —
`src/config.py`.

### 5. Comprehensive Evaluation
After all four have run:
```bash
jupyter notebook Evluation.ipynb
```
**Inputs:** the `outputs/*.npy` + `outputs/*_intrinsic.json` files, and the
ground-truth raster `geology_reference_from_polygons.tif`.

**Outputs (written to `Figures/`):**
- `model_comparison_results.csv` — full metrics table
- `model_comparison_heatmap.png`, `model_comparison_maps.png`, `confusion_matrices.png`
- Metrics reported: **Overall Accuracy, Balanced Accuracy, Kappa (corrected), Kanimblan recall**, ARI, NMI, Purity, Homogeneity, Completeness, V-measure, Silhouette, CH, DB.

> The original notebooks (with their embedded figures used in the current draft
> of the paper) remain in git history — recover any with
> `git show <old-commit>:SOM_Spectral.ipynb`.

## Everything from the terminal (no Jupyter needed)

The notebooks are optional; these scripts run the same `src/` code headlessly.

```powershell
.venv\Scripts\Activate.ps1

python build_geology_reference.py    # ground truth from the public NSW WFS
python run_all_methods.py            # train all 4 methods  -> outputs/
python run_evaluation.py             # score vs geology     -> Figures/
python make_figures.py               # all paper figures    -> Figures/ (PNG + PDF)
```

### Additional experiments (each answers a reviewer question)

| Script | Question it answers | Output |
|---|---|---|
| `run_som_training_fix.py` | How much did the training bug cost? | `Figures/som_training_fix.csv` |
| `run_som_tuning.py` | Are the SOM hyper-parameters justified? (quantitative, not visual) | `Figures/som_tuning.csv` |
| `run_multiseed.py` | Error bars + **stability** (does the map change per seed?) | `Figures/multiseed_results.csv` |
| `run_cluster_selection.py` | Is 3×3 / k=4 justified? (QE-TE for SOM, silhouette/DB for k-means) | `Figures/som_grid_selection.csv`, `k_selection_*.csv` |
| `run_common_space_metrics.py` | Fair comparison: all methods scored in the **same** 219-D space | `Figures/common_space_intrinsic_metrics.csv` |
| `run_matched_k_comparison.py` | Fair comparison: SOM vs k-means at the **same** cluster count | `Figures/matched_k_comparison.csv` |
| `run_ablation_standardize.py` | Does standardising before PCA actually help? (**measured: no — it hurts**) | `Figures/ablation_standardize.csv` |

> **An ablation that changed our minds.** Standardising bands before PCA is the
> textbook default, so it was applied here at first. Measured in the common
> 219-D space it is *worse*: Silhouette 0.310 → 0.294, DB 0.991 → 1.038. All 219
> bands are already the same physical quantity (reflectance in [0,1]), so they
> are commensurable and z-scoring only inflates the noisy low-variance bands.
> The original code's choice was right; `config.PCA_STANDARDIZE = False`.

### Why the intrinsic metrics favour k-means (and why that isn't a SOM failure)

Silhouette / CH / DB all measure within-cluster compactness vs between-cluster
separation — **which is precisely the objective k-means explicitly minimises**.
The SOM optimises topology preservation instead, so scoring it that way is
structurally biased: k-means winning is close to a tautology, not evidence of
better geology. `run_matched_k_comparison.py` removes the feature-space and
cluster-count confounds; `src/som_metrics.py` supplies the SOM's own scoreboard.

## Paper Citation

Cheng Wang, Ayush Bansal, Ehsan Farahbakhsh, and Rohitash Chandra (2026). *SOM-Gold: Comparative Representation Learning Framework for Hyperspectral Mineral Exploration*. **Under review.**

## Contributors

- **Author:** Cheng Wang (UNSW)
- **Supervisor:** Dr. Rohitash Chandra (UNSW)
- **Co-supervisor:** Dr. Ehsan Farahbakhsh (USYD)
- **Industry Collaborator:** Sam Johnson (Datarock)

## Contact

For questions or collaboration opportunities:
- Email: [victorwang1995@hotmail.com]
- GitHub: [@Vi90195]

## Acknowledgments

- UNSW Faculty of Computer Science and Engineering
- EnMAP satellite mission for hyperspectral data
- NSW Seamless Geology dataset
- MiniSom library developers

---

**Note:** This is thesis research code. For production mineral exploration applications, additional validation and domain expert review is recommended.
