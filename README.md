# Unsupervised Learning Using SOM for Remote Sensing Mineral Prospectivity

A comparative study of topology-preserving (Self-Organizing Maps) versus non-topology-preserving methods (k-means with dimensionality reduction) for hyperspectral geological mapping in the Tibooburra region, NSW, Australia.

## 📋 Project Overview

This repository contains the implementation code for a thesis comparing four machine learning approaches for processing EnMAP hyperspectral satellite data:

1. **Self-Organizing Maps (SOM)** - Topology-preserving clustering
2. **k-means + PCA** - Principal Component Analysis for dimensionality reduction
3. **k-means + Canonical Autoencoder** - Neural network-based dimensionality reduction
4. **k-means + Stacked Autoencoder** - Deep learning approach with multiple encoder layers

**Research Focus:** Identifying Kanimblan metamorphic unit boundaries for gold exploration in the Tibooburra region.

## 📁 Repository Structure
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

## 🔧 Requirements
```bash
# Core libraries
numpy
matplotlib
rioxarray
scikit-learn
scipy
minisom

# Deep learning (for autoencoder methods)
torch
torchvision

# Geospatial
GDAL
rasterio

# Visualization
seaborn
```

Install dependencies:
```bash
pip install numpy matplotlib rioxarray scikit-learn scipy minisom torch GDAL rasterio seaborn
```

## Data

### Included in Repository:
- **Datasets/**: Sentinel-2 L2A multispectral imagery (12 bands, 10-20m resolution)

### Download Separately (Due to Size):
- **EnMAP Hyperspectral Image** (363.6 MB): 
  - File: `ENMAP01-____L2A-DT0000160657_20251016T011429Z_002_V010502_20251017T031330Z-SPECTRAL_IMAGE.TIF`
  - Download from: https://unsw-my.sharepoint.com/:u:/g/personal/z5042599_ad_unsw_edu_au1/Eb53i2FdyeVPhapjG6vh494BSfKrV1ZVBIOBl6jQ44zu-Q?e=pARiRh
  - Specifications: 219 valid spectral bands, 30m resolution, 1,044,103 valid pixels
  - Place in: `./Datasets/` directory

### Ground Truth Data:
- MinView simplified geological maps (shapefile format)
- Processed using `QGIS.ipynb` to generate rasterized ground truth
- **Important:** When running QGIS notebook, input shapefiles from bottom to top (youngest to oldest geological units)

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

### 4. Run Main Methods (Hyperspectral)

Each notebook processes the EnMAP hyperspectral image and saves results for evaluation:

**SOM Method:**
```bash
jupyter notebook SOM_Spectral.ipynb
```
- Outputs: `som_encoded_data.npy`, `som_labels_spatial.npy`

**PCA + k-means:**
```bash
jupyter notebook k-MeansPCA_Spectral.ipynb
```
- Outputs: `pca_encoded_data.npy`, `pca_labels_spatial.npy`

**Canonical Autoencoder + k-means:**
```bash
jupyter notebook k-MeansCA_Spectral.ipynb
```
- Outputs: `ca_encoded_data.npy`, `ca_labels_spatial.npy`

**Stacked Autoencoder + k-means:**
```bash
jupyter notebook k-MeansSA_Spectral.ipynb
```
- Outputs: `sa_encoded_data.npy`, `sa_labels_spatial.npy`

### 5. Comprehensive Evaluation
Run evaluation notebook after generating all method outputs:
```bash
jupyter notebook Evluation.ipynb
```
**Inputs:** All saved `.npy` files from above methods + ground truth `.tif`

**Outputs:**
- Confusion matrices
- Validation metrics (ARI, NMI, Purity, Cohen's Kappa, Calinski-Harabasz, Davies-Bouldin, Silhouette)
- Comparative visualisations
- Performance analysis plots

## Contributors

- **Author:** Cheng Wang (UNSW)
- **Supervisor:** Dr. Rohitash Chandra (UNSW)
- **Co-supervisor:** Dr. Ehsan Farahbakhsh
- **Industry Collaborator:** Sam Johnson (Datarock)

## Contact

For questions or collaboration opportunities:
- Email: [z5042599@unsw.edu.au]
- GitHub: [@Vi90195]

## Acknowledgments

- UNSW Faculty of Computer Science and Engineering
- EnMAP satellite mission for hyperspectral data
- NSW Seamless Geology dataset
- MiniSom library developers

---

**Note:** This is thesis research code. For production mineral exploration applications, additional validation and domain expert review is recommended.
