from pathlib import Path


SEED = 42


ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "Datasets"


ENMAP_TIF = DATASETS_DIR / (
    "ENMAP01-____L2A-DT0000160657_20251016T011429Z_002_V010502_"
    "20251017T031330Z-SPECTRAL_IMAGE.TIF"
)


GEOLOGY_REF_TIF = ROOT / "geology_reference_from_polygons.tif"


OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = ROOT / "Figures"


FILL_VALUE = -32768        
REFLECTANCE_SCALE = 10000.0 


SOM_GRID = (3, 3)          # final SOM lattice -> 9 clusters (paper's choice)
SOM_SIGMA = 0.5            # tuned in the paper's sensitivity analysis
SOM_LEARNING_RATE = 0.1


SOM_EPOCHS = 1
SOM_ITERATIONS = 1000      # retained only to reproduce the original behaviour

PCA_N_COMPONENTS = 10


PCA_STANDARDIZE = False

KMEANS_K = 4               # elbow-selected k for the k-means methods
KMEANS_N_INIT = 10


MAJORITY_FILTER_SIZE = 7


CLASS_NAMES = {1: "Kanimblan", 2: "GAB", 3: "CSP"}

TARGET_CLASS = 1  # Kanimblan
