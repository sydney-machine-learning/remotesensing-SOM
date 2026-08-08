
import itertools

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src import config
from src.metrics import apply_mapping, map_clusters_to_classes

METHODS = {"SOM": "som", "k-means + PCA": "pca", "k-means + CA": "ca",
           "k-means + SA": "sa"}
ALPHA = 0.05


def mapped_predictions(ref_flat, valid_classes):
   
    out = {}
    for name, key in METHODS.items():
        p = config.OUTPUTS_DIR / f"{key}_labels_spatial.npy"
        if not p.exists():
            continue
        pred = np.load(p).flatten()
        mask = ~np.isnan(pred) & np.isin(ref_flat, valid_classes)
        mp = map_clusters_to_classes(ref_flat[mask].astype(int),
                                     pred[mask].astype(int))
        mapped = apply_mapping(pred[mask].astype(int), mp)
        out[name] = (mapped, mask)
    return out


def mcnemar_exact(correct_a, correct_b):

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def holm(pvals):
    
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def run_family(correct, label):
    print(f"\n=== {label}: pairwise McNemar (exact) ===")
    pairs = list(itertools.combinations(correct.keys(), 2))
    rows, praw = [], []
    for a, b in pairs:
        nb, nc, p = mcnemar_exact(correct[a], correct[b])
        acc_a = correct[a].mean(); acc_b = correct[b].mean()
        rows.append({"A": a, "B": b, "acc_A": acc_a, "acc_B": acc_b,
                     "b (A>B)": nb, "c (B>A)": nc, "p_raw": p})
        praw.append(p)
    adj = holm(np.array(praw))
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        r["significant"] = "YES" if pa < ALPHA else "no"
    df = pd.DataFrame(rows)
    print(df.to_string(index=False,
          formatters={"acc_A": "{:.3f}".format, "acc_B": "{:.3f}".format,
                      "p_raw": "{:.2e}".format, "p_holm": "{:.2e}".format}))
    return df


def main():
    if not config.GEOLOGY_REF_TIF.exists():
        raise SystemExit("Ground-truth raster missing - run build_geology_reference.py")
    import rioxarray as rxr
    ref = rxr.open_rasterio(str(config.GEOLOGY_REF_TIF)).squeeze().values.flatten()
    valid_classes = list(config.CLASS_NAMES.keys())

    mp = mapped_predictions(ref, valid_classes)
  
    common = None
    for _, (_, mask) in mp.items():
        common = mask if common is None else (common & mask)
    ref_common = ref[common].astype(int)
    print(f"Common comparison set: {common.sum():,} pixels")

    def realign(method):
        mapped, mask = mp[method]
        full = np.full(ref.shape, -1, dtype=int)
        full[mask] = mapped
        return full[common]

    preds = {m: realign(m) for m in mp}

  
    kani_pixels = ref_common == config.TARGET_CLASS
    correct_recall = {m: (preds[m][kani_pixels] == config.TARGET_CLASS)
                      for m in preds}
    print(f"\n(Kanimblan recall family restricted to {kani_pixels.sum():,} "
          f"true-Kanimblan pixels)")
    df1 = run_family(correct_recall, "Kanimblan RECALL (true-Kanimblan pixels only)")

    
    correct_all = {m: preds[m] == ref_common for m in preds}
    df2 = run_family(correct_all, "Overall 3-class accuracy")

    config.FIGURES_DIR.mkdir(exist_ok=True)
    df1.to_csv(config.FIGURES_DIR / "significance_kanimblan.csv", index=False)
    df2.to_csv(config.FIGURES_DIR / "significance_overall.csv", index=False)
    print(f"\nSaved -> {config.FIGURES_DIR}/significance_*.csv")
    print("\nHow to read this:")
    print("  * 'significant = YES' -> the difference is not noise (Holm p < 0.05).")
    print("  * BUT with ~10^5-10^6 pixels, even a trivial gap tests significant, so")
    print("    look at the effect size too: acc_A - acc_B is the real recall/accuracy")
    print("    difference. Report it as 'X points higher (p < .001)', letting the")
    print("    reader judge whether the size matters - do not lean on the p-value alone.")
    print("  * The pixels are spatially autocorrelated, so the effective sample is")
    print("    smaller than the raw count; treat these p-values as a lower bound on")
    print("    the true p (a spatial-block test would be even more conservative).")


if __name__ == "__main__":
    main()
