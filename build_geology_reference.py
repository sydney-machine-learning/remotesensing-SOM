"""
Rebuild the ground-truth geology raster from the PUBLIC NSW Seamless Geology WFS.

Why this exists
---------------
The reference raster (`geology_reference_from_polygons.tif`) is not in the repo:
it was produced once, by hand, from a MinView export sitting on the original
author's laptop (`QGIS.ipynb` still points at `/Users/vicwang/...`). That makes
the paper's headline validation numbers **impossible for anyone to reproduce** -
including the authors, on a different machine.

This script removes that dependency. It pulls the same source geology straight
from the Geological Survey of NSW's public WFS, clipped to the EnMAP footprint,
and rasterises it onto the EnMAP grid. It needs no QGIS, no GDAL bindings and no
manual export - just `rasterio`, which is already a dependency.

Classes (matching the paper's 3-class simplified reference):
    1 = Kanimblan / metamorphic basement   (geology:basement_rock_units_wfs)
    2 = GAB - Great Australian Basin       (geology:rock_units_gab)
    3 = CSP - Cenozoic sedimentary province(geology:rock_units_csp)
    -9999 = no data

Burn order follows QGIS.ipynb and the README instruction ("bottom/oldest to
top/youngest"): basement, then GAB, then CSP last, so the youngest cover sits on
top where units overlap.

IMPORTANT CAVEAT
----------------
This is a faithful *reconstruction* from the public source, not a byte-copy of
the original author's file. The paper's class 1 is labelled "Kanimblan
Tablelands metamorphic facies"; the Seamless Geology WFS exposes the basement as
rock units (over this footprint: Koonenberry Belt / Tibooburra-Louth Zone,
Delamerian & Thomson orogens) and does not publish a separate metamorphic-facies
layer. The paper's own Section 3.1 describes exactly these basement inliers, so
the two should correspond closely - but numbers may differ slightly from the
published ones. Verify against the original raster if it ever turns up.

    .venv/Scripts/python.exe build_geology_reference.py
"""
import json
import ssl
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.features import rasterize

from src import config

WFS = "https://gs-seamless.geoscience.nsw.gov.au/geoserver/ows"
EPSG = "urn:ogc:def:crs:EPSG::32754"
# Fetched in a single request rather than paged: some of these layers have no
# primary key, and GeoServer rejects `startIndex` on them with
# "Cannot do natural order without a primary key". All layers are well under
# this cap over the EnMAP footprint (largest is ~865 features), and
# fetch_layer() hard-fails if the server ever reports more than we received.
MAX_FEATURES = 20000

# (class value, layer, label) - burned in this order, later overwrites earlier.
LAYERS = [
    (1, "geology:basement_rock_units_wfs", "Kanimblan / metamorphic basement"),
    (2, "geology:rock_units_gab", "GAB"),
    (3, "geology:rock_units_csp", "CSP"),
]
NODATA = -9999


def fetch_layer(layer, bbox):
    """Fetch every feature of `layer` intersecting bbox, in EPSG:32754."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Built by hand rather than with urlencode: GeoServer rejects the
    # percent-encoded colons in the EPSG URN with HTTP 400.
    url = (f"{WFS}?service=WFS&version=2.0.0&request=GetFeature"
           f"&typeNames={layer}&outputFormat=application/json"
           f"&srsName={EPSG}&count={MAX_FEATURES}"
           f"&bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{EPSG}")
    with urllib.request.urlopen(url, timeout=300, context=ctx) as r:
        gj = json.loads(r.read().decode("utf-8", "ignore"))

    feats = gj.get("features", [])
    matched = gj.get("numberMatched")
    # Never silently truncate the ground truth.
    if isinstance(matched, int) and len(feats) < matched:
        raise RuntimeError(
            f"{layer}: got {len(feats)} of {matched} features - raise MAX_FEATURES "
            f"(paging is unavailable: layer has no primary key)")
    return feats, matched


def main():
    if not config.ENMAP_TIF.exists():
        raise SystemExit(f"EnMAP image not found: {config.ENMAP_TIF}")

    # Grid definition straight from the EnMAP image, so the reference aligns
    # pixel-for-pixel with every method's output.
    with rasterio.open(config.ENMAP_TIF) as src:
        transform, width, height = src.transform, src.width, src.height
        crs, bounds = src.crs, src.bounds
    print(f"EnMAP grid: {height} rows x {width} cols | CRS {crs}")
    print(f"Bounds: {bounds}\n")

    out = np.full((height, width), NODATA, dtype=np.int16)
    bbox = (bounds.left, bounds.bottom, bounds.right, bounds.top)

    for value, layer, label in LAYERS:
        feats, matched = fetch_layer(layer, bbox)
        print(f"{label:34s} <- {layer}")
        print(f"   fetched {len(feats)} of {matched} features")
        shapes = [(f["geometry"], value) for f in feats if f.get("geometry")]
        if not shapes:
            print("   WARNING: no geometries, skipping"); continue
        rasterize(shapes, out=out, transform=transform, all_touched=False)
        print(f"   burned as class {value}\n")

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "int16", "crs": crs, "transform": transform, "nodata": NODATA,
        "compress": "lzw",
    }
    with rasterio.open(config.GEOLOGY_REF_TIF, "w", **profile) as dst:
        dst.write(out, 1)

    print("=" * 60)
    print(f"Wrote {config.GEOLOGY_REF_TIF}")
    total = out.size
    for v, _, label in LAYERS:
        n = int((out == v).sum())
        print(f"  class {v} {label:34s} {n:>9,} px ({100*n/total:5.2f}%)")
    n = int((out == NODATA).sum())
    print(f"  nodata {'':36s} {n:>9,} px ({100*n/total:5.2f}%)")
    print("\nNOTE: reconstructed from the public NSW Seamless Geology WFS - see the")
    print("caveat in this script's docstring before comparing to the published numbers.")


if __name__ == "__main__":
    main()
