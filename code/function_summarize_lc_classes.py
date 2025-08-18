

import numpy as np
import pandas as pd

def summarize_lc_classes(arr, labels, nodata=None, px_area_m2=None, sort_by="class"):
    """
    Summarize class counts, proportions, and (optional) areas for a labeled raster.
    - arr: 2D numpy array of class codes
    - labels: dict {code: "label"}
    - nodata: value to exclude (or None)
    - px_area_m2: pixel area in m² (optional; requires projected CRS)
    - sort_by: "class" or "proportion"
    """
    if nodata is not None:
        mask = arr != nodata
    else:
        mask = np.ones_like(arr, dtype=bool)

    vals, counts = np.unique(arr[mask], return_counts=True)
    total = counts.sum()

    # Build DataFrame
    df = pd.DataFrame({
        "class_code": vals,
        "label": [labels.get(int(v), f"Class {int(v)}") for v in vals],
        "count": counts,
        "proportion": counts / total
    })
    df["percent"] = 100 * df["proportion"]

    if px_area_m2 is not None:
        df["area_m2"] = df["count"] * px_area_m2
        df["area_ha"] = df["area_m2"] / 10_000
        df["area_km2"] = df["area_m2"] / 1e6

    if sort_by == "proportion":
        df = df.sort_values("proportion", ascending=False)
    else:
        df = df.sort_values("class_code")

    # Nice rounding for display
    df["proportion"] = df["proportion"].round(6)
    df["percent"] = df["percent"].round(3)
    for col in ("area_m2", "area_ha", "area_km2"):
        if col in df.columns:
            df[col] = df[col].round(2)

    return df.reset_index(drop=True)


