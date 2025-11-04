

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def make_map_scale(values: np.ndarray,
                   pad: float = 0.05
                   ):
    """
    Given a 1D array of values (e.g. change = 2021 - 2020),
    return (vmin, vmax, cmap, norm, extend_mode) suitable for geopandas/plt plots.

    pad: fraction of data range to pad on each side (for sequential cases).

    Rules:
    - empty / all-nan → [-1, 1], diverging
    - mixed signs → symmetric diverging around 0
    - all ≥ 0 → sequential up from 0
    - all ≤ 0 → sequential down to 0
    """
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]

    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
        cmap = plt.cm.PRGn_r
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        return vmin, vmax, cmap, norm, "both"

    vmin_raw, vmax_raw = float(finite.min()), float(finite.max())

    # all zero → force small diverging span
    if vmin_raw == 0.0 and vmax_raw == 0.0:
        vmin_raw, vmax_raw = -1.0, 1.0

    # 1) mixed signs → diverging
    if (vmin_raw < 0) and (vmax_raw > 0):
        vmax_abs = max(abs(vmin_raw), abs(vmax_raw))
        vmin, vmax = -vmax_abs, vmax_abs
        cmap = plt.cm.PRGn_r
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        extend_mode = "both"

    # 2) all non-negative → sequential, start near actual min
    elif vmin_raw >= 0:
        data_range = vmax_raw - vmin_raw
        if data_range == 0:
            # flat but positive: give narrow band around value
            vmin = vmin_raw - 0.5
            vmax = vmax_raw + 0.5
        else:
            pad_abs = data_range * pad
            vmin = vmin_raw - pad_abs
            vmax = vmax_raw + pad_abs
        cmap = plt.cm.YlOrRd
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        extend_mode = "neither"

    # 3) all non-positive → sequential (reversed green)
    else:
        data_range = abs(vmin_raw - vmax_raw)
        if data_range == 0:
            vmin = vmin_raw - 0.5
            vmax = vmin_raw + 0.5
        else:
            pad_abs = data_range * pad
            vmin = vmin_raw - pad_abs
            vmax = vmax_raw + pad_abs
        cmap = plt.cm.Greens_r
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        extend_mode = "neither"

    return vmin, vmax, cmap, norm, extend_mode


# # Example usage: --------------------------------------------------------------------------

# vmin, vmax, cmap, norm, extend_mode = make_map_scale(gdf_base["var_change"].to_numpy())

# ax = gdf_base.plot(
#     column="var_change",
#     cmap=cmap,
#     norm=norm,
#     edgecolor="black",
#     linewidth=0.3,
#     figsize=(7, 7),
#     legend=False,
# )

# sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# sm._A = []
# cbar = plt.colorbar(sm, ax=ax, extend=extend_mode)
# cbar.set_label("Temperature change (°C)")

# ax.set_axis_off()
