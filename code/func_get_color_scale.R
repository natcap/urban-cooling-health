


library(ggplot2)
library(sf)
library(scales)

# Function to determine color scale parameters
get_color_scale <- function(values) {
  finite_vals <- values[is.finite(values)]
  
  if (length(finite_vals) == 0) {
    # Degenerate case: nothing to plot
    return(list(
      vmin = -1.0,
      vmax = 1.0,
      scale_type = "diverging",
      # colors = c("#762a83", "#ffffbf", "#1b7837"),  # PRGn_r equivalent
      colors = c("#ca0020", "#ffffbf", "#0571b0"),  # PRGn_r equivalent
      midpoint = 0
    ))
  }
  
  vmin_raw <- min(finite_vals)
  vmax_raw <- max(finite_vals)
  
  # If everything is exactly zero, give a small span
  if (vmin_raw == 0.0 && vmax_raw == 0.0) {
    vmin_raw <- -1.0
    vmax_raw <- 1.0
  }
  
  # Mixed signs → diverging
  if (vmin_raw < 0 && vmax_raw > 0) {
    vmax_abs <- max(abs(vmin_raw), abs(vmax_raw))
    return(list(
      vmin = -vmax_abs,
      vmax = vmax_abs,
      scale_type = "diverging",
      # colors = c("#762a83", "#ffffbf", "#1b7837"),  # PRGn_r (purple-green reversed)
      colors = c("#ca0020", "#ffffbf", "#0571b0"),  # PRGn_r equivalent
      midpoint = 0
    ))
  }
  # All non-negative → sequential
  else if (vmin_raw >= 0) {
    vmax <- ifelse(vmax_raw <= 0, 1.0, vmax_raw)
    return(list(
      vmin = 0.0,
      vmax = vmax,
      scale_type = "sequential_positive",
      colors = c("#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", 
                 "#fc4e2a", "#e31a1c", "#bd0026", "#800026"),  # YlOrRd
      midpoint = NULL
    ))
  }
  # All non-positive → sequential
  else {
    vmax <- 0.0
    vmin <- vmin_raw
    if (vmax <= vmin) {
      vmax <- vmin + 1.0
    }
    return(list(
      vmin = vmin,
      vmax = vmax,
      scale_type = "sequential_negative",
      colors = rev(c("#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476",
                     "#41ab5d", "#238b45", "#006d2c", "#00441b")),  # Greens_r
      midpoint = NULL
    ))
  }
}

