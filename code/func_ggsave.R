
library(ggplot2)

func_ggsave <- function(fname, w = 7, h = 4, dpi = 300, unit = "in", save_png = TRUE) {
  if (save_png == T) {
    ggplot2::ggsave(filename = fname, 
                    plot = ggplot2::last_plot(), 
                    width = w, height = h, units = unit, dpi = dpi)
  } else {
    print('The plot will not be saved.')
  }
}