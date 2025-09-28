
## ------------------------------------------------------------------------------------- #
library(dplyr)
library(rstatix) # dunn_test(); wilcox_test(); add_xy_position()
library(rlang)
library(ggpubr)


## ------------------------------------------------------------------------------------- #

## https://www.datanovia.com/en/lessons/kruskal-wallis-test-in-r/
##' The above function is the same as the below, while the latter can `add_xy_position` by
##'   each MH indicator, and the y.position can better represent the value of each indicator
##'   gtoup 

func_test_dif_dunn2 <- function(
    df,
    value     = "value",
    group     = "group",
    facet_by  = "ind_sub",
    which_test = c("dunn","wilcoxon"),
    ind_levels = NULL,
    out_dir   = NULL,
    add_mean  = F,
    tag       = ""
    ) {
  
  which_test <- match.arg(which_test)
  
  # collect pairwise results by facet
  res <- lapply(unique(df[[facet_by]]), function(ind) {
    dat <- df %>%
      dplyr::filter(.data[[facet_by]] == ind) %>%
      dplyr::transmute(
        Value = .data[[value]],
        Group = .data[[group]]
      )
    
    if (which_test == "wilcoxon") {
      pwc <- dat %>%
        rstatix::wilcox_test(Value ~ Group, p.adjust.method = "bonferroni")
    } else {
      pwc <- dat %>%
        rstatix::dunn_test(Value ~ Group, p.adjust.method = "bonferroni")
    }
    
    pwc %>%
      dplyr::mutate(!!facet_by := ind) %>%
      rstatix::add_xy_position(data = dat, x = "Group", fun = "median_iqr") %>%
      dplyr::mutate(xmin = .data$group1, xmax = .data$group2)
  })
  
  dunn_results <- dplyr::bind_rows(res)
  
  if (!is.null(ind_levels)) {
    dunn_results[[facet_by]] <- factor(dunn_results[[facet_by]], levels = ind_levels)
  }
  
  if (!is.null(out_dir)) {
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    readr::write_csv(
      dunn_results,
      file.path(out_dir, sprintf("%s_dunn_results.csv", tag))
    )
  }
  
  # dunn_results
  
  
  # Create the plot with ggplot2 ------------------------------------------------------- -
  # Visualization: box plots with p-values
  
  ##' 1. if `facet` plot
  
  if (!is.null(facet_by)) {
    
    p <- 
      # ggpubr::ggboxplot(
      ggpubr::ggbarplot(
        df, 
        x = group, y = value, 
        # fill = group,
        fill = 'gray',
        
        # add = "jitter", 
        # add.params = list(shape=1, size=0.5, alpha=0.5), 
        # add = "mean_se", width = 0.7,
        
        size = 0.2) +
      ggpubr::stat_pvalue_manual(dunn_results, step.increase = 0.06, color = 'gray50', hide.ns = TRUE) +  # Add p-values
      scale_y_continuous(expand = expansion(mult = c(0.05, 0.08))) +  # Add 8% space at the top
      # scale_fill_manual(values = color_bygroup) +
      facet_wrap(~ get(facet_by), scales = "free_y") +  # Facet by group
      theme_bw(base_size = text_size) +
      # labs(x = 'group_title', y = "Mean effect sizes") +
      theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
      theme(legend.position = "none",
            text = element_text(size = text_size),
            axis.text = element_text(size = text_size),
            strip.text = element_text(size = text_size)
      )
    p
    
  } else {
    p <- 
      ggboxplot(
        df, 
        x = group, y = value, fill = 'gray'
        ) +
      stat_pvalue_manual(test_comb, color = 'gray50', hide.ns = TRUE) +
      scale_fill_manual(values = color_bygroup) +
      theme_bw(base_size = text_size) +
      theme(legend.position="none",
            text = element_text(size = text_size), 
            axis.text = element_text(size = text_size),
            strip.text = element_text(size = text_size)
      )
  }
  
  
  ##' 2. if add `mean` values to the boxplot
  ##'    To create dose-response curve by add mean values as red dots and a line connecting them
  if (add_mean == T) {
    p <- p + 
      # Add mean as red dots
      stat_summary(fun = mean, geom = "point", 
                   position = position_dodge(width = 0.75),
                   shape = 18, size = 2, alpha = 0.5, color = "red") +
      # Add line connecting means (across groups within each facet)
      stat_summary(fun = mean, geom = "line", aes(group = 1), 
                   position = position_dodge(width = 0.75),
                   color = "red", linewidth = 0.3) 
  }
  
  
  
  ## return the figure
  return(p)
}


