# Production Figure 3 builder, called by the original temperature notebook.
# Styling follows its signed labels and the shared Figure 4 scenario palette.
build_figure3_borough_reporting <- function(summary_dir, code_dir) {
  suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
  env <- new.env();sys.source(file.path(code_dir,"func_colors.R"),env)
  order <- c("baseline","allbuilt","treerisk","treeopp","green10","target10",
             "green20","target20","green30","target30")
  labels <- c("Baseline","AllBuilt","TreeRisk","TreeOpp","Green10","Target10",
              "Green20","Target20","Green30","Target30")
  colors <- env$scenario_colors[order];colors[1] <- "grey40"
  stem <- "figure3_main_v01_20261111"
  output_stem <- "figure3_main_v03_20261111"
  d <- read.csv(file.path(summary_dir,paste0(stem,"_conditions.csv")))
  b <- read.csv(file.path(summary_dir,paste0(stem,"_borough.csv")))
  d$scenario <- factor(d$scenario,levels=order)
  d$condition <- factor(d$condition,levels=c("22/2","22/5","25/5","28/5"))
  b$scenario <- factor(b$scenario,levels=order[-1])
  means <- aggregate(change_c~scenario,b,mean)
  th <- theme_bw(base_size=8)+theme(plot.title=element_blank(),
      plot.tag=element_text(face="bold",size=11),axis.text=element_text(size=7),
      panel.grid.minor=element_blank())
  # Equal-borough mean, not uncertainty: no SD/sqrt(33) bars.
  a <- ggplot(d,aes(condition,mean_change_c,colour=scenario,group=scenario))+
    geom_hline(yintercept=0,linewidth=.25,colour="grey70")+
    geom_linerange(aes(ymin=0,ymax=mean_change_c),position=position_dodge(.9),linewidth=.7,alpha=.6,show.legend=FALSE)+
    geom_point(position=position_dodge(.9),size=2,alpha=.7)+
    scale_colour_manual(values=colors,labels=labels,drop=FALSE)+
    # Restore original year labels, but retain their assumed temperature/UHI
    # mapping in the legend: these are not projections derived from Figure 1.
    scale_x_discrete(labels=c("22/2"="2021","22/5"="2030","25/5"="2050","28/5"="2070"))+
    labs(x="Year",y="Temperature change from baseline (°C)",colour="Land scenario")+th+
    theme(legend.position="inside",legend.position.inside=c(.02,.98),legend.justification=c(0,1),
      legend.key.height=grid::unit(2.5,"mm"),legend.text=element_text(size=6),
      legend.background=element_rect(fill=NA,colour=NA),legend.key=element_rect(fill=NA,colour=NA))
  base <- list(scale_fill_manual(values=colors),
    scale_x_discrete(labels=setNames(labels[-1],order[-1])),
    labs(x="Land scenarios",y="Temperature change (°C)"),th,
    theme(legend.position="none",axis.text.x=element_text(angle=45,hjust=1),panel.grid=element_blank()))
  p <- ggplot(means,aes(scenario,change_c,fill=scenario))+
    # Match the original notebook's bar transparency (alpha = 0.7).
    geom_hline(yintercept=0,linewidth=.25,colour="grey70")+geom_col(width=.65,linewidth=.2,colour="grey35",alpha=.7)+
    geom_text(aes(label=sprintf("%+.2f",change_c),vjust=ifelse(change_c>0,-.5,1.4)),size=2.1,colour="grey35")+
    scale_y_continuous(expand=expansion(mult=c(.15,.15)))+base+
    labs(y="Temperature change due to land use (°C)")
  main <- a+p+plot_layout(widths=c(1,1))+plot_annotation(tag_levels="a")
  # Longer self-contained labels; modest extra height, unchanged width/data.
  ggsave(file.path(summary_dir,paste0(output_stem,".png")),main,width=7.2,height=4.2,dpi=450,bg="white")
  sm <- ggplot(b,aes(scenario,change_c,fill=scenario))+
    geom_hline(yintercept=0,linewidth=.25,colour="grey70")+
    geom_boxplot(width=.65,outlier.shape=NA,linewidth=.3)+
    geom_point(position=position_jitter(width=.12,seed=1),size=.8,alpha=.5)+base
  if (!file.exists(file.path(summary_dir,"figure3_sm_borough_v02_20261111.png"))) {
    ggsave(file.path(summary_dir,"figure3_sm_borough_v02_20261111.png"),sm,
           width=7.2,height=3.8,dpi=450,bg="white")
  }
  manifest_path <- file.path(summary_dir,paste0(stem,"_manifest.json"))
  m <- jsonlite::read_json(manifest_path,simplifyVector=FALSE)
  script <- file.path(code_dir,"post_processing_layers","plot_figure3_temperature.R")
  m$plot_script <- list(path=script,sha256=digest::digest(file=script,algo="sha256"))
  m$year_labels <- list(`2021`="22/2",`2030`="22/5",`2050`="25/5",`2070`="28/5")
  m$png_sha256 <- lapply(c(paste0(output_stem,".png"),"figure3_sm_borough_v02_20261111.png"),
    function(f) list(file=f,sha256=digest::digest(file=file.path(summary_dir,f),algo="sha256")))
  jsonlite::write_json(m,file.path(summary_dir,paste0(output_stem,"_manifest.json")),auto_unbox=TRUE,pretty=TRUE)
  main
}

build_figure3_temperature <- function(summary_dir, code_dir) {
  suppressPackageStartupMessages({library(ggplot2); library(patchwork)})
  env <- new.env()
  sys.source(file.path(code_dir, "func_colors.R"), env)
  order <- c("allbuilt", "treerisk", "treeopp", "green10", "target10",
             "green20", "target20", "green30", "target30")
  labels <- c("All\nBuilt", "Tree\nRisk", "Tree\nOpp", "Green\n10%", "Target\n10%",
              "Green\n20%", "Target\n20%", "Green\n30%", "Target\n30%")
  stem <- "figure3_alternative_area_weighted_v01_20261111"
  city <- read.csv(file.path(summary_dir, paste0(stem, "_citywide.csv")))
  borough <- read.csv(file.path(summary_dir, paste0(stem, "_borough.csv")))
  stopifnot(nrow(city) == 9, nrow(borough) == 297,
            setequal(city$scenario, order), all(table(borough$scenario) == 33))
  city$scenario <- factor(city$scenario, levels=order)
  borough$scenario <- factor(borough$scenario, levels=order)
  common <- list(scale_x_discrete(labels=setNames(labels, order)),
    scale_fill_manual(values=env$scenario_colors[order]),
    scale_y_continuous(expand=expansion(mult=c(.12,.14))),
    theme_classic(base_size=7, base_family="sans"),
    theme(legend.position="none", axis.text.x=element_text(size=5.8),
          axis.text.y=element_text(size=6), plot.title=element_text(size=7, face="bold", hjust=0),
          plot.tag=element_text(size=8, face="bold"),
          plot.margin=margin(5,5,5,5)))
  a <- ggplot(city, aes(scenario, change_c, fill=scenario)) +
    geom_hline(yintercept=0, colour="grey60", linewidth=.25) +
    geom_col(width=.7, colour="grey25", linewidth=.25) +
    geom_text(aes(label=sprintf("%+.2f", change_c),
                  vjust=ifelse(change_c >= 0, -.6, 1.5)),
              size=2.1, colour="grey35") +
    labs(title="London-wide change", x=NULL, y="Temperature change (°C)") + common
  b <- ggplot(borough, aes(scenario, change_c, fill=scenario)) +
    geom_hline(yintercept=0, colour="grey60", linewidth=.25) +
    geom_boxplot(width=.66, linewidth=.3, outlier.shape=NA) +
    geom_point(position=position_jitter(width=.14, height=0, seed=20261111),
               size=.65, alpha=.45, colour="grey25") +
    labs(title="Variation across boroughs", x=NULL, y="Temperature change (°C)") + common
  # Independent y ranges distinguish area-wide means from borough distributions.
  combined <- (a + b) + plot_annotation(tag_levels="a")
  out <- file.path(summary_dir, paste0(stem, ".png"))
  ggsave(out, combined, width=7.2, height=3.6, dpi=450, bg="white")
  manifest_path <- file.path(summary_dir, paste0(stem, "_manifest.json"))
  m <- jsonlite::read_json(manifest_path, simplifyVector=FALSE)
  script <- file.path(code_dir, "post_processing_layers", "plot_figure3_temperature.R")
  m$plot_generator <- list(path=script, sha256=digest::digest(file=script, algo="sha256"))
  m$palette <- list(path=file.path(code_dir,"func_colors.R"),
                   sha256=digest::digest(file=file.path(code_dir,"func_colors.R"),algo="sha256"))
  m$outputs$png <- list(path=normalizePath(out), sha256=digest::digest(file=out,algo="sha256"))
  jsonlite::write_json(m, manifest_path, auto_unbox=TRUE, pretty=TRUE)
  combined
}
