# Figure 1 reporting only: reuse validated inputs, never replace historical CSVs.
# Author-selected v2: original boxplot/map styling, CHELSA reference only in main.
build_figure1 <- function(comparison_dir, boundary, output_dir) {
  suppressPackageStartupMessages({library(ggplot2);library(sf);library(patchwork)})
  stem <- "figure1_baseline_comparison_20261111_v1"
  paths <- file.path(comparison_dir,paste0(stem,c("_baselines.csv","_by_model.csv","_summary.csv")))
  b <- read.csv(paths[1]); d <- read.csv(paths[2]); expected <- read.csv(paths[3])
  stopifnot(nrow(b)==33,nrow(d)==594,!anyDuplicated(b$borough),
            !anyDuplicated(d[c("borough","model","ssp","period")]))
  z <- aggregate(cbind(future,vs_chelsa,vs_worldclim)~borough+ssp+period,d,mean)
  stopifnot(nrow(z)==198,all(table(z$ssp,z$period)==33))
  for(i in seq_len(nrow(expected))) {
    q <- z[z$ssp==expected$ssp[i]&z$period==expected$period[i],]
    stopifnot(abs(mean(q$future)-expected$future[i])<1e-8,
              abs(mean(q$vs_chelsa)-expected$vs_chelsa[i])<1e-8)
  }
  years <- c("2021-2040"="2030","2041-2060"="2050","2061-2080"="2070")
  refs <- data.frame(borough=b$borough,period="2021",group="Baseline",value=b$chelsa_2021)
  future <- data.frame(borough=z$borough,period=unname(years[z$period]),
                       group=toupper(z$ssp),value=z$future)
  df <- rbind(refs,future);df$period <- factor(df$period,levels=c("2021","2030","2050","2070"))
  df$group <- factor(df$group,levels=c("Baseline","SSP370","SSP585"))
  pd <- position_dodge(width=.65)
  a <- ggplot(df,aes(period,value,colour=group,group=interaction(period,group)))+
    geom_boxplot(position=position_dodge2(width=.8,padding=.3,preserve="single"),
                 width=.6,outlier.shape=NA,alpha=.4,show.legend=FALSE)+
    stat_summary(fun=mean,geom="point",position=pd,shape=18,size=2,alpha=.7)+
    stat_summary(fun=mean,geom="text",position=position_dodge(width=.65*3),
      aes(label=after_stat(sprintf("%.1f",y)),hjust=ifelse(group=="Baseline",1.5,.5)),size=3,show.legend=FALSE)+
    scale_colour_manual(values=c("Baseline"="#4e79a7","SSP370"="#f28e2b","SSP585"="#e15759"))+
    scale_y_continuous(expand=expansion(mult=c(.02,.10)))+
    labs(x="Year",y="Temperature (°C)",colour=NULL)+
    theme_bw(base_size=11)+theme(panel.grid.major.x=element_blank(),panel.grid.minor=element_blank(),
      legend.position="inside",legend.position.inside=c(.22,.90))
  geometry <- st_read(boundary,quiet=TRUE)
  stopifnot(nrow(geometry)==33,setequal(geometry$NAME,b$borough))
  maps <- merge(geometry,z,by.x="NAME",by.y="borough")
  maps$ssp <- toupper(maps$ssp);maps$year <- unname(years[maps$period])
  # Original pooled decile method, now restricted to the four displayed maps.
  # Share exact breaks between main SSP370 and supplementary SSP585. No
  # period-specific reclassification: a colour retains the same temperature range.
  displayed <- z[z$period!="2021-2040",]
  breaks <- unique(quantile(displayed$vs_chelsa,probs=seq(0,1,.1),type=7))
  labels <- sprintf("%.1f–%.1f",head(breaks,-1),tail(breaks,-1))
  maps$bin <- cut(maps$vs_chelsa,breaks=breaks,include.lowest=TRUE,labels=labels)
  palette <- colorRampPalette(RColorBrewer::brewer.pal(9,"YlOrRd"))(length(labels))
  histogram_checks <- list()
  # Author-selected automatic five-bin insets restored; fixed-bin trial retained
  # as main v04 / SSP585 v03, but is no longer the production selection.
  original_maps <- function(ssp) {
    sel <- maps[maps$ssp==ssp & maps$year %in% c("2050","2070"),]
    stopifnot(nrow(sel)==66,all(table(sel$year)==33))
    p <- ggplot(sel)+geom_sf(aes(fill=bin),colour="grey40",linewidth=.3,show.legend=TRUE)+
      facet_wrap(~year,ncol=1)+scale_fill_manual(values=palette,drop=FALSE,name="ΔT (°C)")+
      guides(fill=guide_legend(reverse=TRUE))+
      theme_minimal(base_size=11)+theme(panel.grid=element_blank(),axis.text=element_blank(),
        legend.position="inside",legend.position.inside=c(.90,.53),
        legend.key.size=grid::unit(9,"pt"),legend.title=element_text(lineheight=.9))
    histogram <- function(year) {
      data <- st_drop_geometry(sel[sel$year==year,])
      stopifnot(nrow(data)==33,!anyDuplicated(data$NAME),all(is.finite(data$vs_chelsa)))
      h <- ggplot(data,aes(vs_chelsa))+
        geom_histogram(bins=5,fill="grey60",colour="white",alpha=.5)+
        geom_text(stat="bin",bins=5,aes(label=ifelse(after_stat(count)>0,after_stat(count),"")),
                  vjust=-.25,size=2)+
        scale_y_continuous(breaks=seq(0,35,5),expand=expansion(mult=c(0,.18)))+
        scale_x_continuous(n.breaks=3,labels=scales::label_number(accuracy=.1))+
        labs(x="ΔT (°C)",y="Count")+theme_minimal(base_size=8)+
        theme(panel.grid=element_blank(),plot.margin=margin(0,0,0,0),
          axis.title=element_text(size=8),axis.text=element_text(size=7),
          panel.background=element_rect(fill=NA,colour=NA),
          plot.background=element_rect(fill=NA,colour=NA))
      bins <- ggplot_build(h)$data[[1]][c("xmin","xmax","count")]
      # Independently count the right-closed ggplot bins, including the first edge.
      independent <- as.integer(table(cut(data$vs_chelsa,
        breaks=c(bins$xmin[1],bins$xmax),include.lowest=TRUE,right=TRUE)))
      stopifnot(sum(bins$count)==33,identical(independent,as.integer(bins$count)))
      histogram_checks[[paste(ssp,year,sep="_")]] <<- bins
      h
    }
    # Keep transparent insets in the lower-left whitespace, not over boroughs.
    cowplot::ggdraw(p)+cowplot::draw_plot(histogram("2050"),x=.00,y=.46,width=.29,height=.13)+
      cowplot::draw_plot(histogram("2070"),x=.00,y=.00,width=.29,height=.13)
  }
  # Facet on BOTH period and SSP: never draw two SSP geometries on top of each other.
  map_panel <- function(value,title) {
    ggplot(maps)+geom_sf(aes(fill=.data[[value]]),colour="grey60",linewidth=.08)+
      facet_grid(period~ssp)+scale_fill_distiller(palette="YlOrRd",direction=1,
        name="Temperature\ndifference (°C)")+
      labs(title=title)+theme_void(base_size=8)+
      theme(strip.text=element_text(face="bold",size=8),legend.title=element_text(size=8),
            plot.title=element_text(face="bold",size=9,hjust=0),
            plot.margin=margin(5,5,5,5),legend.key.height=grid::unit(9,"mm"))
  }
  main <- cowplot::plot_grid(a,original_maps("SSP370"),nrow=1,labels=c("a","b"),
                             label_size=14,rel_widths=c(1,1.05))
  sm <- original_maps("SSP585")
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  files <- file.path(output_dir,c("figure1_main_v03_20261111.png",
                                 "figure1_sm_ssp585_v02_20261111.png"))
  # Explicit versioned names preserve the original figure and comparison exports.
  ggsave(files[1],main,width=7.2,height=5.7,dpi=450,bg="white")
  ggsave(files[2],sm,width=3.7,height=5.7,dpi=450,bg="white")
  manifest <- list(inputs=lapply(paths,function(p) list(path=p,sha256=digest::digest(file=p,algo="sha256"))),
    boundary=boundary,borough_count=33,models=sort(unique(d$model)),
    main_reference="CHELSA July 2021",main_maps="SSP370, 2050 and 2070",
    supplement_reference="CHELSA July 2021; SSP585 maps; existing WorldClim-matched SM retained separately",
    map_breaks=breaks,
    map_break_population="132 borough-period-SSP values, 2050/2070, both SSPs",
    histogram_checks=histogram_checks,
    histogram_method="Automatic five bins per SSP/year; axes vary by inset",
    aggregation="Equal GCM mean within borough, equal borough weight for displayed means",
    script="code/post_processing_layers/plot_figure1.R",
    outputs=lapply(files,function(p) list(path=normalizePath(p),sha256=digest::digest(file=p,algo="sha256"))))
  jsonlite::write_json(manifest,file.path(output_dir,"figure1_main_v03_20261111_manifest.json"),
                      auto_unbox=TRUE,pretty=TRUE)
  invisible(files)
}
