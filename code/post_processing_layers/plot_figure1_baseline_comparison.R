# Review/supplement only: does not overwrite Figure 1 or change its CHELSA reference.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
input <- args[1]; boundary <- args[2]; output <- args[3]
suppressPackageStartupMessages({library(ggplot2);library(sf);library(patchwork)})
stem <- "figure1_baseline_comparison_20261111_v1"
b <- read.csv(file.path(input,paste0(stem,"_baselines.csv")))
s <- read.csv(file.path(input,paste0(stem,"_summary.csv")))
d <- read.csv(file.path(input,paste0(stem,"_by_model.csv")))
theme_set(theme_classic(base_size=9))
refs <- data.frame(reference=rep(c("WorldClim\n1970–2000", "CHELSA\nJuly 2021"),each=33),
                   temperature=c(b$worldclim_1970_2000,b$chelsa_2021))
a <- ggplot(refs,aes(reference,temperature))+geom_boxplot(width=.45,outlier.shape=NA)+
  geom_point(position=position_jitter(width=.1,seed=1),size=1,alpha=.4)+
  stat_summary(fun=mean,geom="point",shape=18,size=3,colour="#B35806")+
  labs(x=NULL,y="July maximum temperature (°C)",title="Historical references")
long <- rbind(transform(s,reference="CHELSA July 2021",difference=vs_chelsa),
              transform(s,reference="WorldClim 1970–2000",difference=vs_worldclim))
c <- ggplot(long,aes(period,difference,colour=reference,group=reference))+
  geom_line(linewidth=.6)+geom_point(size=2)+facet_wrap(~ssp)+
  scale_colour_manual(values=c("#B35806","#2166AC"))+
  labs(x=NULL,y="Future minus reference (°C)",colour=NULL,title="Effect of reference choice")+
  theme(legend.position="bottom",axis.text.x=element_text(angle=30,hjust=1))
dir.create(output,recursive=TRUE,showWarnings=FALSE)
ggsave(file.path(output,paste0(stem,".png")),a+c+plot_layout(widths=c(1,2))+plot_annotation(tag_levels="a"),
       width=9,height=4,dpi=300,bg="white")
map <- st_read(boundary,quiet=TRUE)
z <- aggregate(cbind(vs_worldclim,vs_chelsa)~borough+ssp+period,d,mean)
map <- merge(map,z,by.x="NAME",by.y="borough")
m <- ggplot(map)+geom_sf(aes(fill=vs_worldclim),colour="white",linewidth=.05)+
  facet_grid(period~ssp)+scale_fill_viridis_c(name="Temperature\ndifference (°C)",option="C")+
  theme_void(base_size=9)+theme(strip.text=element_text(face="bold"),legend.position="right")
ggsave(file.path(output,"figure1_sm_worldclim_v01_20261111.png"),m,
       width=7.2,height=8,dpi=300,bg="white")
