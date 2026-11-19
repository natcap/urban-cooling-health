#!/usr/bin/env Rscript
# Seasonal replacement for the annual cause-specific plot in plot_figure4_citywide.R.
# Usage: Rscript code/health_assessment/plot_extended_data_health.R INPUT_CSV OUTPUT_STEM
# Only output names are versioned; existing artifacts are never overwritten.
suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2); library(jsonlite)})
a <- commandArgs(TRUE); stopifnot(length(a)==2)
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)))
root <- dirname(dirname(dirname(script)))
source(file.path(root,'code/post_processing_layers/figure4_panel_helpers.R'))
pal <- new.env(); sys.source(file.path(root,'code/func_colors.R'),pal)
stem <- a[2]; stopifnot(!any(file.exists(paste0(stem,c('.png','_data.csv','_manifest.json')))))
labels <- c(all_cause='a  All-cause',cardiovascular='b  Cardiovascular',
 mental_disorder='c  Mental disorders',respiratory='d  Respiratory',self_harm='e  Self-harm')
x <- read.csv(a[1]) %>% filter(reconstruction=='worldclim_aligned',ssp=='ssp370',
 scenario %in% figure4_scenarios,cause %in% names(labels))
stopifnot(nrow(x)==135,!anyDuplicated(x[c('scenario','cause','rr_variant')]),
 setequal(x$rr_variant,c('central','rr_low','rr_high')),all(is.finite(x$deaths_averted)))
# Endpoint envelopes preserve signed harms; these are not Monte Carlo intervals.
d <- x %>% group_by(scenario,cause) %>% summarise(
 estimate=deaths_averted[rr_variant=='central'],lower95=min(deaths_averted),
 upper95=max(deaths_averted),.groups='drop')
stopifnot(nrow(d)==45,all(d$lower95<=d$estimate),all(d$upper95>=d$estimate))
# Rounded Figure 4 anchors catch accidental selection of annual or sensitivity data.
stopifnot(abs(d$estimate[d$scenario=='green30' & d$cause=='all_cause']-38.37)<.005,
 abs(d$estimate[d$scenario=='target30' & d$cause=='all_cause']-50.40)<.005)
pdata <- d %>% figure4_prepare_scenarios() %>% mutate(
 panel=factor(cause,levels=names(labels),labels=labels),
 text=ifelse(cause=='self_harm',sprintf('%.2f',estimate),sprintf('%.1f',estimate)),
 anchor=ifelse(estimate>=0,upper95,lower95),vjust=ifelse(estimate>=0,-.5,1.4))
p <- ggplot(pdata,aes(scenario,estimate,colour=scenario_id))+
 geom_hline(yintercept=0,colour='grey65',linewidth=.3)+
 geom_segment(aes(xend=scenario,y=0,yend=estimate),linewidth=.6)+
 geom_errorbar(aes(ymin=lower95,ymax=upper95),width=.18,linewidth=.4)+geom_point(size=2)+
 geom_text(aes(y=anchor,label=text,vjust=vjust),colour='grey35',size=3.3)+
 scale_colour_manual(values=pal$scenario_colors[figure4_scenarios])+
 facet_wrap(~panel,ncol=3,scales='free_y')+
 scale_x_discrete(labels=function(x) gsub('\n','',x,fixed=TRUE))+
 scale_y_continuous(expand=expansion(mult=c(.14,.20)))+
 labs(x=NULL,y='Heat-related deaths averted per modelled summer')+
 theme_classic(base_size=14)+theme(legend.position='none',strip.background=element_blank(),
 strip.text=element_text(size=15,face='bold',hjust=0,margin=margin(4,0,4,0)),
 axis.title.y=element_text(size=14),axis.text.y=element_text(size=12),
 axis.text.x=element_text(size=10.5,angle=45,hjust=1),panel.spacing=grid::unit(1.3,'lines'))
dir.create(dirname(stem),recursive=TRUE,showWarnings=FALSE)
ggsave(paste0(stem,'.png'),p,width=12,height=7.5,dpi=400,bg='white')
write.csv(d,paste0(stem,'_data.csv'),row.names=FALSE)
write_json(list(source=normalizePath(a[1]),source_md5=unname(tools::md5sum(a[1])),
 script=script,script_md5=unname(tools::md5sum(script)),
 selection='worldclim_aligned / ssp370 / central and coefficient endpoints',
 units='London-wide deaths per modelled June-August summer',rows=nrow(d),
 uncertainty='95% exposure-response coefficient endpoints; not total model uncertainty'),
 paste0(stem,'_manifest.json'),pretty=TRUE,auto_unbox=TRUE)
