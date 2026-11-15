#!/usr/bin/env Rscript
# Supplementary candidate only: instantaneous temperature-based capacity indices,
# not worker-hours or an average summer working day. Versions belong to outputs.
suppressPackageStartupMessages({library(ggplot2);library(dplyr)})
a <- commandArgs(trailingOnly=TRUE)
if(length(a)!=2) stop('Usage: INPUT_CSV NEW_OUTPUT_PNG')
if(file.exists(a[[2]])) stop('Refusing to replace existing figure')
script <- sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[[1]])
source(file.path(dirname(normalizePath(script)), 'figure4_panel_helpers.R'))
env <- new.env(); sys.source(file.path(dirname(normalizePath(script)), '..','func_colors.R'),env)
d <- read.csv(a[[1]]) %>% filter(scenario!='baseline',method!='legacy_fixed')
stopifnot(setequal(unique(d$scenario),figure4_scenarios),nrow(d)==36)
stopifnot(!anyDuplicated(d[c('scenario','method','rh_percent')]),all(is.finite(d$change_pp)))
for(s in figure4_scenarios) stopifnot(setequal(with(d[d$scenario==s,],paste(method,rh_percent)),
  c('corrected_fixed 45','daily_tmax_sensitivity 35','daily_tmax_sensitivity 45','daily_tmax_sensitivity 55')))
d <- d %>% mutate(scenario_id=scenario,
  scenario=factor(scenario,levels=figure4_scenarios,labels=figure4_labels),
  panel=ifelse(method=='corrected_fixed','a  Fixed 25/5°C benchmark (RH45)',
    paste0(c('b','c','d')[match(rh_percent,c(35,45,55))],
      '  JJA daily-Tmax mean (RH',rh_percent,')')))
p <- ggplot(d,aes(scenario,change_pp,fill=scenario_id))+
  geom_hline(yintercept=0,colour='grey55',linewidth=.3)+
  geom_col(alpha=.7,colour='grey20',linewidth=.2,width=.72)+
  facet_wrap(~panel,ncol=2)+scale_fill_manual(values=env$scenario_colors)+
  labs(x=NULL,y='Change in modelled heavy-work capacity (percentage points)')+
  theme_classic(base_size=11)+theme(legend.position='none',strip.background=element_blank(),
    strip.text=element_text(face='bold',hjust=0,size=10),axis.text.x=element_text(size=8),
    panel.spacing=grid::unit(1.2,'lines'))
ggsave(a[[2]],p,width=12,height=7,dpi=300,bg='white')
