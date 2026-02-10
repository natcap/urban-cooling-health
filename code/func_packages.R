Sys.setenv(LANG = "en")

#' commonly used R packages
library(here)
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(ggplot2)

ymd   <- format(Sys.time(), "%Y%m%d"); ymd