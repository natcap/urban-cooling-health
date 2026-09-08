# Reproduce Figure 7 outputs without requiring Pandoc or an HTML render.
# Run from any directory with: Rscript code/run-fig7.R

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) != 1) {
  stop("Run this file with Rscript: Rscript code/run-fig7.R")
}

script_path <- normalizePath(sub("^--file=", "", file_arg))
repository_root <- dirname(dirname(script_path))
setwd(repository_root)

input_rmd <- file.path("code", "equity-health-fig7-production.Rmd")
temporary_r <- tempfile(fileext = ".R")
on.exit(unlink(temporary_r), add = TRUE)

knitr::purl(input_rmd, output = temporary_r, quiet = TRUE)
sys.source(temporary_r, envir = new.env(parent = globalenv()))
