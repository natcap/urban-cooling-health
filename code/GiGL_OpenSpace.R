

f <- 'G:/Shared drives/Wellcome Trust Project Data/0_source_data/GiGL land use data/GiGL_OpenSpace_Sites_All_region'

d <- sf::st_read(f)

names(d)

unique(d$PrimaryUse) %>% sort()
unique(d$OtherUses) %>% sort()
unique(d$POSGrade) %>% sort()


PrimaryUse_select <- c("Disused quarry/gravel pit", "Disused railway trackbed", "Land reclamation",
                       "Other hard surfaced areas", "Other recreational", "Road island/verge",
                       "Vacant land")
d_ <- d %>%
  filter(PrimaryUse %in% PrimaryUse_select)
