# libraries
library(dplyr)
library(tidyverse)
library(rgdal)
library(lubridate)
library(raster)
library(sp)
library(rgeos)
library(geometry)
library(sf)
library(geometry)
library(rmapshaper)
library(devtools)
library(data.table)

# load in ovitrap data
ovitrap <- read.csv("..../OVITRAP_COMPLETE_JMARGUTTI_20190909.csv")

# convert date column into date
ovitrap <- ovitrap %>%
  mutate(date = as.Date(date)) 

# create spatial index for points of ovitraps
ovispatial <- SpatialPointsDataFrame(coords = ovitrap[, c("longitude", "latitude")], 
                                     data = ovitrap,
                                     proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

# fetch google earth engine data from directory and change raster layer names to
# subdirectory names, which include dates
# for this create object with directory and sub directory names
# This is just one example for the precipitation raster, but can be done to all
primary.folder.precip <- list.files("..../Climatic data/data_biweekly/JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRate/", full.names = TRUE)
sub.folder.precip <- list.files("..../Climatic data/data_biweekly/JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRate/", full.names = FALSE)

# now create object with directory path to include rasterlayer names
precip.layers <- list.files("..../Climatic data/data_biweekly/JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRate/", 
                            full.names = TRUE, pattern = "*.tif$", recursive = TRUE)

# make sure that the layer names will be changed to the subdirectory folder
# to include the dates, for this paste the first part of the created object
# after the subdirectory folder name and end with the raster extension
precip.names <- paste0(primary.folder.precip, "/", sub.folder.precip, ".tif")

# final step to rename all documents
# rename all files to date column
file.rename(from = precip.layers, to = precip.names)

# Now move to creation of dataset containing all climatic data per ovitrap location

# first
# stack all rasters in folder to extract 
# all climatic values for unique mosquito density points 
# combine all rasters in one object
grids <- list.files("..../Climatic data/data_biweekly/JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRate/", 
                    recursive = TRUE, 
                    pattern = "*.tif$")

# check the number of files in the raster list (grids)
length(grids) # is 20 which is correct

# second
# create a raster stack of the rasters in grids
# ideally these rasters are named after their respective dates
# because column that is created has name of raster
s <- stack(paste0("/Users/fleurhierink/Dropbox/student data geomatics/Climatic data/data_biweekly/JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRate/", grids))

# third
# extract raster values for each unique ovitrap point
# and save the value with the long-lat of each point (maintain spatial index), sp = TRUE
ovipoints.unique <- ovipoints %>%
  distinct(id)

ovistack <- raster::extract(s, ovipoints.unique, sp = TRUE) # works fine, all layers renamed!

# isolate the variable name, in this case JAXA..
# and transform data to long format
ovistack1 <- ovistack@data %>%
  gather(variable, value, 2:21) %>%
  separate(variable, c("variable", "start.date", "end.date"), "_") %>%
  spread(variable, value) %>%
  mutate(start.date = as.Date(start.date, "%Y.%m.%d"), 
         end.date = as.Date(end.date, "%Y.%m.%d"))

# merge ovi data with climatic data in ovistack1
# merge based on closest date
# note that the climatic data now only represents 2014 data
ovitrap <- data.table(ovitrap)
ovistack1 <- data.table(ovistack1)

setkey(ovitrap, id, date)
setkey(ovistack1, id, end.date)
ovi.final <- ovitrap[ovistack1, roll = T]
