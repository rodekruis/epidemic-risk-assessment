#R script to transform the ovitrap data in bi-weekly averages
#In case of any questions you can send an e-mail to fleur.hierink@unige.ch

#libraries
library(dplyr)
library(lubridate)

#fetch data from personal directory
#load in ovitrap data
ovitrap <- read.csv("/Users/...../....csv")

#convert the date column into a date
ovitrap$date <- as.Date(ovitrap$date)

#create bi-weekly averages
ovitrap_week <- ovitrap %>%
  mutate(two_weeks = round_date(date, "14 days")) %>%
  group_by(id, longitude, latitude, two_weeks) %>%
  summarise(average_ovi = mean(value)) 

#save data as csv
write.csv("/Users/.../...csv")
