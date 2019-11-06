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

#2014 is data richest year, subset 2014 and continue with this
ovitrap_2014 <- ovitrap_week %>%
  mutate(two_weeks = as.Date(two_weeks)) %>%
  filter(two_weeks >= "2014-01-2014" & two_weeks <= "2014-12-31")

#save data as csv
write.csv(ovitrap_week, "/Users/.../...csv")
write.csv(ovitrap_2014, "/Users/.../...csv")
