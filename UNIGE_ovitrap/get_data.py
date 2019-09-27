"""
This is a simple script to access, aggregate and download a series of raster files from Google Earth Engine.
The data source is the GSMaP dataset.
This script gets GSMaP data on rainfall within the Philippines and sum all values within a month, i.e. it
computes the monthly cumulative rainfall, expressed in mm/h.
Author: Jacopo Margutti (jmargutti@redcross.nl)
Date: 27-09-2019
"""

import ee
import ee.mapclient
from geetools import batch
from geetools import tools
import datetime

# initialize Google Eartg Engine
ee.Initialize()

# define bounding box of the Philippines
bounding_box_philippines = ee.Geometry.Rectangle([117.17427453, 5.58100332277, 126.537423944, 18.5052273625])

# convenience function to get list of dates
def get_dates_in_range(begin, end):
    """
    this function returns two lists of dates (start_dates and end_dates),
    which correspond to the beginning and end of each month within the input date range (begin, end).
    Example: input: begin='2019-07-01', end='2019-08-31'
             output: start_dates=['2019-07-01', '2019-08-01'], end_dates=['2019-07-31', '2019-08-31']
    """
    dt_start = datetime.datetime.strptime(begin, '%Y-%m-%d')
    dt_end = datetime.datetime.strptime(end, '%Y-%m-%d')
    one_day = datetime.timedelta(1)
    start_dates = [dt_start]
    end_dates = []
    today = dt_start
    while today <= dt_end:
        tomorrow = today + one_day
        if tomorrow.month != today.month:
            start_dates.append(tomorrow)
            end_dates.append(today)
        today = tomorrow
    end_dates.append(dt_end)
    return start_dates, end_dates

# get list of dates from given date range
start_dates, end_dates = get_dates_in_range('2015-01-01', '2015-02-27')

# loop over list of dates
for start, end in zip(start_dates, end_dates):

    # convert datetime objects to strings
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    # print to screen which dates are being selected
    print(start_str, end_str)

    # get GSMaP ImageCollection within given dates and bounding box
    col = (ee.ImageCollection("JAXA/GPM_L3/GSMaP/v6/operational")
        .filterDate(start_str, end_str)
        .filterBounds(bounding_box_philippines))

    # get list of images in collection
    clist = col.toList(col.size().getInfo())

    # save the scale of first image (need to use it later to save aggregated raster)
    image_scale = int(tools.image.minscale(ee.Image(clist.get(0)).select('hourlyPrecipRate')).getInfo())

    # sum all values in ImageCollection, to compute monthly cumulative rainfall
    monthly_cumulative = col.sum()
    # N.B. other aggregation methods are available (e.g. average), see https://developers.google.com/earth-engine/reducers_intro

    # select the variable 'hourlyPrecipRate'
    monthly_cumulative_rainfall = monthly_cumulative.select('hourlyPrecipRate')

    # set a name to the file and download to disk
    name = 'monthly_cumulative_rainfall_' + start.strftime('%Y-%m-%d')
    print('downloading '+name)
    batch.image.toLocal(monthly_cumulative_rainfall, name, scale=image_scale, region=bounding_box_philippines)
