#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 18:08:58 2019

@author: johnlong
"""

import pandas as pd
import numpy as np

cd = ‎iCloud Drive/Documents/University/Imperial College London/SEF Project/SEF-ML/

generation_per_type=pd.read_csv('actual_aggregated_generation_per_type.csv')
apx_pricee= pd.read_csv('actual_aggregated_generation_per_type.csv')
generation_forcast('day_ahead_generation_forecast_wind_and_solar.csv')