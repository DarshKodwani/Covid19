# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:03:22 2020

@author: tharshi
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# make pretty
plt.style.use('seaborn')

#%% Energy Data

# import UK energy data
df = pd.read_csv("../data/UK_grid/energy_uk_2019_20.csv")

# drop uneeded columns and fix datatypes
df = df.drop(df.columns[3:], axis=1)
#df = df.drop("time_days", axis=1)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].dt.date

# aggregate by date
daily_en = df.groupby(["timestamp"]).mean().reset_index()

# filter by covid timeframe (date > feb 20th)
daily_en_2020 = daily_en[(daily_en["timestamp"] >= pd.to_datetime("2020-02-20")) & 
                         (daily_en["timestamp"] < pd.to_datetime("2020-04-15"))]
daily_en_2019 = daily_en[(daily_en["timestamp"] >= pd.to_datetime("2019-02-20")) & 
                         (daily_en["timestamp"] < pd.to_datetime("2019-04-15"))]

en_2020 = df[(df["timestamp"] >= pd.to_datetime("2020-03-20")) & 
                         (df["timestamp"] < pd.to_datetime("2020-04-15"))]
en_2019 = df[(df["timestamp"] >= pd.to_datetime("2019-03-20")) & 
                         (df["timestamp"] < pd.to_datetime("2019-04-15"))]

#%% Plot Energy Data

fig, ax = plt.subplots()
ax.plot(daily_en_2019["demand"].to_list(), label="2019")
ax.plot(daily_en_2020["demand"].to_list(), label="2020")
ax.set_xlabel("Days since Feb 20th")
ax.set_ylabel("Demand (GW)")
ax.legend()
ax.set_title("Mean Energy Demand During COVID19 (UK)")
plt.savefig("daily_energy.png", dpi=600)


fig2, ax2 = plt.subplots()
ax2.plot(en_2019["demand"].to_list(), label="2019")
ax2.plot(en_2020["demand"].to_list(), label="2020")
ax2.set_xlabel("Measurements since March 20th (freq ~ 5 min)")
ax2.set_ylabel("Demand (GW)")
ax2.legend()
ax2.set_title("Energy Demand During COVID19 (UK)")
plt.savefig("_energy.png", dpi=600)