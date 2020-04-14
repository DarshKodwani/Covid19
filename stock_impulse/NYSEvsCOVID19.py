# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:34:32 2020

@author: Tharshi S and Darsh K.
"""

# import useful frameworks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# make pretty
plt.style.use('seaborn')

#%% NY DATA

# import NYT COVID data
nyt_df = pd.read_csv("../data/NYT/covid-19-data/us-counties.csv")

# drop uneeded columns
nyt_df = nyt_df.drop(["county", "state", "fips"], axis=1)

# aggregate by date
nyt_df = nyt_df.groupby(['date']).sum().reset_index()

# plot NYT data
fig1 = plt.figure()
plt.semilogy(nyt_df["cases"],  ".", label="cases")
plt.semilogy(nyt_df["deaths"], ".", label="deaths")
plt.xlabel("Days since {}".format(nyt_df["date"].loc[0]))
plt.ylabel("Count")
plt.legend()
plt.title("COVID19 Cumulative # of Cases and Deaths (USA)")
plt.grid("on")
plt.savefig("cumulative_cases_deaths.pdf", dpi=600)

#%% S&P500 DATA

# import S&P500 data
yf_df = pd.read_csv("../data/Yahoo Finance/covid_SP500.csv")

# drop uneeded columns
yf_df = yf_df.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1)

## add weekends to the dataframe

# all dates spanning the NYT dataset
all_dates = pd.date_range(start=nyt_df["date"].iloc[0],
                          end=nyt_df["date"].iloc[-1])

all_dates = pd.to_datetime(all_dates)

# all business dates in that same range
biz_dates = pd.bdate_range(start=nyt_df["date"].iloc[0],
                          end=nyt_df["date"].iloc[-1])

biz_dates = pd.to_datetime(biz_dates)

# weekends from that date range
weekends = all_dates.difference(biz_dates)

# list of holidays where the NYSE is closed
holidays = pd.to_datetime(['2020-02-17', '2020-04-10'])
holidays_no_weekends = holidays.difference(weekends)

## fill in the weekends and holidays
## with last closing values for NYSE
for i in range(len(weekends)):
    date = weekends[i]
    
    # we handle saturdays and sundays differently
    if i % 2 == 0:
        delayed_date = date - datetime.timedelta(days=1)
    else:
        delayed_date = date - datetime.timedelta(days=2)
        
    if delayed_date in holidays:
        delayed_date = delayed_date - datetime.timedelta(days=1)
    else:
        pass
    
    close = yf_df[yf_df["Date"] == delayed_date.strftime("%Y-%m-%d")]["Close"].iloc[0]
    row = pd.DataFrame([[date.strftime("%Y-%m-%d"), close]], columns=["Date", "Close"])
    yf_df = yf_df.append(row)
    
for i in range(len(holidays_no_weekends)):
    date = holidays_no_weekends[i]
    delayed_date = date - datetime.timedelta(days=1)
    
    if delayed_date in weekends:
        delayed_date = delayed_date - datetime.timedelta(days=2)
    else:
        pass
    
    close = yf_df[yf_df["Date"] == delayed_date.strftime("%Y-%m-%d")]["Close"].iloc[0]
    row = pd.DataFrame([[date.strftime("%Y-%m-%d"), close]], columns=["Date", "Close"])
    yf_df = yf_df.append(row)
    
# final sort of dataframe by date
yf_df = yf_df.sort_values(by="Date")

# plot S&P500 data
fig2 = plt.figure()
plt.semilogx(nyt_df["cases"], yf_df["Close"],  ".")
plt.xlabel("Cases since {}".format(nyt_df["date"].loc[0]))
plt.ylabel("Closing Price (USD)")
plt.title("S&P500 During COVID19 Crisis")
plt.grid("on")
plt.savefig("sp500_v_inf.pdf", dpi=600)

fig3 = plt.figure()
plt.semilogx(nyt_df["deaths"], yf_df["Close"], ".")
plt.xlabel("Deaths since {}".format(nyt_df["date"].loc[0]))
plt.ylabel("Closing Price (USD)")
plt.title("S&P500 During COVID19 Crisis")
plt.grid("on")
plt.savefig("sp500_v_deaths.pdf", dpi=600)