import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


dj = pd.read_csv('historical_dj.csv', index_col = 'Date', parse_dates = True)
dj = dj['Value']
dates_dj = np.array(dj.index.values)
ddj = np.gradient(dj) # Gradient of dj 
dddj = np.gradient(ddj) # Gradient of dj 


sp500 = pd.read_csv('historical_sp500.csv', index_col = 'Date', parse_dates = True)
print(sp500.head())
sp500 = sp500['Value']
dates_sp500 = np.array(sp500.index.values)
dsp500 = np.gradient(sp500)
ddsp500 = np.gradient(dsp500)

### Plotting stock data and overlaying pandaemic dates ###

fig = plt.figure(figsize = [12,8])

ax = fig.add_axes([0.15, 0.5, 0.7, 0.45])
ax.plot(dates_dj, dj, label = 'Dow jones')
ax.plot(dates_sp500, sp500, label = 'sp500')
ax.set_ylabel('Value')
ax.set_xlabel('Date')

ax1 = fig.add_axes([0.15, 0.3, 0.7, 0.2])
ax1.plot(dates_dj, ddj)
ax1.plot(dates_sp500, dsp500)
ax1.set_ylabel('First derivative')
ax1.set_xlabel('Date')

ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.2])
ax2.plot(dates_dj, dddj)
ax2.plot(dates_sp500, ddsp500)
ax2.set_ylabel('Second derivative')
ax2.set_xlabel('Date')

# Spanish flu #

start_date_spanish_flu = 1918
end_date_spanish_flu = 1920
color_spanish_flu = 'red'

ax.axvspan(str(start_date_spanish_flu), str(end_date_spanish_flu), alpha=0.2, color=color_spanish_flu, label = 'Spanish flu')
ax1.axvspan(str(start_date_spanish_flu), str(end_date_spanish_flu), alpha=0.2, color=color_spanish_flu)
ax2.axvspan(str(start_date_spanish_flu), str(end_date_spanish_flu), alpha=0.2, color=color_spanish_flu)

# Asian flu #

start_date_asian_flu = 1957
end_date_asian_flu = 1958
color_asian_flu = 'green'

ax.axvspan(str(start_date_asian_flu), str(end_date_asian_flu), alpha=0.2, color=color_asian_flu, label = 'Asian flu')
ax1.axvspan(str(start_date_asian_flu), str(end_date_asian_flu), alpha=0.2, color=color_asian_flu)
ax2.axvspan(str(start_date_asian_flu), str(end_date_asian_flu), alpha=0.2, color=color_asian_flu)

# Swine flu #

start_date_swine_flu = 2009
end_date_swine_flu = 2010
color_swine_flu = 'yellow'

ax.axvspan(str(start_date_swine_flu), str(end_date_swine_flu), alpha=0.2, color=color_swine_flu, label = 'Swine flu')
ax1.axvspan(str(start_date_swine_flu), str(end_date_swine_flu), alpha=0.2, color=color_swine_flu)
ax2.axvspan(str(start_date_swine_flu), str(end_date_swine_flu), alpha=0.2, color=color_swine_flu)

# Ebola #

start_date_ebola = 2014
end_date_ebola = 2016
color_ebola = 'blue'

ax.axvspan(str(start_date_ebola), str(end_date_ebola), alpha=0.2, color=color_ebola, label = 'Ebola')
ax1.axvspan(str(start_date_ebola), str(end_date_ebola), alpha=0.2, color=color_ebola)
ax2.axvspan(str(start_date_ebola), str(end_date_ebola), alpha=0.2, color=color_ebola)

ax.legend()

plt.savefig('dj.pdf')
plt.show()
