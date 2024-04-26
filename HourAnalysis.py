import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
import calendar
import plotly as pt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from pylab import *
import matplotlib.patheffects as PathEffects

import descartes
import geopandas as gpd
from Levenshtein import distance
from itertools import product
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point, Polygon

import geoplot
from geopy.geocoders import Nominatim

import warnings

warnings.filterwarnings('ignore')

# 讀取資料集並將其載入到 pandas 資料框中
# read & load the dataset into pandas dataframe
df = pd.read_csv('small_data.csv', encoding='ISO-8859-1')
# 將 'End_Time' 和 'Start_Time' 轉換為日期時間對象
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

hour_df = pd.DataFrame(df.Start_Time.dt.hour.value_counts()).reset_index().rename(
    columns={'Start_Time': 'Hours', 'count': 'Cases'}).sort_values('Hours')
print(hour_df.columns)

# 一天中不同時間的道路事故百分比
def visualize_hourly_accident_distribution():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

    clrs = []
    for x in hour_df['Cases']:
        if int(hour_df[hour_df['Cases'] == x]['Hours']) <= 11:
            if (x == max(list(hour_df['Cases'])[:12])):
                clrs.append('grey')
            else:
                clrs.append('#05ffda')
        else:
            if (x == max(list(hour_df['Cases'])[12:])):
                clrs.append('grey')
            else:
                clrs.append('#2426b3')
    ax = sns.barplot(y=hour_df['Cases'], x=hour_df['Hours'], palette=clrs)
    ax1 = ax.twinx()

    sns.lineplot(data=hour_df, marker='o', x='Hours', y='Cases', color='white', alpha=1)

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x(), i.get_height() + 1000, \
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=10,
                color='black')

    plt.ylim(1000, 150000)
    plt.title('\nRoad Accident Percentage \nfor different hours along the day\n', size=20, color='grey')

    ax1.axes.yaxis.set_visible(False)
    ax.set_xlabel('\nHours\n', fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    for i in ['bottom', 'top', 'left', 'right']:
        ax.spines[i].set_color('white')
        ax.spines[i].set_linewidth(1.5)
        ax1.spines[i].set_color('white')
        ax1.spines[i].set_linewidth(1.5)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, alpha=.3)
    ax.tick_params(axis='both', which='major', labelsize=12)

    MA = mpatches.Patch(color='grey', label='Hour with Maximum\n no. of Road Accidents')
    MO = mpatches.Patch(color='#05ffda', label='Monrning Hours')
    NI = mpatches.Patch(color='#2426b3', label='Night Hours')

    ax.legend(handles=[MA, MO, NI], prop={'size': 10.5}, loc='upper left', borderpad=1, edgecolor='white');
    plt.show()

visualize_hourly_accident_distribution()