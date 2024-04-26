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

accident_duration_df = pd.DataFrame(df['End_Time'] - df['Start_Time']).reset_index().rename(columns={'index':'Id', 0:'Duration'})
print(accident_duration_df.head(10))
print(accident_duration_df.columns)

top_10_accident_duration_df = pd.DataFrame(accident_duration_df['Duration'].value_counts().head(10).sample(frac = 1)).reset_index().rename(columns={'count': 'Cases'})

print(top_10_accident_duration_df.head(10))
print(top_10_accident_duration_df.columns)
Duration = [str(i).split('days')[-1].strip() for i in top_10_accident_duration_df.Duration]

top_10_accident_duration_df['Duration'] = Duration

# 事故對交通流量影響最大的時段
def visualize_impact_of_accident_duration_on_traffic_flow():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)
    ax.set_facecolor('#e6f2ed')
    fig.patch.set_facecolor('#e6f2ed')

    cmap = cm.get_cmap('bwr', 10)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=top_10_accident_duration_df['Cases'], x=top_10_accident_duration_df['Duration'], palette='bwr')
    ax1 = ax.twinx()
    sns.lineplot(data=top_10_accident_duration_df, marker='o', x='Duration', y='Cases', color='white', alpha=1)

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x(), i.get_height() + 5000, \
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15,
                color='black')

    ax.set(ylim=(1000, 400000))
    ax1.set(ylim=(1000, 500000))

    plt.title('\nMost Impacted Durations on the \nTraffic flow due to the Accidents \n', size=20, color='grey')

    ax1.axes.yaxis.set_visible(False)
    ax.set_xlabel('\nDuration of Accident (HH:MM:SS)\n', fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    for i in ['bottom', 'top', 'left', 'right']:
        ax.spines[i].set_color('white')
        ax.spines[i].set_linewidth(1.5)
        ax1.spines[i].set_color('white')
        ax1.spines[i].set_linewidth(1.5)

    ax.set_axisbelow(True)
    ax.grid(color='white', linewidth=1.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    MA = mpatches.Patch(color=clrs[-3], label='Duration with Maximum\n no. of Road Accidents')
    ax.legend(handles=[MA], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=clrs[-3], facecolor='#e6f2ed', edgecolor='#e6f2ed');
    plt.show()

visualize_impact_of_accident_duration_on_traffic_flow()