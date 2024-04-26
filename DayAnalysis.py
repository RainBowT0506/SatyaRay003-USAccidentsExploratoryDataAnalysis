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

day_df = pd.DataFrame(df.Start_Time.dt.day_name().value_counts()).reset_index().rename(
    columns={'Start_Time': 'Day', 'count': 'Cases'})


# 一週內不同日期的道路事故百分比
def visualize_weekly_accident_distribution():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

    ax = sns.barplot(y=day_df['Cases'], x=day_df['Day'],
                     palette=['#D50000', '#FF1744', '#FF5252', '#ff7530', '#ffa245', '#50fa9d', '#7eedb0'])

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x() + 0.1, i.get_height() - 20000, \
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15, weight='bold',
                color='white')

    plt.ylim(-10000, 300000)
    plt.title('\nRoad Accident Percentage \nfor different days over the week\n', size=20, color='grey')
    plt.ylabel('\nAccident Cases\n', fontsize=15, color='grey')
    plt.xlabel('\nDay of the Week\n', fontsize=15, color='grey')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)

    for i in ['top', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    ax.set_axisbelow(True)
    ax.spines['bottom'].set_bounds(0.005, 6)
    ax.spines['left'].set_bounds(0, 300000)

    MA = mpatches.Patch(color='#D50000', label='Day with Maximum\n no. of Road Accidents')
    MI = mpatches.Patch(color='#7eedb0', label='Day with Minimum\n no. of Road Accidents')

    ax.legend(handles=[MA, MI], prop={'size': 10.5}, loc='best', borderpad=1, edgecolor='white',
              labelcolor=['#D50000', '#7eedb0'])
    plt.show()


visualize_weekly_accident_distribution()
