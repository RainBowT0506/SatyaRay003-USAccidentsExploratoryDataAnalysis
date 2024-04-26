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

month_df = pd.DataFrame(df.Start_Time.dt.month.value_counts()).reset_index().rename(
    columns={'Start_Time': 'Month', 'count': 'Cases'})

month_names = list(calendar.month_name)[1:]
month_df.Month = month_names

print(month_df.columns)
print(month_df.head(10))


# 美國不同月份道路事故百分比（2016-2020）
def visualize_monthly_accident_distribution():
    fig, ax = plt.subplots(figsize=(10, 8), dpi=80)

    cmap = cm.get_cmap('plasma', 12)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(x=month_df['Cases'], y=month_df['Month'], palette='plasma')

    total = df.shape[0]
    for p in ax.patches:
        plt.text(p.get_width() - 17000, p.get_y() + 0.4,
                 '{:.2f}%'.format(p.get_width() * 100 / total), ha='center', va='center', fontsize=15, color='white',
                 weight='bold')

    plt.title('\nRoad Accident Percentage \nfor different months in US (2016-2020)\n', size=20, color='grey')
    plt.xlabel('\nAccident Cases\n', fontsize=15, color='grey')
    plt.ylabel('\nMonths\n', fontsize=15, color='grey')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    plt.xlim(0, 300000)

    for i in ['top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    ax.set_axisbelow(True)
    ax.spines['bottom'].set_bounds(0, 300000)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)

    MA = mpatches.Patch(color=clrs[0], label='Month with Maximum\n no. of Road Accidents')
    MI = mpatches.Patch(color=clrs[-1], label='Month with Minimum\n no. of Road Accidents')

    ax.legend(handles=[MA, MI], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=[clrs[0], 'grey'], edgecolor='white')
    plt.show()


visualize_monthly_accident_distribution()
