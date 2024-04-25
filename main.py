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

# 檢查號碼。 列數和行數
# check the no. of columns & rows
print('The Dataset Contains, Rows: {:,d} & Columns: {}'.format(df.shape[0], df.shape[1]))

# 將 Start_Time 和 End_Time 變數轉換為日期時間特徵
# convert the Start_Time & End_Time Variable into Datetime Feature
df.Start_Time = pd.to_datetime(df.Start_Time, format='ISO8601', errors='coerce')
df.End_Time = pd.to_datetime(df.End_Time, format='ISO8601', errors='coerce')


def plot_top_accident_cities():
    # 建立城市及其相應事故案例的資料框
    # create a dataframe of city and their corresponding accident cases
    city_df = pd.DataFrame(df['City'].value_counts()).reset_index().rename(columns={'count': 'Cases'})

    # 只保留 'Cases' 欄位
    top_10_cities = pd.DataFrame(city_df.head(10))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=80)

    cmap = cm.get_cmap('rainbow', 10)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=top_10_cities['Cases'], x=top_10_cities['City'], palette='rainbow')

    total = sum(city_df['Cases'])
    for i in ax.patches:
        ax.text(i.get_x() + .03, i.get_height() - 2500, \
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15, weight='bold',
                color='white')

    plt.title('\nTop 10 Cities in US with most no. of \nRoad Accident Cases (2016-2020)\n', size=20, color='grey')

    plt.ylim(1000, 50000)
    plt.xticks(rotation=10, fontsize=12)
    plt.yticks(fontsize=12)

    ax.set_xlabel('\nCities\n', fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    for i in ['bottom', 'left']:
        ax.spines[i].set_color('white')
        ax.spines[i].set_linewidth(1.5)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)
    MA = mpatches.Patch(color=clrs[0], label='City with Maximum\n no. of Road Accidents')
    ax.legend(handles=[MA], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=clrs[0], edgecolor='white');
    plt.show()


plot_top_accident_cities()


# top_5_cities = df.head(5)
# print(top_5_cities)
#
# features = df.columns
# print(features)