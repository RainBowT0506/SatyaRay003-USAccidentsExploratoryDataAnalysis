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

year_df = pd.DataFrame(df.Start_Time.dt.year.value_counts()).reset_index()\
    .rename(columns={'Start_Time':'Year', 'count':'Cases'}).sort_values(by='Cases', ascending=True)

print(year_df.columns)
print(year_df.head(10))

def visualize_accident_percentage_over_years():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

    ax = sns.barplot(y=year_df['Cases'], x=year_df['Year'],
                     palette=['#9a90e8', '#5d82de', '#3ee6e0', '#40ff53', '#2ee88e'])

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x() + 0.2, i.get_height() - 50000, \
                str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=15, weight='bold',
                color='white')

    plt.ylim(10000, 900000)
    plt.title('\nRoad Accident Percentage \nover past 5 Years in US (2016-2020)\n', size=20, color='grey')
    plt.ylabel('\nAccident Cases\n', fontsize=15, color='grey')
    plt.xlabel('\nYears\n', fontsize=15, color='grey')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    for i in ['bottom', 'top', 'left', 'right']:
        ax.spines[i].set_color('white')
        ax.spines[i].set_linewidth(1.5)

    for k in ['top', 'right', "bottom", 'left']:
        side = ax.spines[k]
        side.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=0.3)
    MA = mpatches.Patch(color='#2ee88e', label='Year with Maximum\n no. of Road Accidents')
    MI = mpatches.Patch(color='#9a90e8', label='Year with Minimum\n no. of Road Accidents')
    ax.legend(handles=[MA, MI], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=['#2ee88e', '#9a90e8'], edgecolor='white');
    plt.show()

visualize_accident_percentage_over_years()