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

timezone_df = pd.DataFrame(df['Timezone'].value_counts()).reset_index().rename(columns={'count': 'Cases'})

print(timezone_df)

# 美國不同時區的事故案例百分比（2016-2020）
def visualize_accident_cases_by_timezone_percentage():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)

    cmap = cm.get_cmap('spring', 4)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=timezone_df['Cases'], x=timezone_df['Timezone'], palette='spring')

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x() + 0.3, i.get_height() - 50000, \
                '{}%'.format(round(i.get_height() * 100 / total)), fontsize=15, weight='bold',
                color='white')

    plt.ylim(-20000, 700000)
    plt.title('\nPercentage of Accident Cases for \ndifferent Timezone in US (2016-2020)\n', size=20, color='grey')
    plt.ylabel('\nAccident Cases\n', fontsize=15, color='grey')
    plt.xlabel('\nTimezones\n', fontsize=15, color='grey')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)

    for i in ['top', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)
    ax.spines['bottom'].set_bounds(0.005, 3)
    ax.spines['left'].set_bounds(0, 700000)

    MA = mpatches.Patch(color=clrs[0], label='Timezone with Maximum\n no. of Road Accidents')
    MI = mpatches.Patch(color=clrs[-1], label='Timezone with Minimum\n no. of Road Accidents')
    ax.legend(handles=[MA, MI], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=[clrs[0], 'grey'], edgecolor='white');
    plt.show()


visualize_accident_cases_by_timezone_percentage()