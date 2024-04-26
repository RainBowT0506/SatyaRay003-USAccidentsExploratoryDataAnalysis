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

states = gpd.read_file('Shapefiles/States_shapefile.shp')
geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)


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

# 美國不同時區道路事故視覺化（2016-2020）
def visualization_of_road_accidents_for_different_timezones():
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim([-125, -65])
    ax.set_ylim([22, 55])
    states.boundary.plot(ax=ax, color='black');

    colors = ['#00db49', '#ff5e29', '#88ff33', '#fffb29']
    # 4132
    count = 0
    for i in list(timezone_df.Timezone):
        geo_df[geo_df['Timezone'] == i].plot(ax=ax, markersize=1, color=colors[count], marker='o', label=i);
        count += 1

    plt.legend(markerscale=10., prop={'size': 15}, edgecolor='white', title="Timezones", title_fontsize=15,
               loc='lower right');

    for i in ['bottom', 'top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)

    plt.title('\nVisualization of Road Accidents \nfor different Timezones in US (2016-2020)', size=20, color='grey');
    plt.show()

# visualize_accident_cases_by_timezone_percentage()

visualization_of_road_accidents_for_different_timezones()