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

# 建立街道及其對應事故案例的資料框
# create a dataframe of Street and their corresponding accident cases
street_df = pd.DataFrame(df['Street'].value_counts()).reset_index().rename(
    columns={'Street': 'Street No.', 'count': 'Cases'})
print(street_df.columns)

states = gpd.read_file('Shapefiles/States_shapefile.shp')
geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)


# 美國十大事故多發街道（2016-2020）
def visualize_top_10_accident_prone_streets():
    top_ten_streets_df = pd.DataFrame(street_df.head(10))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

    cmap = cm.get_cmap('gnuplot2', 10)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=top_ten_streets_df['Cases'], x=top_ten_streets_df['Street No.'], palette='gnuplot2')
    ax1 = ax.twinx()
    sns.lineplot(data=top_ten_streets_df, marker='o', x='Street No.', y='Cases', color='white', alpha=.8)

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x() + 0.04, i.get_height() - 2000, \
                '{:,d}'.format(int(i.get_height())), fontsize=12.5, weight='bold',
                color='white')

    ax.axes.set_ylim(-1000, 30000)
    ax1.axes.set_ylim(-1000, 40000)
    plt.title('\nTop 10 Accident Prone Streets in US (2016-2020)\n', size=20, color='grey')

    ax1.axes.yaxis.set_visible(False)
    ax.set_xlabel('\nStreet No.\n', fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    for i in ['top', 'right']:
        side1 = ax.spines[i]
        side1.set_visible(False)
        side2 = ax1.spines[i]
        side2.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)

    ax.spines['bottom'].set_bounds(0.005, 9)
    ax.spines['left'].set_bounds(0, 30000)
    ax1.spines['bottom'].set_bounds(0.005, 9)
    ax1.spines['left'].set_bounds(0, 30000)
    ax.tick_params(axis='both', which='major', labelsize=12)

    MA = mpatches.Patch(color=clrs[1], label='Street with Maximum\n no. of Road Accidents')
    MI = mpatches.Patch(color=clrs[-2], label='Street with Minimum\n no. of Road Accidents')
    ax.legend(handles=[MA, MI], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=[clrs[1], 'grey'], edgecolor='white')
    plt.show()


visualize_top_10_accident_prone_streets()
