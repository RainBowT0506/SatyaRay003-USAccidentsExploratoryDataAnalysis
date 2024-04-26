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
df['Year'] = pd.DataFrame(df.Start_Time.dt.year)

year_df = pd.DataFrame(df.Start_Time.dt.year.value_counts()).reset_index().rename(columns={'Start_Time':'Year', 'count':'Cases'}).sort_values(by='Cases', ascending=True)

print(year_df.columns)
print(year_df.head(10))

states = gpd.read_file('Shapefiles/States_shapefile.shp')
geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)
print("geo_df")
print(geo_df.columns)
print(geo_df.head(10))

# 美國過去 5 年道路事故百分比（2016-2020 年）
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

# 美國過去5年的事故案例
def visualize_accident_cases_over_past_years():
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    fig.suptitle('Accident Cases over the past 5 years in US', fontsize=20, fontweight="bold", color='grey')
    count = 0
    years = ['2016', '2017', '2018', '2019', '2020']
    colors = ['#77fa5a', '#ffff4d', '#ffab36', '#ff894a', '#ff513b']
    for i in [ax1, ax2, ax3, ax4, ax5]:
        i.set_xlim([-125, -65])
        i.set_ylim([22, 55])
        states.boundary.plot(ax=i, color='black');
        print("int(years[count])")
        print(int(years[count]))

        geo_df[geo_df['Year'] == int(years[count])].plot(ax=i, markersize=1, color=colors[count], marker='+', alpha=0.5)
        for j in ['bottom', 'top', 'left', 'right']:
            side = i.spines[j]
            side.set_visible(False)
        i.set_title(years[count] + '\n({:,} Road Accident Cases)'.format(list(year_df.Cases)[count]), fontsize=12,
                    color='grey', weight='bold')
        i.axis('off')
        count += 1
        if year_df.shape[0] == count:
            break

    sns.lineplot(data=year_df, marker='o', x='Year', y='Cases', color='#734dff', ax=ax6, label="Yearly Road Accidents");

    for k in ['bottom', 'top', 'left', 'right']:
        side = ax6.spines[k]
        side.set_visible(False)
    ax6.xaxis.set_ticks(year_df.Year);
    ax6.legend(prop={'size': 12}, loc='best', edgecolor='white');
    plt.show()


# visualize_accident_percentage_over_years()

visualize_accident_cases_over_past_years()