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

# 建立城市及其相應事故案例的資料框
# create a dataframe of city and their corresponding accident cases
city_df = pd.DataFrame(df['City'].value_counts()).reset_index().rename(columns={'count': 'Cases'})
# 只保留 'Cases' 欄位
top_10_cities = pd.DataFrame(city_df.head(10))


def plot_top_accident_cities():
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


# 計算事故率或事故發生頻率
def cal_accident_rate():
    hightest_cases = city_df.Cases[0]
    print(round(hightest_cases / 5))
    print(round(hightest_cases / (5 * 365)))


def visualize_top_10_accident_cities_US():
    states = gpd.read_file('Shapefiles/States_shapefile.shp')

    def lat(city):
        address = city
        geolocator = Nominatim(user_agent="Your_Name")
        location = geolocator.geocode(address)
        return (location.latitude)

    def lng(city):
        address = city
        geolocator = Nominatim(user_agent="Your_Name")
        location = geolocator.geocode(address)
        return (location.longitude)

    # list of top 10 cities
    top_ten_city_list = list(city_df.City.head(10))

    top_ten_city_lat_dict = {}
    top_ten_city_lng_dict = {}
    for i in top_ten_city_list:
        top_ten_city_lat_dict[i] = lat(i)
        top_ten_city_lng_dict[i] = lng(i)

    top_10_cities_df = df[df['City'].isin(list(top_10_cities.City))]

    top_10_cities_df['New_Start_Lat'] = top_10_cities_df['City'].map(top_ten_city_lat_dict)
    top_10_cities_df['New_Start_Lng'] = top_10_cities_df['City'].map(top_ten_city_lng_dict)

    geometry_cities = [Point(xy) for xy in zip(top_10_cities_df['New_Start_Lng'], top_10_cities_df['New_Start_Lat'])]
    geo_df_cities = gpd.GeoDataFrame(top_10_cities_df, geometry=geometry_cities)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim([-125, -65])
    ax.set_ylim([22, 55])
    states.boundary.plot(ax=ax, color='grey');

    colors = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#aaffc3', '#42d4f4', '#4363d8', '#911eb4',
              '#f032e6']
    markersizes = [50 + (i * 20) for i in range(10)][::-1]
    for i in range(10):
        geo_df_cities[geo_df_cities['City'] == top_ten_city_list[i]].plot(ax=ax, markersize=markersizes[i],
                                                                          color=colors[i], marker='o',
                                                                          label=top_ten_city_list[i], alpha=0.7);

    plt.legend(prop={'size': 13}, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), edgecolor='white', title="Cities",
               title_fontsize=15);

    for i in ['bottom', 'top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)

    plt.title('\nVisualization of Top 10 Accident Prone Cities in US (2016-2020)', size=20, color='grey');
    plt.show()


# 分析城市中的車禍案例數量的百分比。
def city_cases_percentage(val, operator):
    if operator == '<':
        res = city_df[city_df['Cases'] < val].shape[0]
    elif operator == '>':
        res = city_df[city_df['Cases'] > val].shape[0]
    elif operator == '=':
        res = city_df[city_df['Cases'] == val].shape[0]
    print(f'{res} Cities, {round(res * 100 / city_df.shape[0], 2)}%')

# 分析城市中的車禍案例數量的百分比
def city_cases_percentage_analysis():
    city_cases_percentage(1, '=')
    city_cases_percentage(100, '<')
    city_cases_percentage(1000, '<')
    city_cases_percentage(1000, '>')
    city_cases_percentage(5000, '>')
    city_cases_percentage(10000, '>')


# plot_top_accident_cities(city_df)

# cal_accident_rate(city_df)

# visualize_top_10_accident_cities_US()

# city_cases_percentage_analysis()
