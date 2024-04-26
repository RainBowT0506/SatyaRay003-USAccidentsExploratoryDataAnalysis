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

# 使用美國州代碼及其對應的名稱建立字典
# create a dictionary using US State code and their corresponding Name
us_states = {'AK': 'Alaska',
             'AL': 'Alabama',
             'AR': 'Arkansas',
             'AS': 'American Samoa',
             'AZ': 'Arizona',
             'CA': 'California',
             'CO': 'Colorado',
             'CT': 'Connecticut',
             'DC': 'District of Columbia',
             'DE': 'Delaware',
             'FL': 'Florida',
             'GA': 'Georgia',
             'GU': 'Guam',
             'HI': 'Hawaii',
             'IA': 'Iowa',
             'ID': 'Idaho',
             'IL': 'Illinois',
             'IN': 'Indiana',
             'KS': 'Kansas',
             'KY': 'Kentucky',
             'LA': 'Louisiana',
             'MA': 'Massachusetts',
             'MD': 'Maryland',
             'ME': 'Maine',
             'MI': 'Michigan',
             'MN': 'Minnesota',
             'MO': 'Missouri',
             'MP': 'Northern Mariana Islands',
             'MS': 'Mississippi',
             'MT': 'Montana',
             'NC': 'North Carolina',
             'ND': 'North Dakota',
             'NE': 'Nebraska',
             'NH': 'New Hampshire',
             'NJ': 'New Jersey',
             'NM': 'New Mexico',
             'NV': 'Nevada',
             'NY': 'New York',
             'OH': 'Ohio',
             'OK': 'Oklahoma',
             'OR': 'Oregon',
             'PA': 'Pennsylvania',
             'PR': 'Puerto Rico',
             'RI': 'Rhode Island',
             'SC': 'South Carolina',
             'SD': 'South Dakota',
             'TN': 'Tennessee',
             'TX': 'Texas',
             'UT': 'Utah',
             'VA': 'Virginia',
             'VI': 'Virgin Islands',
             'VT': 'Vermont',
             'WA': 'Washington',
             'WI': 'Wisconsin',
             'WV': 'West Virginia',
             'WY': 'Wyoming'}

warnings.filterwarnings('ignore')

# 讀取資料集並將其載入到 pandas 資料框中
# read & load the dataset into pandas dataframe
df = pd.read_csv('small_data.csv', encoding='ISO-8859-1')

# create a dataframe of State and their corresponding accident cases
state_df = pd.DataFrame(df['State'].value_counts()).reset_index().rename(columns={'count': 'Cases'})

print(state_df)


# Function to convert the State Code with the actual corressponding Name
def convert(x): return us_states[x]


state_df['State'] = state_df['State'].apply(convert)

top_ten_states_name = list(state_df['State'].head(10))
states = gpd.read_file('Shapefiles/States_shapefile.shp')


# 最多的前 10 州。 美國事故案例數（2016-2020 年）
def top_10_states_accident_cases_visualization():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=80)

    cmap = cm.get_cmap('winter', 10)
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=state_df['Cases'].head(10), x=state_df['State'].head(10), palette='winter')
    ax1 = ax.twinx()
    sns.lineplot(data=state_df[:10], marker='o', x='State', y='Cases', color='white', alpha=.8)

    total = df.shape[0]
    for i in ax.patches:
        ax.text(i.get_x() - 0.2, i.get_height() + 10000, \
                ' {:,d}\n  ({}%) '.format(int(i.get_height()), round(100 * i.get_height() / total, 1)), fontsize=15,
                color='black')

    ax.set(ylim=(-10000, 600000))
    ax1.set(ylim=(-100000, 1700000))

    plt.title('\nTop 10 States with most no. of \nAccident cases in US (2016-2020)\n', size=20, color='grey')
    ax1.axes.yaxis.set_visible(False)
    ax.set_xlabel('\nStates\n', fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    for i in ['top', 'right']:
        side1 = ax.spines[i]
        side1.set_visible(False)
        side2 = ax1.spines[i]
        side2.set_visible(False)

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, axis='y', alpha=.3)

    ax.spines['bottom'].set_bounds(0.005, 9)
    ax.spines['left'].set_bounds(0, 600000)
    ax1.spines['bottom'].set_bounds(0.005, 9)
    ax1.spines['left'].set_bounds(0, 600000)
    ax.tick_params(axis='y', which='major', labelsize=10.6)
    ax.tick_params(axis='x', which='major', labelsize=10.6, rotation=10)

    MA = mpatches.Patch(color=clrs[0], label='State with Maximum\n no. of Road Accidents')
    ax.legend(handles=[MA], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=clrs[0], edgecolor='white');
    plt.show()


# 美國 10 個事故多發州的視覺化（2016-2020 年）
def visualize_top_accident_prone_states_US():
    geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)

    print(geo_df.columns)
    print(geo_df["Start_Time"])
    geo_df['Start_Time'] = pd.to_datetime(geo_df['Start_Time'])
    geo_df['year'] = geo_df["Start_Time"].dt.year

    geo_df['State'] = geo_df['State'].apply(convert)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim([-125, -65])
    ax.set_ylim([22, 55])

    print(states.columns)

    states.boundary.plot(ax=ax, color='grey');
    states.apply(lambda x: None
    if (x.State_Name not in top_ten_states_name)
    else ax.annotate(
        s=x.State_Name,
        xy=x.geometry.centroid.coords[
            0], ha='center',
        color='black', weight='bold',
        fontsize=12.5), axis=1);

    # CFOTNYMVNPI
    colors = ['#FF5252', '#9575CD', '#FF8A80', '#FF4081', '#FFEE58', '#7C4DFF', '#00E5FF', '#81D4FA', '#64FFDA',
              '#8C9EFF']
    count = 0
    for i in list(state_df['State'].head(10)):
        geo_df[geo_df['State'] == i].plot(ax=ax, markersize=1, color=colors[count], marker='o');
        count += 1

    for i in ['bottom', 'top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)

    plt.title('\nVisualization of Top 10 Accident Prone States in US (2016-2020)', size=20, color='grey');
    plt.show()

# top_10_states_accident_cases_visualization()

visualize_top_accident_prone_states_US()
