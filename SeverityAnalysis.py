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

# 建立一個嚴重程度的資料框和對應的事故案例
# create a dataframe of Severity and the corresponding accident cases
severity_df = pd.DataFrame(df['Severity'].value_counts()).rename(columns={'count': 'Cases'})
print(severity_df.columns)
states = gpd.read_file('Shapefiles/States_shapefile.shp')
geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

# 事故對交通的影響
def visualize_traffic_impact_due_to_accidents():
    fig = go.Figure(go.Funnelarea(
        text=["Severity - 2", "Severity - 3", "Severity - 4", "Severity - 1"],
        values=severity_df.Cases,
        title={"position": "top center",
               "text": "<b>Impact on the Traffic due to the Accidents</b>",
               'font': dict(size=18, color="#7f7f7f")},
        marker={"colors": ['#14a3ee', '#b4e6ee', '#fdf4b8', '#ff4f4e'],
                "line": {"color": ["#e8e8e8", "wheat", "wheat", "wheat"], "width": [7, 0, 0, 2]}}
    ))

    fig.show()

def visualize_severity_levels_on_US_map():
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim([-125, -65])
    ax.set_ylim([22, 55])
    states.boundary.plot(ax=ax, color='black');

    geo_df[geo_df['Severity'] == 1].plot(ax=ax, markersize=50, color='#5cff4a', marker='o', label='Severity 1');
    geo_df[geo_df['Severity'] == 3].plot(ax=ax, markersize=10, color='#ff1c1c', marker='x', label='Severity 3');
    geo_df[geo_df['Severity'] == 4].plot(ax=ax, markersize=1, color='#6459ff', marker='v', label='Severity 4');
    geo_df[geo_df['Severity'] == 2].plot(ax=ax, markersize=5, color='#ffb340', marker='+', label='Severity 2');

    for i in ['bottom', 'top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)

    plt.title('\nDifferent level of Severity visualization in US map', size=20, color='grey');

    One = mpatches.Patch(color='#5cff4a', label='Severity 1')
    Two = mpatches.Patch(color='#ffb340', label='Severity 2')
    Three = mpatches.Patch(color='#ff1c1c', label='Severity 3')
    Four = mpatches.Patch(color='#6459ff', label='Severity 4')

    ax.legend(handles=[One, Two, Three, Four], prop={'size': 15}, loc='lower right', borderpad=1,
              labelcolor=['#5cff4a', '#ffb340', '#ff1c1c', '#6459ff'], edgecolor='white');
    plt.show()

# visualize_traffic_impact_due_to_accidents()

visualize_severity_levels_on_US_map()