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

visualize_traffic_impact_due_to_accidents()