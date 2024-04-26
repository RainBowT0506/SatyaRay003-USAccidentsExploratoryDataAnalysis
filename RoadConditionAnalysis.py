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

def visualize_road_condition_distribution():
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))

    road_conditions = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'Stop', 'No_Exit', 'Traffic_Signal', 'Turning_Loop']
    colors = [('#6662b3', '#00FF00'), ('#7881ff', '#0e1ce8'), ('#18f2c7', '#09ad8c'), ('#08ff83', '#02a352'),
              ('#ffcf87', '#f5ab3d'),
              ('#f5f53d', '#949410'), ('#ff9187', '#ffc7c2'), ('tomato', '#008000')]
    count = 0

    def func(pct, allvals):
        absolute = int(round(pct / 100 * np.sum(allvals), 2))
        return "{:.2f}%\n({:,d} Cases)".format(pct, absolute)

    for i in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:

        size = list(df[road_conditions[count]].value_counts())
        if len(size) != 2:
            size.append(0)

        labels = ['False', 'True']

        i.pie(size, labels=labels, colors=colors[count],
              autopct=lambda pct: func(pct, size), labeldistance=1.1,
              textprops={'fontsize': 12}, explode=[0, 0.2])

        title = '\nPresence of {}'.format(road_conditions[count])

        i.set_title(title, fontsize=18, color='grey')

        count += 1

    plt.show()

visualize_road_condition_distribution()
