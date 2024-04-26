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


states = gpd.read_file('Shapefiles/States_shapefile.shp')
geometry = [Point(xy) for xy in zip(df['Start_Lng'], df['Start_Lat'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)
accident_severity_df = geo_df.groupby(['Year', 'Severity']).size().unstack()
year_df = pd.DataFrame(df.Start_Time.dt.year.value_counts()).reset_index().rename(columns={'Start_Time':'Year', 'count':'Cases'}).sort_values(by='Cases', ascending=True)

print("geo_df")
print(geo_df.columns)
print(geo_df.head(10))

# 美國過去 5 年的嚴重程度和相應事故百分比
def visualize_severity_percentage():
    ax = accident_severity_df.plot(kind='barh', stacked=True, figsize=(12, 6),
                                   color=['#fcfa5d', '#ffe066', '#fab666', '#f68f6a'],
                                   rot=0);

    ax.set_title('\nSeverity and Corresponding Accident \nPercentage for past 5 years in US\n', fontsize=20,
                 color='grey');

    for i in ['top', 'left', 'right']:
        side = ax.spines[i]
        side.set_visible(False)

    ax.spines['bottom'].set_bounds(0, 800000);
    ax.set_ylabel('\nYears\n', fontsize=15, color='grey');
    ax.set_xlabel('\nAccident Cases\n', fontsize=15, color='grey');
    ax.legend(prop={'size': 12.5}, loc='best', fancybox=True, title="Severity", title_fontsize=15, edgecolor='white');
    ax.tick_params(axis='both', which='major', labelsize=12.5)
    # ax.set_facecolor('#e6f2ed')

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        var = width * 100 / df.shape[0]
        if var > 0:
            if var > 4:
                ax.text(x + width / 2,
                        y + height / 2 - 0.05,
                        '{:.2f}%'.format(width * 100 / df.shape[0]),
                        fontsize=12, color='black', alpha=0.8)
            elif var > 1.8 and var < 3.5:
                ax.text(x + width / 2 - 17000,
                        y + height / 2 - 0.05,
                        '{:.2f}%'.format(width * 100 / df.shape[0]),
                        fontsize=12, color='black', alpha=0.8)
            elif var > 1.5 and var < 1.8:
                ax.text(x + width / 2 + 7000,
                        y + height / 2 - 0.05,
                        '  {:.2f}%'.format(width * 100 / df.shape[0]),
                        fontsize=12, color='black', alpha=0.8)
            elif var > 1:
                ax.text(x + width / 2 - 20000,
                        y + height / 2 - 0.05,
                        '  {:.2f}%'.format(width * 100 / df.shape[0]),
                        fontsize=12, color='black', alpha=0.8)
            else:
                ax.text(x + width / 2 + 10000,
                        y + height / 2 - 0.05,
                        '  {:.2f}%'.format(width * 100 / df.shape[0]),
                        fontsize=12, color='black', alpha=0.8)

    plt.show()

# 美國平均事故案例（2016-2020）
def visualize_accident_rates():
    year_df['accident/day'] = round(year_df['Cases'] / (5 * 365))
    year_df['accident/hour'] = round(year_df['Cases'] / (5 * 365 * 24))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6), dpi=80)

    count = 0
    plots = ['accident/day', 'accident/hour']
    plots_limit = [(-10, 500), (-0.5, 22.5)]
    plots_bound = [(0, 500), (0, 20)]
    plot_text = [60, 2.5]

    colors = [['#ffb74b', '#ffd6a4', '#ceb1f2', '#a071ff', '#6f71f7'],
              ['#cd7cf2', '#f27ec8', '#fa70b3', '#ff5e86', '#ff1732']]

    for i in [ax1, ax2]:

        sns.barplot(ax=i, y=year_df[plots[count]], x=year_df['Year'], palette=colors[count])

        var = plots[count].split('/')[-1].capitalize()

        for j in i.patches:
            i.text(j.get_x() + 0.06, j.get_height() - plot_text[count], \
                   str(int(j.get_height())) + '\nAccidents\nPer {}'.format(var), fontsize=8.5, color='white',
                   weight='bold')

        i.axes.set_ylim(plots_limit[count])
        i.axes.set_ylabel('\nAccident Cases\n', fontsize=12, color='grey')
        i.axes.set_xlabel('\nYears\n', fontsize=12, color='grey')
        i.tick_params(axis='both', which='major', labelsize=10)

        i.set_title('\nAverage Cases \nof Accident/{} in US (2016-2020)\n'.format(var), fontsize=15, color='grey')
        i.spines['bottom'].set_bounds(0.005, 4)
        i.spines['left'].set_bounds(plots_bound[count])

        for k in ['top', 'right']:
            side = i.spines[k]
            side.set_visible(False)

        i.set_axisbelow(True)
        MA = mpatches.Patch(color=colors[count][-1], label='Year with Maximum\n no. of Road Accidents')
        MI = mpatches.Patch(color=colors[count][0], label='Year with Minimum\n no. of Road Accidents')
        i.legend(handles=[MA, MI], prop={'size': 10}, loc='best', borderpad=1,
                 labelcolor=[colors[count][-1], colors[count][0]], edgecolor='white');
        count += 1

        plt.show()

# visualize_severity_percentage()

visualize_accident_rates()