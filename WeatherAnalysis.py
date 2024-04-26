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
df = df.dropna(subset=['Wind_Chill(F)'])

# 將 'End_Time' 和 'Start_Time' 轉換為日期時間對象
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
pd.set_option('display.max_columns', None)  # 顯示所有列
pd.set_option('display.max_rows', None)     # 顯示所有行
print(df.columns)
print(df.head(10))

def generate_intervals_labels(attribute, split, gap):
    var_min = min(df[attribute])
    intervals = [int(var_min)]
    labels = []
    for i in range(1, split + 1):

        lower_limit = int(var_min + ((i - 1) * gap))

        if i == split:
            upper_limit = int(max(df[attribute]))
        else:
            upper_limit = int(var_min + (i * gap))

        # 修正：確保 upper_limit 大於 lower_limit
        if upper_limit <= lower_limit:
            upper_limit = lower_limit + 1

        # intervals
        intervals.append(upper_limit)

        # labels
        label_var = '({} to {})'.format(lower_limit, upper_limit)
        labels.append(label_var)

    return intervals, labels


def Feature_Bin_Plot(dataframe, attribute, clrs, intervals, labels, fig_size, font_size, y_lim, adjust, title):
    new_df = dataframe.copy()
    xlabel = 'Different {} Grouped Value'.format(attribute)
    new_df[xlabel] = pd.cut(x=new_df[attribute], bins=intervals, labels=labels, include_lowest=True)
    temp_df = pd.DataFrame(new_df[xlabel].value_counts()).reset_index() \
        .rename(columns={ xlabel : 'Bins', 'count': 'Cases'}).sort_values('Bins')

    print(temp_df.columns)

    count, max_index = 0, 0
    cases_list = list(temp_df['Cases'])
    for i in cases_list:
        if i == max(temp_df['Cases']):
            max_index = count
            break
        count += 1

    total = len(new_df[xlabel])
    plt.figure(figsize=fig_size)

    #     clrs = ['mediumspringgreen' if (x < max(temp_df['Cases'])) else 'grey' for x in temp_df['Cases']]
    cmap = cm.get_cmap(clrs, len(intervals))
    clrs = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    ax = sns.barplot(y=temp_df['Cases'], x=temp_df['Bins'], palette=clrs);

    for i in ax.patches:
        ax.text(i.get_x() + adjust[0], i.get_height() + adjust[-1], \
                '{:,d}\nCases\n({}%) '.format(int(i.get_height()), round(100 * i.get_height() / total, 2)),
                fontsize=font_size,
                color='black')

    plt.title(title, size=20, color='grey')
    plt.ylim(y_lim)

    for i in ['bottom', 'top', 'left', 'right']:
        ax.spines[i].set_color('white')
        ax.spines[i].set_linewidth(1.5)

    ax.set_xlabel('\n{}\n'.format(xlabel), fontsize=15, color='grey')
    ax.set_ylabel('\nAccident Cases\n', fontsize=15, color='grey')

    ax.set_axisbelow(True)
    ax.grid(color='#b2d6c7', linewidth=1, alpha=.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    MA = mpatches.Patch(color=clrs[max_index], label='{} Range with Maximum\n no. of Road Accidents'.format(attribute))
    ax.legend(handles=[MA], prop={'size': 10.5}, loc='best', borderpad=1,
              labelcolor=[clrs[max_index]], edgecolor='white');

    plt.show()


# 不同溫度範圍的百分比
temp_intervals, temp_labels = generate_intervals_labels('Temperature(F)', 9, 30)
Feature_Bin_Plot(df, 'Temperature(F)', 'gist_ncar', temp_intervals, temp_labels,
                 (12, 6), 14, (-20000, 800000), [0.01, 10000], '\nPercentage of different Temperature range\n')


# 不同濕度範圍的百分比
Humidity_intervals, Humidity_labels = generate_intervals_labels('Humidity(%)', 10, 10)
Feature_Bin_Plot(df, 'Humidity(%)', 'magma', Humidity_intervals, Humidity_labels,
                 (12, 6), 14, (-20000, 500000), [0.01, 10000], '\nPercentage of different Humidity range\n')

# 不同壓力範圍的百分比
Pressure_intervals, Pressure_labels = generate_intervals_labels('Pressure(in)', 6, 10)
Feature_Bin_Plot(df, 'Pressure(in)', 'Paired', Pressure_intervals, Pressure_labels,
                 (12, 6), 14, (-20000, 1500000), [0.01, 10000], '\nPercentage of different Pressure range\n')

# 不同風寒範圍的百分比
Wind_Chill_intervals, Wind_Chill_labels = generate_intervals_labels('Wind_Chill(F)', 10, 20)
Feature_Bin_Plot(df, 'Wind_Chill(F)', 'inferno', Wind_Chill_intervals, Wind_Chill_labels,
                 (12, 6), 14, (-20000, 700000), [0.01, 10000], '\nPercentage of different Wind Chill range\n')

# 不同風速範圍的百分比
Wind_Speed_intervals, Wind_Speed_labels = generate_intervals_labels('Wind_Speed(mph)', 10, 5)
Feature_Bin_Plot(df, 'Wind_Speed(mph)', 'turbo',Wind_Speed_intervals, Wind_Speed_labels,
                 (12, 6), 14, (-20000, 900000), [0.01, 10000], '\nPercentage of different Wind Speed range\n')


Visibility_intervals, Visibility_labels = generate_intervals_labels('Visibility(mi)', 12, 1)
Feature_Bin_Plot(df, 'Visibility(mi)', 'prism', Visibility_intervals, Visibility_labels,
                 (12, 6), 14, (-20000, 1900000), [0.01, 30000], '\nPercentage of different Visibility range\n')