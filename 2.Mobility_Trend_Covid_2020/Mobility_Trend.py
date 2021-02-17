# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np  # matematicas
import seaborn as sns  # regresiones
import calendar
from scipy import stats  # stadisticas
from sklearn import linear_model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats
import statsmodels.api as sm

url = "./applemobilitytrends-2020-11-14_UnitedStates_2.csv"
df = pd.read_csv(url, header=0)
path = "./applemobilitytrends-2020-11-14_UnitedStates_3.csv"

# Regresion 1
df = df.set_index("Date")
df = df.reset_index()
df["Date"] = pd.to_datetime(df["Date"])
print(df.head())

x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 25000)
ax.set_ylim(0, 150)
ax.set_xlabel('Amount of COVID cases (US)')
ax.set_ylabel("Average Mobility (US)")
ax.set_title('Mobility Trend until:')

def animation_frame(i):
    x = df['Number_of_Cases'][i]
    y = df['Mobility Average'][i]
    x_data.append(x)
    y_data.append(y)
    plt.cla()
    plt.xlim(0, 25000)
    plt.ylim((0, 150))
    plot1 = sns.regplot(x=x_data, y=y_data)
    sns.set(palette='Set2')
    month_numb = df['Date'][i].month
    day_numb = df['Date'][i].day
    year_numb = df['Date'][i].year
    month_name = calendar.month_abbr[month_numb]
    plt.title("Mobility Trend until: " + str(day_numb) + ' ' + str(month_name) + ' ' + str(year_numb))
    plt.xlabel("Amount of COVID cases (US)")
    plt.ylabel("Average Mobility (US)")

    return plot1


animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 279, 1), interval=0.1)
plt.show()

