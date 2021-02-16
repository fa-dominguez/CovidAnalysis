# %%
import os
import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np  # matematicas
import seaborn as sns  # regresiones
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

url = "./public_dataset_original.csv"
df = pd.read_csv(url, header=0)
path = "./public_dataset_2.csv"

# It is needed to transform dummies variables (Yes/No) in columns and categorize the answers ranked
columns_dummies = ["gender", "vaccinated_flu", "smoker", "access_openair_none", "access_openair_balcony",
                   "access_openair_terrace", "access_openair_smallgarden", "access_openair_biggarden",
                   "access_openair_commonyard", "access_openair_other", "traveled_abroad"]

for column_dum in columns_dummies:
    df = pd.get_dummies(df, columns=[column_dum])

columns_categories = ["age_group", "labor_status", "income_group", "contacted_doctor", "current_living_area",
                      "current_home", "current_living_arrangement", "freq_norm_car", "freq_norm_short_trans_norush",
                      "freq_norm_short_trans_rush", "freq_norm_long_trans", "freq_norm_plane", "travelled_to_wuhan",
                      "travelled_to_chinaother"]

for column_cat in columns_categories:
    df = pd.get_dummies(df, columns=[column_cat])

columns_Q = ["Q291_1", "Q291_2", "Q291_3", "Q291_4", "Q291_5", "Q291_6", "Q291_7", "Q291_8", "Q291_9", "Q291_10",
             "Q291_11", "Q291_12", "Q291_13", "Q291_14", "Q291_15", "Q292_1", "Q292_2", "Q292_3", "Q292_4", "Q292_5",
             "Q292_6", "Q292_7", "Q292_8", "Q292_9", "Q292_10", "Q292_11", "Q292_12", "Q292_13", "Q292_14", "Q292_15",
             "Q293_1", "Q293_2", "Q293_3", "Q293_4", "Q293_5", "Q293_6", "Q293_7", "Q293_8", "Q293_9", "Q293_10",
             "Q293_11", "Q293_12", "Q293_13", "Q293_14", "Q293_15"]
columns_Q_types = ["Never", "Rarely", "Sometimes", "Very often", "Always"]

for column_q in columns_Q:
    columns_Q_category = CategoricalDtype(categories=columns_Q_types, ordered=True)
    df[column_q] = df[column_q].astype(columns_Q_category)
    df[column_q] = df[column_q].cat.codes

# Washing hands belief, transform categorize in ranking
columns_washing_types = ["Not effective at all", "Slightly effective", "Moderately effective", "Very effective",
                         "Extremely effective"]

columns_washing_hands_category = CategoricalDtype(categories=columns_washing_types, ordered=True)
df["Washinghands_Effective"] = df["Washinghands_Effective"].astype(columns_washing_hands_category)
df["Washinghands_Effective"] = df["Washinghands_Effective"].cat.codes

# Beliefs on COVID consequences compared with Flu consequences
# Transform categorize variable in ranking
columns_consequence_covid_types = ["Much lower", "Lower", "About the same", "Higher", "Much higher"]
columns_consequence_covid_category = CategoricalDtype(categories=columns_consequence_covid_types, ordered=True)
df["consequences_covid_vs_flu"] = df["consequences_covid_vs_flu"].astype(columns_consequence_covid_category)
df["consequences_covid_vs_flu"] = df["consequences_covid_vs_flu"].cat.codes

# It is necessary to clean the data, deleting all the empty values, and transform in percentage.
principal = ["exposure"]
df[principal] = df[principal].replace("", np.nan)
df = df.dropna(axis=0, subset=['exposure'])

df[principal] = df[principal].astype(int)

df[principal] = df[principal] / 100

df.to_csv(path)
# %%
# Regresion 1 Covid Mortality First Regression
print("--------------Regresion 1--------------------------")
X1 = df[["Age_greater_46", "n_infected_interactions", "smoker_No", "Washinghands_Effective", "Infected_yourarea",
         "Infected_hosp", "exposure"]]
Y1 = df["Covid Mortality"]
X1 = sm.add_constant(X1)
modelreg1 = sm.OLS(Y1, X1).fit()
summary1 = modelreg1.summary()

corr = X1.corr()
print(summary1)

print(df[["Age_greater_46", "n_infected_interactions", "smoker_No", "Washinghands_Effective", "Infected_yourarea",
          "Infected_hosp", "exposure"]].describe())
print(corr)

# %%

# %%    
# Regresion 2 Hombre y Mujer en creencia de muerte
print("--------------Regresion 2--------------------------")
X2 = df[["smoker_No", "Washinghands_Effective", "Infected_yourarea", "Infected_hosp", "exposure"]]
Y2 = df["Covid Mortality"]
X2 = sm.add_constant(X2)
modelreg2 = sm.OLS(Y2, X2).fit()
summary2 = modelreg2.summary()

corr = X2.corr()
print(summary2)
# %%
# Linear regression plot
reg = linear_model.LinearRegression()
reg.fit(X2, Y2)
yhat = reg.predict(X2)

ax1 = sns.distplot(df['Covid Mortality'], color='r', label='Normal Value')
ax1.set(xlabel="Covid Mortality Awarness")
sns.distplot(yhat, color='b', label='Predicted Value', ax=ax1)
plt.legend()
plt.show()
# %%
