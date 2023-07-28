# Simple random forests model for the Meyers Lab for
#   rapid time-sensitive malaria risk assessment in United States
# Also used to validate MaxEnt model (maxnet package in R)
#
# Supervised by Spencer Fox
# Guided by Jose Luis Herrera Diestra
# Data from Jose Luis Herrera Diestra, Oluwasegun Michael Ibrahim,
#   and Mathew Abraham
# Maximum entropy implementation from Jose Luis Herrera Diestra
#
###################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Jose says
# Of all these variables, the ones I used in the last model
#   (due to their similarity with the ones used in the zika paper) are:
#   GDP,PER_PROF,maxTemp,PER_POV,preciMax,POP_GRAD,PER_NO_INS,PER_COMM,PER_MALE

selected_variable_names = ["GDP", "PER_PROF", "maxTemp", "PER_POV", "preciMax",
                           "POP_GRAD", "PER_NO_INS", "PER_COMM", "PER_MALE"]

variables2016 = pd.read_csv("allVariables2016.csv")
variables2017 = pd.read_csv("allVariables2017.csv")

variables2016.sort_values(by="GEOID", inplace=True)
variables2017.sort_values(by="GEOID", inplace=True)

# These dataframes only contain counties with occurrences

occurrences2016 = pd.read_csv("occurrences2016.csv")
occurrences2017 = pd.read_csv("occurrences2017.csv")

occurrences2016.rename(columns={"FIPS":"GEOID"}, inplace=True)
occurrences2017.rename(columns={"FIPS":"GEOID"}, inplace=True)

variables2016 = pd.merge(variables2016,occurrences2016, how='left', on = 'GEOID', suffixes = ('_left', '_right'))
variables2016 = variables2016.fillna(0)
variables2016.rename(columns={"MAL_FREQ_2016":"NUM_IMPORTS"},inplace=True)

variables2017 = pd.merge(variables2017,occurrences2017, how='left', on = 'GEOID', suffixes = ('_left', '_right'))
variables2017 = variables2017.fillna(0)
variables2017.rename(columns={"MAL_FREQ_2017":"NUM_IMPORTS"},inplace=True)

originalvariables2016 = variables2016.copy(deep=True)
originalvariables2017 = variables2017.copy(deep=True)

variables2016 = variables2016[selected_variable_names + ["GEOID", "NUM_IMPORTS"]]
variables2017 = variables2017[selected_variable_names + ["GEOID", "NUM_IMPORTS"]]

breakpoint()

# Spencer
# 1. Fit maxent and RF models using the 2016 importation data
# 2. Make importation probability predictions for 2017 with both models
# 3. Compare those predictions with the actual probability distribution with the scatterplots and
#       maybe some summary statistics (e.g. R^2, correlation, MAE, etc.)?
# 4. Use whichever model seems to have performed best for the analysis

###################################################################################################

# To get states

counties2017 = list(originalvariables2017["COUNTY_left"])
states2017 = []

for county in counties2017:
    states2017.append(county.split(", ")[1])

rf_results = pd.read_csv("malaria_probability_next_import_comes_from_county.csv")
rf_results["State"] = states2017
rf_results["Reported Counts"] = variables2017["NUM_IMPORTS"]

breakpoint()

counts_nonzero = np.array(variables2017["NUM_IMPORTS"] > 0)

rf_results = pd.merge(rf_results[~counts_nonzero].groupby("State").sum(), rf_results[counts_nonzero].groupby("State").sum(), how="left", on="State",
                      suffixes=("left", "right"))
rf_results = rf_results.fillna(0)

rf_results.to_csv("rf_results.csv")

breakpoint()

###################################################################################################

# New stuff for averaging 2016 and 2017 model

rfr2016 = RandomForestRegressor(n_estimators=5000, max_depth=5)
rfr2016.fit(variables2016.loc[:, variables2016.columns != "NUM_IMPORTS"], variables2016["NUM_IMPORTS"])

rfr2017 = RandomForestRegressor(n_estimators=5000, max_depth=5)
rfr2017.fit(variables2017.loc[:, variables2017.columns != "NUM_IMPORTS"], variables2017["NUM_IMPORTS"])

y_test2016 = variables2016.loc[:, variables2016.columns != "NUM_IMPORTS"]
y_pred2016 = rfr2016.predict(y_test2016)
y_pred2016_scaled = y_pred2016 / np.sum(y_pred2016)

y_test2017 = variables2017.loc[:, variables2017.columns != "NUM_IMPORTS"]
y_pred2017 = rfr2017.predict(y_test2017)
y_pred2017_scaled = y_pred2017 / np.sum(y_pred2017)

y_pred_final_scaled = (y_pred2016_scaled + y_pred2017_scaled)/2

y_actual2016 = variables2016["NUM_IMPORTS"]
y_actual_scaled2016 = np.array(y_actual2016) / np.sum(np.array(y_actual2016))

y_actual2017 = variables2017["NUM_IMPORTS"]
y_actual_scaled2017 = np.array(y_actual2017) / np.sum(np.array(y_actual2017))

df_final = pd.DataFrame({"FIPS": variables2016["GEOID"], "Next Import Probability": y_pred_final_scaled})
df_final.to_csv("malaria_probability_next_import_comes_from_county.csv")

breakpoint()

###################################################################################################

# Old stuff for training on 2016 and testing on 2017

breakpoint()

rfr = RandomForestRegressor(n_estimators=1000, max_depth=5)
rfr.fit(variables2016.loc[:, variables2016.columns != "NUM_IMPORTS"], variables2016["NUM_IMPORTS"])
y_test = variables2017.loc[:, variables2017.columns != "NUM_IMPORTS"]
y_pred = rfr.predict(y_test)
y_actual = variables2017["NUM_IMPORTS"]

y_pred_scaled = y_pred / np.sum(y_pred)
y_actual_scaled = np.array(y_actual) / np.sum(np.array(y_actual))

np.savetxt("y_pred_scaled_2017.csv", y_pred_scaled, delimiter=",")
np.savetxt("y_actual_scaled_2017.csv", y_actual_scaled, delimiter=",")

def BC(x,y):
    return np.sum(np.sqrt(np.multiply(x, y)))

breakpoint()

plt.scatter(variables2017["NUM_IMPORTS"], y_pred)
plt.xlabel("Observed # imports 2017")
plt.ylabel("Predicted # imports 2017")
plt.title("Predicted vs observed 2017 imports")
caption = "Predictions generated from simple random forest, no CV, 1000 estimators, 5 max depth, fit to 2016 predictors and occurrences."
plt.text(80, 5, caption, wrap=True, ha="center", fontsize=8)
plt.savefig("scatterplot.png", dpi=1200)

output = pd.DataFrame([])
output["GEOID"] = variables2017["GEOID"]
output["PredictedImports2017"] = y_pred
output["ActualImports2017"] = variables2017["NUM_IMPORTS"]

output.to_csv("predicted_imports_2017_randomforest.csv")

breakpoint()

