# Simple random forests model for the Meyers Lab for
#   rapid time-sensitive malaria risk assessment in United States
# Also used to validate MaxEnt model (maxnet package in R)
# Linda Pei 2023
#
# This code uses 2016 & 2017 per-county data to generate per-county
#   "next importation probability" estimations for malaria cases in the contiguous US.
#   We define "next importation probability" for a county as the probability that
#   the next importation of malaria occurs in that county.
# We use in-sample 2016 and 2017 predictions for our estimates.
# We use a Monte Carlo simulation approach to randomly fill in missing cases
#   for each state. Missing cases are the difference between the reported cases
#   in a state and the sum of the per-county reported cases for counties in that state.
#   The per-county reported cases were only documented for counties with 5 or more cases,
#   resulting in missing values for counties with fewer than 5 cases.
#
# Supervised by Spencer Fox
# Guided by Jose Luis Herrera Diestra
# Data from Jose Luis Herrera Diestra, Oluwasegun Michael Ibrahim,
#   and Mathew Abraham
# Maximum entropy implementation from Jose Luis Herrera Diestra
#
###################################################################################################

# Currently uses MPI to quickly obtain multiple simulation replications
# To run serially, comment out all MPI imports and variable definitions
#   and similarly remove or redefine "rank" variable
# If run serially, consider changing num_reps to a smaller number so that runtime
#   is manageable on a personal laptop

###################################################################################################

# Overview of data streams

# occurrences2016.csv and occurrences2017.csv are from
#   https://data.cdc.gov/w/r3zz-ivb8/tdwk-ruhb
#   https://data.cdc.gov/widgets/puzh-5wax
#   respectively
# CDC data corresponding to malaria cases per US county
# However, only counties with more than 5 cases are recorded
# "MAL_FREQ_2016" (or "MAL_FREQ_2017") column has
#   number of cases -- we assume all cases are importations

# Therefore cases and imports are used interchangeably
#   in this document

# variables2016.csv and variables2017.csv contain
#   per-county predictor variables

# imported_cases_by_state_2016.csv and imported_cases_by_state_2017.csv
#   contain total number of cases per state

###################################################################################################

# Imports and MPI setup

import pandas as pd
import numpy as np
import time

from collections import Counter

# import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import copy

from mpi4py import MPI

comm = MPI.COMM_WORLD
num_processors_evaluation = comm.Get_size()
rank = comm.Get_rank()

###################################################################################################

# Notes from group discussions

# Jose says
# Of all these variables, the ones I used in the last model
#   (due to their similarity with the ones used in the zika paper) are:
#   GDP,PER_PROF,maxTemp,PER_POV,preciMax,POP_GRAD,PER_NO_INS,PER_COMM,PER_MALE

###################################################################################################

# This section creates datasets of predictor variables + case counts for 2016 and 2017

selected_variable_names = ["GDP", "PER_PROF", "maxTemp", "PER_POV", "preciMax",
                           "POP_GRAD", "PER_NO_INS", "PER_COMM", "PER_MALE"]

variables2016 = pd.read_csv("allVariables2016.csv")
variables2017 = pd.read_csv("allVariables2017.csv")

# Sort datasets by GEOID
variables2016.sort_values(by="GEOID", inplace=True)
variables2017.sort_values(by="GEOID", inplace=True)

# These dataframes only contain counties with 5 or more occurrences,
#   not all counties
occurrences2016 = pd.read_csv("occurrences2016.csv")
occurrences2017 = pd.read_csv("occurrences2017.csv")

occurrences2016.rename(columns={"FIPS":"GEOID"}, inplace=True)
occurrences2017.rename(columns={"FIPS":"GEOID"}, inplace=True)

# Add occurrences (case counts) to datasets with predictor variables
# Append according to "GEOID" column, and fill in NA with 0
variables2016 = pd.merge(variables2016,occurrences2016,
                         how='left', on = 'GEOID', suffixes = ('_left', '_right'))
variables2016 = variables2016.fillna(0)
variables2016.rename(columns={"MAL_FREQ_2016":"REPORTED_IMPORTS"},inplace=True)

variables2017 = pd.merge(variables2017,occurrences2017,
                         how='left', on = 'GEOID', suffixes = ('_left', '_right'))
variables2017 = variables2017.fillna(0)
variables2017.rename(columns={"MAL_FREQ_2017":"REPORTED_IMPORTS"},inplace=True)

# Create (deep) copies of datasets with predictor variables + case counts
originalvariables2016 = variables2016.copy(deep=True)
originalvariables2017 = variables2017.copy(deep=True)

variables2016 = variables2016[selected_variable_names + ["GEOID", "REPORTED_IMPORTS"]]
variables2017 = variables2017[selected_variable_names + ["GEOID", "REPORTED_IMPORTS"]]

###################################################################################################

# This section computes the number of missing cases for each state

# The number of missing cases for each state for 2016 (and analogously for 2017) is
#   Total number of cases in state (data from imported_cases_by_state_2016.csv)
#   minus sum of reported cases per county for counties in that state
#   (data from occurrences2016.csv)

# Missing cases for each state come from the fact that per-county cases
#   are only reported for counties with 5 or more cases, so
#   per-county cases are not reported for counties with fewer than 5 cases

# Creates missing_cases_state_2016 and missing_cases_state_2017
# Creates dictionaries for 2016, 2017 mapping state to number of missing imports

# Note: Washington DC FIPS code 11001 is not in occurrences data
#   because it is not technically a county

# Read data for actual total number of imports by state for continental US
actual_imports_state_2016 = pd.read_csv("imported_cases_by_state_2016.csv")

counties2016 = list(originalvariables2016["COUNTY_left"])
states2016 = []

# "COUNTY_left" column values have format County Name, State
#   so we extract the state only and save it as a string
for county in counties2016:
    states2016.append(county.split(", ")[1])
variables2016["State"] = states2016
variables2016["State"] = variables2016["State"].astype("str")

# Compute "REPORTED_IMPORTS" per state -- which is the sum of reported per-county cases
#   across counties in each state
# Create column "MISSING_IMPORTS" for the number of missing cases/imports
missing_cases_state_2016 = pd.merge(variables2016.groupby("State").sum(),
                                    actual_imports_state_2016,
                                    how="left", on="State", suffixes=("left", "right"))
missing_cases_state_2016.rename(columns={"Imported cases":"ACTUAL_IMPORTS"},inplace=True)
missing_cases_state_2016["MISSING_IMPORTS"] = \
    missing_cases_state_2016["ACTUAL_IMPORTS"] - missing_cases_state_2016["REPORTED_IMPORTS"]
missing_cases_state_2016 = missing_cases_state_2016[["State", "MISSING_IMPORTS"]]

# Create a dictionary that maps the state name (string) to the number of missing imports/cases (integer)
missing_cases_state_2016 = dict(zip(missing_cases_state_2016["State"],
                                    missing_cases_state_2016["MISSING_IMPORTS"]))

# This is the same as the 2016 case but for 2017 data
# Sorry, this is research code done quickly, so I am violating the
#   "do not repeat yourself" rule :)
# TODO: create function to do for 2016, 2017 instead of repeating 2016 code for 2017

actual_imports_state_2017 = pd.read_csv("imported_cases_by_state_2017.csv")

counties2017 = list(originalvariables2017["COUNTY_left"])
states2017 = []

for county in counties2017:
    states2017.append(county.split(", ")[1])
variables2017["State"] = states2017
variables2017["State"] = variables2017["State"].astype("str")

missing_cases_state_2017 = pd.merge(variables2017.groupby("State").sum(),
                                    actual_imports_state_2017,
                                    how="left", on="State", suffixes=("left", "right"))
missing_cases_state_2017.rename(columns={"Imported cases":"ACTUAL_IMPORTS"},inplace=True)
missing_cases_state_2017["MISSING_IMPORTS"] = \
    missing_cases_state_2017["ACTUAL_IMPORTS"] - missing_cases_state_2017["REPORTED_IMPORTS"]
missing_cases_state_2017 = missing_cases_state_2017[["State", "MISSING_IMPORTS"]]

missing_cases_state_2017 = dict(zip(missing_cases_state_2017["State"],
                                    missing_cases_state_2017["MISSING_IMPORTS"]))

###################################################################################################

# This section creates two dictionaries, and both have state names (strings) as keys

# The first dictionary contains the indices in the variables2016 (variables2017) dataset
#   corresponding to counties with no reported cases in the per-county data
#   (this means these counties had less than 5 cases)
# The second dictionary contains the number of counties in a particular state
#   with no reported cases in the per-county data

# Recall that variables2016 and variables2017 have been sorted by GEOID so the ordering is preserved
#   -- the ordering matters for our indexing

# dict_ix_zero_reports_2016 dictionary
#   Key is state, value is list of ix for that state where REPORTED_IMPORTS == 0
# dict_num_counties_zero_reports_2016 dictionary
#   Key is state, value is number of counties in that state where REPORTED_IMPORTS == 0
dict_ix_zero_reports_2016 = {}
dict_num_counties_zero_reports_2016 = {}

states = np.asarray(variables2016["State"].unique(), dtype=str)
for state in states:
    dict_ix_zero_reports_2016[state] = \
        variables2016.index[(variables2016["REPORTED_IMPORTS"] == 0) & (variables2016["State"] == state)].tolist()
    dict_num_counties_zero_reports_2016[state] = len(dict_ix_zero_reports_2016[state])

# This is the same as the 2016 case but for 2017 data

dict_ix_zero_reports_2017 = {}
dict_num_counties_zero_reports_2017 = {}

states = np.asarray(variables2017["State"].unique(), dtype=str)

for state in states:
    dict_ix_zero_reports_2017[state] = \
        variables2017.index[(variables2017["REPORTED_IMPORTS"] == 0) & (variables2017["State"] == state)].tolist()
    dict_num_counties_zero_reports_2017[state] = len(dict_ix_zero_reports_2017[state])

###################################################################################################

# This section artificially adds missing cases to counties with no reported
#   cases in the per-county data, using Monte Carlo simulation

# Specify number of replications per processor and initialize RNG
num_reps = 1
rng = np.random.default_rng(rank)

artificial_datasets_2016 = []
states = np.asarray(variables2016["State"].unique(), dtype=str)

# Artificially add missing cases to counties with no reported cases
#   such that the sum of added missing cases equals the total number of
#   missing cases in each state
# Make sure samples are "valid," meaning that we do not add 5 or more
#   cases per county
for i in range(num_reps):
    new_num_imports = copy.deepcopy(np.array(variables2016["REPORTED_IMPORTS"]))

    for state in states:
        valid_sample = False

        while not valid_sample:
            random_sample_county_occurrences = \
                rng.choice((dict_ix_zero_reports_2016[state]), int(missing_cases_state_2016[state]))

            added_counts_per_county = Counter(random_sample_county_occurrences)

            for key in added_counts_per_county.keys():
                if added_counts_per_county[key] >= 5:
                    break

            for county_ix in random_sample_county_occurrences:
                new_num_imports[county_ix] += 1

            valid_sample = True

    # assert np.sum(new_num_imports) == 2072
    artificial_datasets_2016.append(new_num_imports)

# This is the same as the 2016 case but for 2017 data

artificial_datasets_2017 = []
states = np.asarray(variables2017["State"].unique(), dtype=str)

for i in range(num_reps):
    new_num_imports = copy.deepcopy(np.array(variables2017["REPORTED_IMPORTS"]))

    for state in states:
        valid_sample = False
        while not valid_sample:
            random_sample_county_occurrences = \
                rng.choice((dict_ix_zero_reports_2017[state]), int(missing_cases_state_2017[state]))

            added_counts_per_county = Counter(random_sample_county_occurrences)

            for key in added_counts_per_county.keys():
                if added_counts_per_county[key] >= 5:
                    break

            for county_ix in random_sample_county_occurrences:
                new_num_imports[county_ix] += 1

            valid_sample = True
    # 1 missing case? I thought it should be 2152
    # assert np.sum(new_num_imports) == 2151
    artificial_datasets_2017.append(new_num_imports)

###################################################################################################

# This section runs 100 random forests (for each parallel processor)
#   on 100 samples of random fill-ins for missing cases for 2016
#   and 100 random forests on 100 samples for 2017

# The in-sample occurrence proportions are averaged across the 200
#   total random forests (100 for 2016 and 100 for 2017).

# The final occurrence proportions are used as estimates for
#   "next import probabilities"

y_pred_final_scaled_output = []

variables2016.drop(columns="State", inplace=True)
variables2017.drop(columns="State", inplace=True)

j = 0

for i in range(num_reps):

    # 2016 data random forest
    rfr2016 = RandomForestRegressor(n_estimators=1000, max_depth=5)
    rfr2016.fit(variables2016.loc[:, variables2016.columns != "REPORTED_IMPORTS"], artificial_datasets_2016[j])

    y_test2016 = variables2016.loc[:, variables2016.columns != "REPORTED_IMPORTS"]
    y_pred2016 = rfr2016.predict(y_test2016)
    y_pred2016_scaled = y_pred2016 / np.sum(y_pred2016)

    # 2017 data random forest
    rfr2017 = RandomForestRegressor(n_estimators=1000, max_depth=5)
    rfr2017.fit(variables2017.loc[:, variables2017.columns != "REPORTED_IMPORTS"], artificial_datasets_2017[j])

    y_test2017 = variables2017.loc[:, variables2017.columns != "REPORTED_IMPORTS"]
    y_pred2017 = rfr2017.predict(y_test2017)
    y_pred2017_scaled = y_pred2017 / np.sum(y_pred2017)

    # Average 2016 and 2017 predictions
    # Note that predictions are scaled (turned into proportions)
    y_pred_final_scaled = (y_pred2016_scaled + y_pred2017_scaled)/2
    y_pred_final_scaled_output.append(y_pred_final_scaled)

    j += 1

# Save final results
df_final = pd.DataFrame({"FIPS": variables2016["GEOID"], "Next Import Probability": np.average(y_pred_final_scaled_output, axis=0)})
df_final.to_csv(str(rank) + "_malaria_probability_next_import_comes_from_county_added_missing_state_counts.csv")

# Proportions of actual reported imports for diagnostic purposes
# y_actual2016 = variables2016["REPORTED_IMPORTS"]
# y_actual_scaled2016 = np.array(y_actual2016) / np.sum(np.array(y_actual2016))
# y_actual2017 = variables2017["REPORTED_IMPORTS"]
# y_actual_scaled2017 = np.array(y_actual2017) / np.sum(np.array(y_actual2017))
