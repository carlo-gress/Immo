### Recode and curate
import pandas as pd
import numpy as np
import os

# Read the data
os.chdir("/Users/aleph/Desktop/MDS/semestres/2/machine learning/data/")
data = pd.read_csv("raw_data.csv", sep = ";")

## Recode all missing data to NaN
missing = [-11, -10, -9, -8, -7, -6, -5]
data = data.replace(missing, np.nan)

## Impute data for variables with low NaN count

# What is the NaN count for every predictor?
nans = data.isna().sum()

# 'zimmeranzahl' has too few to drop, impute the mode
data["zimmeranzahl"].isna().sum()
zimmeranzahl_mode = data["zimmeranzahl"].mode()[0]
data["zimmeranzahl"].fillna(zimmeranzahl_mode, inplace = True)

# 'nebenkosten' has too few to drop, impute the mean
data["nebenkosten"].isna().sum()
nebenkosten_mean = data["nebenkosten"].mean()
data["nebenkosten"].fillna(nebenkosten_mean, inplace = True)

## Keep only columns with zero NaNs

# Now we subset our data to keep only variables without NaNs
filter = data.isna().sum() == 0
data = data.loc[:, filter]

## Now we do some feature engineering

# Remove all click variables, since they are metainformation
data = data.drop(['click_schnellkontakte', 'click_weitersagen', 'click_url'], axis = 1)

# Drop metainformation and geoinformation for now
data = data.drop(['gid2019', 'kid2019', 'immobilientyp', 'lieferung'], axis = 1)

# What variables are we left with?
pd.Series(list(data))

# Write the data to a CSV
data.to_csv("curated_data.csv", index = False)


