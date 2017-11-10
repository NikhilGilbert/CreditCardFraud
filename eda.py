from pandas import set_option
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
import pandas as pd
import numpy

file = "HR_comma_sep.csv"
data = pd.read_csv(file)


def make_numeric(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != numpy.int64 and df[column].dtype != numpy.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))


make_numeric(data)

# We removed any data that has null values first
data.dropna()
# Check explicitly for each headings value type
types = data.dtypes

# Overall description of data
set_option('display.width', 400)
set_option('precision', 4)
description = data.describe()

# Checks an important classification, whether or not a person has left the company
class_counts = data.groupby('left').size()
# Check to see that all the values are even
check_size = data.count()
# Checks for correlation between variables
data_correlation = data.corr(method='pearson')
# Data is assumed to be normally distributed
skew = data.skew()
# Density plots  for visual distributions
data.plot(kind='density', subplots=True, layout=(3, 4), sharex=False)

array = data.values

X = numpy.append(array[:, 0:6], array[:, 7:10], axis=1)
Y = array[:, 6]

scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

print((X[0]))
print(Y)

# print(description)
# print('\n')
# print(class_counts)
# print('\n')
# print(data_correlation)
# print('\n')
# # use log transformations for very skewed data
# print(skew)
# print('\n')
