import csv
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

TIME_WEIGHT = 15
TOP = 3


def float_try_parse(x):
    try:
        return float(x)
    except ValueError:
        return x


def get_categories():
    data = {}
    categories = []
    with open('Towns-Categories.csv') as csvfile:
        reader = csv.reader(csvfile)
        weights = next(reader)[2:]
        for i in range(len(weights)):
            weights[i] = float_try_parse(weights[i])
        headers = next(reader)
        for category in headers:
            data[category] = []
            categories.append(category)

        for row in reader:
            for col in range(len(row)):
                data[categories[col]].append(float_try_parse(row[col]))

    df = pd.DataFrame(data)
    return df, weights


def get_times(time):
    data = {}
    categories = []
    with open('Towns-Times.csv') as csvfile:
        reader = csv.reader(csvfile)
        user_headers = next(reader)
        headers = next(reader)
        for category in headers:
            data[category] = []
            categories.append(category)

        for row in reader:
            for col in range(len(row)):
                data[categories[col]].append(float_try_parse(row[col]))

    df = pd.DataFrame({headers[0]: data[headers[0]], "Time": data[str(time)]})
    return df


def standardize_df(df, columns):
    for index in columns:
        col = df.iloc[:, index]
        spread = np.std(col)
        mean = np.mean(col)
        for row in range(len(col)):
            df[col.name][row] = (df[col.name][row] - mean)/spread

    return df


def weight_based_opt(df, weights, columns, top = 5):
    values = []
    for row in df.values:
        value = 0
        for col_index in range(len(columns)):
            value += row[columns[col_index]] * weights[col_index]
        values.append(value)
    top_rows = np.argpartition(values, -top)[-top:]

    return df.iloc[top_rows]


data, weights = get_categories()
time_category = get_times(15)
joined = data.merge(time_category, left_on="ID", right_on="ID")
weights.append(TIME_WEIGHT)

data_columns = range(2, joined.shape[1])
data = standardize_df(joined, data_columns)
top_rows = weight_based_opt(joined, weights, data_columns, top=TOP)

print(top_rows)


