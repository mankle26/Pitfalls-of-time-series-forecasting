
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import itertools


def mape(y_true, y_pred):
    return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2)


def window_input_output(input_length: int, output_length: int, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['close'].shift(-i)
        i = i + 1

    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['close'].shift(-output_length - j)
        j = j + 1

    df = df.dropna(axis=0)

    return df


def n_inp_combinations():
    list_of_n_and_inp = [range(100, 2100, 200), range(10, 201, 1)]
    all_combinations = list(itertools.product(*list_of_n_and_inp))
    return all_combinations


data = pd.read_csv("djia.csv")
data.index = pd.to_datetime(data['date'], format='%Y-%m-%d')
del data['date']
data = data[27673:]

all_mape = []
all_n = []
all_inp = []
for n, inp in n_inp_combinations():
    data_n = data[n:]
    out = inp
    seq_df = window_input_output(inp, out, data_n)
    X_cols = [col for col in seq_df.columns if col.startswith('x')]
    X_cols.insert(0, 'close')
    y_cols = [col for col in seq_df.columns if col.startswith('y')]

    X_train = seq_df[X_cols][:-2].values
    y_train = seq_df[y_cols][:-2].values

    X_test = seq_df[X_cols][-2:].values
    y_test = seq_df[y_cols][-2:].values

    dt_seq = DecisionTreeRegressor(random_state=42)

    dt_seq.fit(X_train, y_train)

    dt_seq_preds = dt_seq.predict(X_test)
    mape_dt_seq = mape(dt_seq_preds.reshape(1, -1), y_test.reshape(1, -1))

    all_n.append(n)
    all_inp.append(inp)
    all_mape.append(mape_dt_seq)

df = pd.DataFrame({'n': all_n, 'inp': all_inp, 'mape': all_mape})
df = df.pivot(index="inp", columns="n", values="mape")
df.plot.line()
df.to_csv("mapes.csv")
plt.show()



