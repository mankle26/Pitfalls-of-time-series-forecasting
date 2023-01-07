import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("djia.csv")
data.index = pd.to_datetime(data['date'], format='%Y-%m-%d')
del data['date']
data = data[36673:]

fig, ax = plt.subplots(figsize=(16, 11))

ax.plot(data['close'])
ax.set_xlabel('Date')
ax.set_ylabel('Close of DJIA')

fig.autofmt_xdate()
plt.tight_layout()


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


# set up model and run it
seq_df = window_input_output(26, 26, data)

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

# plot the prediction
fig, ax = plt.subplots(figsize=(16, 11))
ax.plot(np.arange(0, 26, 1), X_test[1], 'b-', label='input')
ax.plot(np.arange(26, 52, 1), y_test[1], marker='.', color='blue', label='Actual')
ax.plot(np.arange(26, 52, 1), dt_seq_preds[1], marker='^', color='green', label='Decision Tree')
ax.set_xlabel('Date')
ax.set_ylabel('DJIA')
plt.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()

# set up model and run it
seq_df = window_input_output(100, 100, data)

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

# plot the prediction
fig, ax = plt.subplots(figsize=(16, 11))
ax.plot(np.arange(0, 100, 1), X_test[1], 'b-', label='input')
ax.plot(np.arange(100, 200, 1), y_test[1], marker='.', color='blue', label='Actual')
ax.plot(np.arange(100, 200, 1), dt_seq_preds[1], marker='^', color='green', label='Decision Tree')
ax.set_xlabel('Date')
ax.set_ylabel('DJIA')
plt.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()