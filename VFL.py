import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from half import Half
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


df = pd.read_csv(
    "data/bankmarketing/bank-additional-full.csv",
    sep=";"
)

df = df.drop(columns=["duration"])



df["y"] = df["y"].map({
    "no": 0,
    "yes": 1
})

X = df.drop(columns=["y"])
y = df["y"]

# print(X.shape)
# print(X.head())

# print(y.shape)
# print(y.head())

def target_mean_encode(df, column, target):
    means = df.groupby(column)[target].mean()
    return df[column].map(means)

categorical_features = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

for col in categorical_features:
    X[col] = target_mean_encode(
        df,
        col,
        "y"
    )

# print(X.head())

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

active_features = X_scaled.columns[:14]
passive_features = X_scaled.columns[14:]

Y = X_scaled[active_features].values
X_passive = X_scaled[passive_features].values

# print(Y.shape)
# print(X_passive.shape)

Y_train_pred, Y_prediction, \
X_train_pred, X_prediction, \
y_train_pred, y_prediction = train_test_split(
    Y,
    X_passive,
    y.values,
    test_size=0.20,
    random_state=42,
    stratify=y
)

Y_train, Y_test, \
X_train, X_test, \
y_train, y_test = train_test_split(
    Y_train_pred,
    X_train_pred,
    y_train_pred,
    test_size=0.20,
    random_state=42,
    stratify=y_train_pred
)

VFL = LogisticRegression(
    C=np.inf,
    max_iter=1000
)

VFL.fit(
    np.hstack([Y_train, X_train]),
    y_train
)

c = VFL.predict_proba(
    np.hstack([Y_test, X_test])
)

AM = LogisticRegression(
        C=np.inf,
        max_iter=1000
    )

AM.fit(
    Y_train,
    y_train
)

c_hat = AM.predict_proba(Y_test)

# for i in range (5):
#     c_hat_p0 = c_hat[i][0]
#     c_hat_p1 = c_hat[i][1]
#     print(f"c_hat_P(0) = {c_hat_p0}, c_hat_P(1) = {c_hat_p1}")

#     c_p0 = c[i][0]
#     c_p1 = c[i][1]
#     print(f"c_P(0) = {c_p0}, c_P(1) = {c_p1}")

vfl_pred = VFL.predict(
    np.hstack([Y_test, X_test])
)

am_pred = AM.predict(Y_test)

print(
    "VFL accuracy:",
    accuracy_score(y_test, vfl_pred)
)

print(
    "AM accuracy:",
    accuracy_score(y_test, am_pred)
)

score_mse = mean_squared_error(
    c[:, 1],
    c_hat[:, 1]
)

print("Confidence-score MSE:", score_mse)