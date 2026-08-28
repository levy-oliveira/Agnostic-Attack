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
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome"
]

numerical_features = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

# ============================================================
# TARGET MEAN ENCODING
# ============================================================

def target_mean_encode(df, column, target):

    means = df.groupby(column)[target].mean()

    return df[column].map(means)


for col in categorical_features:

    X[col] = target_mean_encode(
        df,
        col,
        "y"
    )


# ============================================================
# NORMALIZATION
# ============================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# ============================================================
# ACTIVE / PASSIVE PARTITION
# ============================================================

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
# ============================================================
# VFL
# ============================================================

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


# ============================================================
# ADVERSARY MODEL
# ============================================================

AM = LogisticRegression(
    C=np.inf,
    max_iter=1000
)

AM.fit(
    Y_train,
    y_train
)

c_hat = AM.predict_proba(Y_test)


# ============================================================
# PERFORMANCE
# ============================================================

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

print(
    "Confidence-score MSE:",
    score_mse
)


# ============================================================
# EXTRACT VFL PARAMETERS
# ============================================================

W = VFL.coef_[0]

W_act = W[:14]
W_pas = W[14:]

b = VFL.intercept_[0]


# ============================================================
# c_hat'
# ============================================================

eps = 1e-12

c_hat_clipped = np.clip(
    c_hat,
    eps,
    1 - eps
)

c_hat_prime = np.log(
    c_hat_clipped[:, 1] /
    c_hat_clipped[:, 0]
)


# ============================================================
# HALF*
# ============================================================

X_hat = Half(
    W_pas,
    W_act,
    Y_test,
    c_hat_prime,
    b
)


# ============================================================
# ATTACK EVALUATION
# ============================================================

attack_mse = np.mean(
    (X_test - X_hat) ** 2
)

print(
    "Half* reconstruction MSE:",
    attack_mse
)


mse_per_feature = np.mean(
    (X_test - X_hat) ** 2,
    axis=0
)

print("\nMSE per passive feature:")

for feature, mse in zip(
    passive_features,
    mse_per_feature
):
    print(
        f"{feature}: {mse}"
    )


# ==============================================
# TESTE
# ==============================================
c_prime = np.log(
    c[:, 1] / c[:, 0]
)

X_hat_real_score = Half(
    W_pas,
    W_act,
    Y_test,
    c_prime,
    b
)

mse_real_score = np.mean(
    (X_test - X_hat_real_score) ** 2
)

print(
    "Half* MSE using REAL c:",
    mse_real_score
)


print("W_act.shape: ", W_act.shape)
print("W_pas.shape: ", W_pas.shape)
print("X_test.shape: ", X_test.shape)
print("X_hat.shape: ", X_hat.shape)


# ============================================================
# VALIDATION USING DIRECT VFL LOGIT
# ============================================================

z_real = VFL.decision_function(
    np.hstack([Y_test, X_test])
)

X_hat_z = Half(
    W_pas,
    W_act,
    Y_test,
    z_real,
    b
)

mse_z = np.mean(
    (X_test - X_hat_z) ** 2
)

print(
    "Half* MSE using real VFL logit:",
    mse_z
)

z_manual = (
    Y_test @ W_act
    +
    X_test @ W_pas
    +
    b
)

print(
    "Logit reconstruction error:",
    np.max(
        np.abs(z_real - z_manual)
    )
)

print("X.dtypes: ", X.dtypes)
print("X.shape: ", X.shape)
print("X_scaled.min(): ", X_scaled.min())
print("X_scaled.max(): ",X_scaled.max())