import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from half import Half
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def generate_feature_scenarios(features, passive_size=5):

    n_features = len(features)

    scenarios = []

    for start in range(n_features):

        passive_indices = [
            (start + offset) % n_features
            for offset in range(passive_size)
        ]

        active_indices = [
            i for i in range(n_features)
            if i not in passive_indices
        ]

        scenarios.append({
            "passive_indices": passive_indices,
            "active_indices": active_indices
        })

    return scenarios

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

def run_attack_scenario(
    X_scaled,
    y,
    active_indices,
    passive_indices
):

    X_all = X_scaled.values
    y_all = y.values

    # --------------------------------------------------
    # Select active/passive features
    # --------------------------------------------------

    Y = X_all[:, active_indices]
    X_passive = X_all[:, passive_indices]

    # --------------------------------------------------
    # Dataset split
    # --------------------------------------------------

    (
        Y_train_pred,
        Y_prediction,
        X_train_pred,
        X_prediction,
        y_train_pred,
        y_prediction
    ) = train_test_split(
        Y,
        X_passive,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all
    )

    (
        Y_train,
        Y_test,
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        Y_train_pred,
        X_train_pred,
        y_train_pred,
        test_size=0.20,
        random_state=42,
        stratify=y_train_pred
    )

    # --------------------------------------------------
    # VFL
    # --------------------------------------------------

    vfl_model = LogisticRegression(
        max_iter=1000
    )

    vfl_model.fit(
        np.hstack([
            Y_train,
            X_train
        ]),
        y_train
    )

    vfl_accuracy = vfl_model.score(
        np.hstack([
            Y_test,
            X_test
        ]),
        y_test
    )

    # --------------------------------------------------
    # AM
    # --------------------------------------------------

    am_model = LogisticRegression(
        max_iter=1000
    )

    am_model.fit(
        Y_train,
        y_train
    )

    am_accuracy = am_model.score(
        Y_test,
        y_test
    )

    # --------------------------------------------------
    # Confidence scores
    # --------------------------------------------------

    c_real = vfl_model.predict_proba(
        np.hstack([
            Y_test,
            X_test
        ])
    )[:, 1]

    c_hat = am_model.predict_proba(
        Y_test
    )[:, 1]

    confidence_mse = np.mean(
        (c_real - c_hat) ** 2
    )

    # --------------------------------------------------
    # Convert probability -> logit
    # --------------------------------------------------

    c_real = np.clip(
        c_real,
        1e-15,
        1 - 1e-15
    )

    c_hat = np.clip(
        c_hat,
        1e-15,
        1 - 1e-15
    )

    z_real = np.log(
        c_real / (1 - c_real)
    )

    z_hat = np.log(
        c_hat / (1 - c_hat)
    )

    # --------------------------------------------------
    # VFL weights
    # --------------------------------------------------

    weights = vfl_model.coef_[0]
    bias = vfl_model.intercept_[0]

    W_act = weights[:len(active_indices)]
    W_pas = weights[len(active_indices):]

    # --------------------------------------------------
    # Half*
    # --------------------------------------------------

    X_hat = Half(
        W_pas,
        W_act,
        Y_test,
        z_hat,
        bias
    )

    reconstruction_mse = np.mean(
        (X_test - X_hat) ** 2
    )

    feature_mse = np.mean(
        (X_test - X_hat) ** 2,
        axis=0
    )

    # --------------------------------------------------
    # Oracle: real confidence score
    # --------------------------------------------------

    X_hat_real = Half(
        W_pas,
        W_act,
        Y_test,
        z_real,
        bias
    )

    real_c_mse = np.mean(
        (X_test - X_hat_real) ** 2
    )

    return {
        "vfl_accuracy": vfl_accuracy,
        "am_accuracy": am_accuracy,
        "confidence_mse": confidence_mse,
        "reconstruction_mse": reconstruction_mse,
        "real_c_mse": real_c_mse,
        "feature_mse": feature_mse
    }


scenarios = generate_feature_scenarios(
    list(X_scaled.columns),
    passive_size=5
)

results = []

for scenario_id, scenario in enumerate(
    scenarios,
    start=1
):

    active_indices = scenario["active_indices"]
    passive_indices = scenario["passive_indices"]

    result = run_attack_scenario(
        X_scaled,
        y,
        active_indices,
        passive_indices
    )

    result["scenario"] = scenario_id
    result["active_indices"] = active_indices
    result["passive_indices"] = passive_indices

    results.append(result)

    passive_names = [
        X_scaled.columns[i]
        for i in passive_indices
    ]

    # print(
    #     f"\nScenario {scenario_id}/19"
    # )

    # print(
    #     "Passive:",
    #     passive_names
    # )

    # print(
    #     "VFL accuracy:",
    #     result["vfl_accuracy"]
    # )

    # print(
    #     "AM accuracy:",
    #     result["am_accuracy"]
    # )

    # print(
    #     "Confidence MSE:",
    #     result["confidence_mse"]
    # )

    # print(
    #     "Half* MSE:",
    #     result["reconstruction_mse"]
    # )

    # print(
    #     "Half* MSE using REAL c:",
    #     result["real_c_mse"]
    # )

mse_values = [
    result["reconstruction_mse"]
    for result in results
]

real_c_mse_values = [
    result["real_c_mse"]
    for result in results
]

mean_mse = np.mean(
    mse_values
)

mean_real_c_mse = np.mean(
    real_c_mse_values
)

# print("\n==============================")
# print("FINAL RESULTS")
# print("==============================")

# print(
#     "Mean Half* reconstruction MSE:",
#     mean_mse
# )

# print(
#     "Mean Half* MSE using REAL c:",
#     mean_real_c_mse
# )

rows = []

for result in results:

    passive_names = [
        X_scaled.columns[i]
        for i in result["passive_indices"]
    ]

    rows.append({
        "scenario": result["scenario"],
        "passive_features": ", ".join(passive_names),
        "vfl_accuracy": result["vfl_accuracy"],
        "am_accuracy": result["am_accuracy"],
        "confidence_mse": result["confidence_mse"],
        "half_star_mse": result["reconstruction_mse"],
        "half_star_real_c_mse": result["real_c_mse"]
    })

results_df = pd.DataFrame(rows)

print("\n")
print(results_df.to_string(index=False))

results_df["attack_gap"] = (
    results_df["half_star_mse"]
    -
    results_df["half_star_real_c_mse"]
)

results_df["attack_gap_percent"] = (
    results_df["attack_gap"]
    /
    results_df["half_star_real_c_mse"]
    * 100
)