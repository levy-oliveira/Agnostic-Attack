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

    logit_mse = np.mean((z_real - z_hat) ** 2)
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
        "feature_mse": feature_mse,
        "W_pas": W_pas.copy(),
        "W_act": W_act.copy(),
        "z_hat": z_hat.copy(),
        "z_real": z_real.copy(),
        "logit_mse": logit_mse,
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

    result["passive_features"] = [
        X_scaled.columns[i]
        for i in passive_indices
    ]

    result["active_features"] = [
        X_scaled.columns[i]
        for i in active_indices
    ]

    results.append(result)

    passive_names = [
        X_scaled.columns[i]
        for i in passive_indices
    ]

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

# ============================================================
# FEATURE-LEVEL RESULTS
# ============================================================

feature_rows = []

for result in results:

    passive_features = result["passive_features"]
    feature_mse = result["feature_mse"]
    W_pas = result["W_pas"]
    logit_mse = result["logit_mse"]

    

    for feature, mse, weight in zip(
        passive_features,
        feature_mse,
        W_pas
    ):
        predicted_mse = logit_mse / (weight ** 2)

        feature_rows.append({
            "scenario": result["scenario"],
            "feature": feature,
            "weight": weight,
            "abs_weight": abs(weight),
            "feature_mse": mse,
            "logit_mse": logit_mse,
            "predicted_mse": predicted_mse
        })

##print("\n")
#print("==============================")
#print("FEATURE-LEVEL RESULTS")
#print("==============================")

feature_df = pd.DataFrame(feature_rows)
#print(
#     feature_df.to_string(index=False)
# )



#print("\n")
#print("==============================")
#print("FEATURE SUMMARY")
#print("==============================")

feature_summary = (
    feature_df
    .groupby("feature")
    .agg(
        mean_mse=("feature_mse", "mean"),
        std_mse=("feature_mse", "std"),
        mean_abs_weight=("abs_weight", "mean"),
        std_abs_weight=("abs_weight", "std")
    )
    .sort_values("mean_mse", ascending=False)
)

#print(
#     feature_summary.to_string()
# )

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

confidence_half_corr = results_df[
    "confidence_mse"
].corr(
    results_df["half_star_mse"]
)

#print("\n")
#print("==============================")
#print("CONFIDENCE MSE x HALF* MSE")
#print("==============================")

#print(
#     "Correlation:",
#     confidence_half_corr
# )


feature_correlation = X_scaled.corr()

#print("\n")
#print("==============================")
#print("FEATURE CORRELATION")
#print("==============================")

#print(
    # feature_correlation.to_string()
# )


# ============================================================
# ACTIVE-PASSIVE CORRELATION ANALYSIS
# ============================================================

correlation_rows = []

corr_matrix = X_scaled.corr()

for result in results:

    active_features = result["active_features"]
    passive_features = result["passive_features"]

    feature_mse = result["feature_mse"]

    for feature, mse in zip(
        passive_features,
        feature_mse
    ):

        correlations = corr_matrix.loc[
            feature,
            active_features
        ].abs()

        max_corr = correlations.max()
        mean_corr = correlations.mean()

        correlation_rows.append({
            "scenario": result["scenario"],
            "feature": feature,
            "max_active_corr": max_corr,
            "mean_active_corr": mean_corr,
            "feature_mse": mse
        })

correlation_df = pd.DataFrame(
    correlation_rows
)

# print("\n")
# print("==============================")
# print("ACTIVE-PASSIVE CORRELATION")
# print("==============================")

# print(
#     correlation_df.to_string(index=False)
# )

# print("\n")
# print("==============================")
# print("CORRELATION SUMMARY")
# print("==============================")

correlation_summary = (
    correlation_df
    .groupby("feature")
    .agg(
        mean_mse=("feature_mse", "mean"),
        mean_max_corr=("max_active_corr", "mean"),
        std_max_corr=("max_active_corr", "std"),
        mean_active_corr=("mean_active_corr", "mean")
    )
    .sort_values(
        "mean_mse",
        ascending=False
    )
)

# print(
#     correlation_summary.to_string()
# )

corr_mse = correlation_df[
    ["max_active_corr", "feature_mse"]
].corr()

# print("\n")
# print("==============================")
# print("ACTIVE CORRELATION x MSE")
# print("==============================")

# print(corr_mse)

# print(
#     "\nCorrelation max_active_corr x feature_mse:",
#     correlation_df["max_active_corr"].corr(
#         correlation_df["feature_mse"]
#     )
# )


# print("\n==============================")
# print("WEIGHT SENSITIVITY x MSE")
# print("==============================")

feature_df["inv_abs_weight"] = 1 / feature_df["abs_weight"]
feature_df["inv_weight_sq"] = 1 / (feature_df["abs_weight"] ** 2)


# print(
#     feature_df[
#         ["inv_abs_weight", "feature_mse"]
#     ].corr()
# )

# print(
#     feature_df[
#         ["logit_mse", "predicted_mse", "feature_mse"]
#     ].corr()
# )
# print(
#     "Correlation 1/|W| x MSE:",
#     feature_df["inv_abs_weight"].corr(
#         feature_df["feature_mse"]
#     )
# )

# print(
#     "Correlation 1/W² x MSE:",
#     feature_df["inv_weight_sq"].corr(
#         feature_df["feature_mse"]
#     )
# )

# print(
#     "Correlation predicted_mse x feature_mse:",
#     feature_df["predicted_mse"].corr(
#         feature_df["feature_mse"]
#     )
# )

feature_variance = X_scaled.var()

feature_df["feature_variance"] = (
    feature_df["feature"]
    .map(feature_variance)
)

feature_df["relative_mse"] = (
    feature_df["feature_mse"] /
    feature_df["feature_variance"]
)

# print(
#     feature_df[
#         [
#             "feature",
#             "feature_mse",
#             "feature_variance",
#             "relative_mse"
#         ]
#     ].sort_values(
#         "relative_mse",
#         ascending=False
#     )
# )

# print(
#     "Correlation variance x MSE:",
#     feature_df["feature_variance"].corr(
#         feature_df["feature_mse"]
#     )
# )

# print(
#     "Correlation variance x relative MSE:",
#     feature_df["feature_variance"].corr(
#         feature_df["relative_mse"]
#     )
# )

feature_df["std"] = np.sqrt(
    feature_df["feature_variance"]
)

feature_df["nrmse"] = (
    np.sqrt(feature_df["feature_mse"]) /
    feature_df["std"]
)

print(
    "Correlation std x feature_mse:",
    feature_df["std"].corr(
        feature_df["feature_mse"]
    )
)

print(
    "Correlation std x NRMSE:",
    feature_df["std"].corr(
        feature_df["nrmse"]
    )
)