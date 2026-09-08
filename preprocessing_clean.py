import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from half import Half
from datetime import datetime
from pathlib import Path
from adversary_model import (
    train_adversary_model,
    predict_confidence,
    predict
)


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


def target_mean_encode(df, column, target):
    means = df.groupby(column)[target].mean()
    return df[column].map(means)


def run_attack_scenario(
    X_scaled,
    y,
    active_indices,
    passive_indices
):
    X_all = X_scaled.values
    y_all = y.values

    Y = X_all[:, active_indices]
    X_passive = X_all[:, passive_indices]

    # First split: hold out the final prediction set.
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

    # Second split: training/test data used by the VFL and AM.
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

    # VFL model.
    vfl_model = LogisticRegression(max_iter=1000)

    vfl_model.fit(
        np.hstack([Y_train, X_train]),
        y_train
    )

    vfl_accuracy = vfl_model.score(
        np.hstack([Y_test, X_test]),
        y_test
    )

    # Adversary Model (AM).
    am_model = LogisticRegression(max_iter=1000)

    am_model.fit(Y_train, y_train)

    am_accuracy = am_model.score(Y_test, y_test)

    # Confidence scores.
    c_real = vfl_model.predict_proba(
        np.hstack([Y_test, X_test])
    )[:, 1]

    c_hat = am_model.predict_proba(Y_test)[:, 1]

    confidence_mse = np.mean((c_real - c_hat) ** 2)

    # Convert probabilities to logits.
    c_real = np.clip(c_real, 1e-15, 1 - 1e-15)
    c_hat = np.clip(c_hat, 1e-15, 1 - 1e-15)

    z_real = np.log(c_real / (1 - c_real))
    z_hat = np.log(c_hat / (1 - c_hat))

    logit_mse = np.mean((z_real - z_hat) ** 2)

    # VFL weights.
    weights = vfl_model.coef_[0]
    bias = vfl_model.intercept_[0]

    W_act = weights[:len(active_indices)]
    W_pas = weights[len(active_indices):]

    # Half* reconstruction using the adversary-estimated logit.
    X_hat = Half(
        W_pas,
        W_act,
        Y_test,
        z_hat,
        bias
    )

    reconstruction_mse = np.mean((X_test - X_hat) ** 2)

    feature_mse = np.mean(
        (X_test - X_hat) ** 2,
        axis=0
    )

    # Oracle reconstruction using the real VFL logit.
    X_hat_real = Half(
        W_pas,
        W_act,
        Y_test,
        z_real,
        bias
    )

    real_c_mse = np.mean((X_test - X_hat_real) ** 2)

    return {
        "vfl_accuracy": vfl_accuracy,
        "am_accuracy": am_accuracy,
        "confidence_mse": confidence_mse,
        "reconstruction_mse": reconstruction_mse,
        "real_c_mse": real_c_mse,
        "feature_mse": feature_mse,
        "W_pas": W_pas.copy(),
        "W_act": W_act.copy(),
        "logit_mse": logit_mse
    }


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------

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

for col in categorical_features:
    X[col] = target_mean_encode(df, col, "y")

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)


# ---------------------------------------------------------------------
# Attack scenarios
# ---------------------------------------------------------------------

scenarios = generate_feature_scenarios(
    list(X_scaled.columns),
    passive_size=5
)

results = []

for scenario_id, scenario in enumerate(scenarios, start=1):
    result = run_attack_scenario(
        X_scaled,
        y,
        scenario["active_indices"],
        scenario["passive_indices"]
    )

    result["scenario"] = scenario_id
    result["active_indices"] = scenario["active_indices"]
    result["passive_indices"] = scenario["passive_indices"]

    result["passive_features"] = [
        X_scaled.columns[i]
        for i in scenario["passive_indices"]
    ]

    result["active_features"] = [
        X_scaled.columns[i]
        for i in scenario["active_indices"]
    ]

    results.append(result)


# ---------------------------------------------------------------------
# Results and CSV export
# ---------------------------------------------------------------------

rows = []

feature_variance = X_scaled.var()

for result in results:
    passive_features = result["passive_features"]

    for feature, mse, weight in zip(
        passive_features,
        result["feature_mse"],
        result["W_pas"]
    ):
        variance = feature_variance[feature]
        std = np.sqrt(variance)

        feature_nrmse = (
            np.sqrt(mse) / std
            if std > 0
            else np.nan
        )

        rows.append({
            "scenario": result["scenario"],
            "passive_features": ", ".join(passive_features),
            "feature": feature,
            "vfl_accuracy": result["vfl_accuracy"],
            "am_accuracy": result["am_accuracy"],
            "confidence_mse": result["confidence_mse"],
            "logit_mse": result["logit_mse"],
            "half_star_mse": result["reconstruction_mse"],
            "half_star_real_c_mse": result["real_c_mse"],
            "attack_gap": (
                result["reconstruction_mse"]
                - result["real_c_mse"]
            ),
            "attack_gap_percent": (
                (
                    result["reconstruction_mse"]
                    - result["real_c_mse"]
                )
                / result["real_c_mse"]
                * 100
                if result["real_c_mse"] != 0
                else np.nan
            ),
            "weight": weight,
            "abs_weight": abs(weight),
            "feature_mse": mse,
            "feature_variance": variance,
            "feature_std": std,
            "feature_nrmse": feature_nrmse
        })

results_df = pd.DataFrame(rows)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = results_dir / f"attack_results_{timestamp}.csv"

results_df = results_df.round(4)

results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"Resultados salvos em: {csv_path}")
