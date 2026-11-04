
from pathlib import Path
import json

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# PRACTICA 2 - SESION 4
BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "processed" / "dataset.parquet"
SALIDA = BASE / "data" / "processed"
N_TRIALS = 20


def main():
    df = pd.read_parquet(DATASET)
    train = df[df["conjunto"] == "train"].copy()

    sensores = [f"sensor_{i}" for i in range(1, 25)]
    X = train[sensores]

    clase_minoritaria = train["clase"].value_counts().idxmin()
    y = (train["clase"] == clase_minoritaria).astype(int)

    validacion = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    def objetivo(trial):
        parametros = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 200, step=25
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 5, 30
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            )
        }

        modelo = RandomForestClassifier(
            **parametros,
            random_state=42,
            n_jobs=-1
        )

        puntuaciones = cross_val_score(
            modelo, X, y,
            cv=validacion,
            scoring="f1",
            n_jobs=1
        )

        return float(puntuaciones.mean())

    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    estudio.optimize(objetivo, n_trials=N_TRIALS)

    resultados = {
        "numero_trials": len(estudio.trials),
        "mejor_f1_validacion": estudio.best_value,
        "mejores_hiperparametros": estudio.best_params
    }

    SALIDA.mkdir(parents=True, exist_ok=True)

    with open(
        SALIDA / "p2_optuna_resultados.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(resultados, archivo, indent=4)

    estudio.trials_dataframe().to_csv(
        SALIDA / "p2_optuna_trials.csv",
        index=False
    )

    print("\n=== RESULTADOS DE OPTUNA ===")
    print("Pruebas:", len(estudio.trials))
    print(f"Mejor F1: {estudio.best_value:.4f}")
    print("Mejores parametros:", estudio.best_params)
    print("Sesion 4 completada.")


if __name__ == "__main__":
    main()
