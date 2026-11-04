
from pathlib import Path
import json
import optuna
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "data" / "processed"
FIGURAS = BASE / "reports" / "figures"
FIGURAS.mkdir(parents=True, exist_ok=True)

COSTO_FP = 50
COSTO_FN = 500


def evaluar(nombre, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    return {
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1": f1_score(y_test, pred, zero_division=0),
        "FP": int(fp),
        "FN": int(fn),
        "Costo": int(fp * COSTO_FP + fn * COSTO_FN)
    }


def main():
    with open(
        DATOS / "p2_optuna_resultados.json",
        encoding="utf-8"
    ) as archivo:
        resultados = json.load(archivo)

    parametros = resultados["mejores_hiperparametros"]
    trials = pd.read_csv(DATOS / "p2_optuna_trials.csv")

    # Grafica 1: historial de optimizacion
    valores = trials["value"]
    mejores = valores.cummax()

    plt.figure(figsize=(9, 5))
    plt.plot(
        range(1, len(valores) + 1),
        valores,
        marker="o",
        label="F1 de cada trial"
    )
    plt.plot(
        range(1, len(valores) + 1),
        mejores,
        label="Mejor F1 acumulado"
    )
    plt.xlabel("Numero de trial")
    plt.ylabel("F1 de validacion cruzada")
    plt.title("Optuna - Historial de optimizacion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURAS / "optuna_historial.png", dpi=300
    )
    plt.close()

    
# Grafica 2: importancia oficial de Optuna
    estudio = optuna.load_study(
        study_name="p2_random_forest",
        storage=f"sqlite:///{(DATOS / 'p2_optuna.db').as_posix()}"
    )

    importancias = optuna.importance.get_param_importances(
        estudio
    )

    serie = pd.Series(importancias).sort_values()

    plt.figure(figsize=(9, 5))
    serie.plot(kind="barh")

    plt.xlabel("Importancia relativa")
    plt.title("Optuna - Importancia de hiperparametros")
    plt.tight_layout()

    plt.savefig(
        FIGURAS / "optuna_parametros.png",
        dpi=300
    )
    plt.close()

    # Guardar configuracion final
    with open(
        BASE / "config.json",
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            {"random_forest": parametros},
            archivo,
            indent=4
        )

    # Comparacion en el conjunto de prueba
    df = pd.read_parquet(DATOS / "dataset.parquet")

    train = df[df["conjunto"] == "train"]
    test = df[df["conjunto"] == "test"]

    sensores = [f"sensor_{i}" for i in range(1, 25)]

    X_train = train[sensores]
    X_test = test[sensores]

    clase = train["clase"].value_counts().idxmin()
    y_train = (train["clase"] == clase).astype(int)
    y_test = (test["clase"] == clase).astype(int)

    original = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    optimizado = RandomForestClassifier(
        **parametros,
        random_state=42,
        n_jobs=-1
    )

    comparacion = pd.DataFrame([
        evaluar(
            "Random Forest original",
            original, X_train, y_train, X_test, y_test
        ),
        evaluar(
            "Random Forest Optuna",
            optimizado, X_train, y_train, X_test, y_test
        )
    ])

    comparacion.to_csv(
        DATOS / "p2_comparacion_optuna.csv",
        index=False
    )

    print("\n=== COMPARACION FINAL ===")
    print(comparacion.to_string(index=False))

    print("\n=== ARCHIVOS GENERADOS ===")
    print("config.json")
    print("reports/figures/optuna_historial.png")
    print("reports/figures/optuna_parametros.png")
    print("data/processed/p2_comparacion_optuna.csv")
    print("\nSesion 5 completada.")


if __name__ == "__main__":
    main()
