
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET = BASE_DIR / "data" / "processed" / "dataset.parquet"
FIGURES = BASE_DIR / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COSTO_FP = 50
COSTO_FN = 500


def calcular_costo(y_real, y_pred):
    """Calcula los errores y el costo económico simulado."""
    tn, fp, fn, tp = confusion_matrix(
        y_real, y_pred, labels=[0, 1]
    ).ravel()

    return int(fp), int(fn), int(fp * COSTO_FP + fn * COSTO_FN)


def main():
    """Busca el umbral con menor costo en el conjunto evaluado."""
    df = pd.read_parquet(DATASET)

    train = df[df["conjunto"] == "train"]
    test = df[df["conjunto"] == "test"]

    sensores = [f"sensor_{i}" for i in range(1, 25)]

    clase_minoritaria = train["clase"].value_counts().idxmin()

    X_train = train[sensores]
    X_test = test[sensores]

    y_train = (train["clase"] == clase_minoritaria).astype(int)
    y_test = (test["clase"] == clase_minoritaria).astype(int)

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(X_train, y_train)

    probabilidades = modelo.predict_proba(X_test)[:, 1]

    resultados = []

    for umbral in np.round(np.arange(0.10, 0.901, 0.05), 2):
        predicciones = (probabilidades >= umbral).astype(int)

        fp, fn, costo = calcular_costo(y_test, predicciones)

        resultados.append({
            "Umbral": umbral,
            "FP": fp,
            "FN": fn,
            "Costo ($)": costo
        })

    tabla = pd.DataFrame(resultados)

    mejor = tabla.loc[tabla["Costo ($)"].idxmin()]
    base = tabla[np.isclose(tabla["Umbral"], 0.50)].iloc[0]

    ahorro = int(base["Costo ($)"] - mejor["Costo ($)"])

    print("\n=== COMPARACION DE UMBRALES ===")
    print(tabla.to_string(index=False))

    print("\n=== RESULTADO FINAL ===")
    print(f"Umbral optimo: {mejor['Umbral']:.2f}")
    print(f"Costo con umbral 0.50: ${base['Costo ($)']:,.0f}")
    print(f"Costo minimo: ${mejor['Costo ($)']:,.0f}")
    print(f"Ahorro simulado: ${ahorro:,}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        tabla["Umbral"],
        tabla["Costo ($)"],
        marker="o",
        label="Costo total"
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Umbral tradicional"
    )

    plt.axvline(
        mejor["Umbral"],
        linestyle=":",
        label="Umbral de menor costo"
    )

    plt.title("Threshold Tuning - Costo vs Umbral")
    plt.xlabel("Umbral de decisión")
    plt.ylabel("Costo simulado ($)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    grafico = FIGURES / "threshold_tuning.png"
    plt.savefig(grafico, dpi=300)
    plt.show()

    print("\nGráfico guardado en:", grafico)


if __name__ == "__main__":
    main()
