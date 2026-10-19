
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from data_loader import load_raw_data, preprocess


# ==========================================
# RUTAS DEL PROYECTO
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "sensor_readings_24.data"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ==========================================
# 1. METODO DEL CODO
# ==========================================

def find_optimal_k(X):
    """
    Calcula la inercia para diferentes valores
    de k y genera la grafica del metodo del codo.
    """

    valores_k = list(range(2, 11))
    inercias = []

    print("\nCalculando el metodo del codo...")

    for k in valores_k:

        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        modelo.fit(X)

        inercias.append(modelo.inertia_)

        print(
            f"k={k} | Inercia={modelo.inertia_:.4f}"
        )

    # Guardar resultados del metodo del codo
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tabla_codo = pd.DataFrame({
        "k": valores_k,
        "inercia": inercias
    })

    tabla_codo.to_csv(
        PROCESSED_DIR / "metodo_codo.csv",
        index=False
    )

    # Crear grafica
    plt.figure(figsize=(9, 5))

    plt.plot(
        valores_k,
        inercias,
        marker="o"
    )

    plt.title("Metodo del codo - K-Means")
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("Inercia")

    plt.xticks(valores_k)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Guardar grafica
    plt.savefig(
        PROCESSED_DIR / "metodo_codo.png",
        dpi=150
    )

    plt.show()

    return valores_k, inercias


# ==========================================
# 2. ENTRENAMIENTO DE K-MEANS
# ==========================================

def train_kmeans(X, k):
    """
    Agrupa las lecturas de los sensores
    utilizando K-Means.
    """

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = model.fit_predict(X)

    return model, etiquetas


# ==========================================
# 3. ENTRENAMIENTO DE DBSCAN
# ==========================================

def train_dbscan(X, eps=3.0, min_samples=10):
    """
    Agrupa las lecturas de los sensores
    utilizando DBSCAN.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    etiquetas = model.fit_predict(X)

    return model, etiquetas


# ==========================================
# 4. PROGRAMA PRINCIPAL
# ==========================================

if __name__ == "__main__":

    # Crear carpetas necesarias
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # CARGAR DATOS
    # --------------------------------------

    print("\nCargando dataset...")

    df = load_raw_data(DATA_PATH)

    X, y = preprocess(df)

    # Escalar los 24 sensores
    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)

    print("\nDatos preparados correctamente.")
    print("Registros:", X_scaled.shape[0])
    print("Sensores:", X_scaled.shape[1])

    # --------------------------------------
    # METODO DEL CODO
    # --------------------------------------

    valores_k, inercias = find_optimal_k(
        X_scaled
    )

    print("\nMetodo del codo finalizado.")
    print("Revisa la grafica para seleccionar k.")

    # Seleccionar el numero de clusters
    while True:

        try:
            k_optimo = int(
                input(
                    "\nIngresa el valor de k "
                    "(entre 2 y 10): "
                )
            )

            if k_optimo in valores_k:
                break

            print(
                "Selecciona un valor "
                "entre 2 y 10."
            )

        except ValueError:
            print(
                "Debes ingresar un numero entero."
            )

    print(
        f"\nValor seleccionado: k={k_optimo}"
    )

    # --------------------------------------
    # ENTRENAR K-MEANS
    # --------------------------------------

    print("\nEntrenando K-Means...")

    kmeans, etiquetas_kmeans = train_kmeans(
        X_scaled,
        k_optimo
    )

    print("K-Means entrenado correctamente.")

    # Mostrar registros por cluster
    print("\nRegistros por cluster:")

    grupos = pd.Series(
        etiquetas_kmeans
    ).value_counts().sort_index()

    print(grupos)

    # Evaluar K-Means
    print("\nEvaluando K-Means...")

    silhouette_kmeans = silhouette_score(
        X_scaled,
        etiquetas_kmeans,
        sample_size=min(2000, len(X_scaled)),
        random_state=42
    )

    print(
        "Silhouette Score K-Means:",
        round(silhouette_kmeans, 4)
    )

    # Guardar modelo K-Means
    joblib.dump(
        kmeans,
        MODELS_DIR / "kmeans.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "unsupervised_scaler.pkl"
    )

    print(
        "\nModelo guardado en models/kmeans.pkl"
    )

    # --------------------------------------
    # ENTRENAR DBSCAN
    # --------------------------------------

    print("\nEntrenando DBSCAN...")

    dbscan, etiquetas_dbscan = train_dbscan(
        X_scaled,
        eps=3.0,
        min_samples=10
    )

    print("DBSCAN entrenado correctamente.")

    # Mostrar registros por cluster
    print("\nRegistros por cluster:")

    grupos_dbscan = pd.Series(
        etiquetas_dbscan
    ).value_counts().sort_index()

    print(grupos_dbscan)

    # Contar clusters y ruido
    n_clusters = len(
        set(etiquetas_dbscan) - {-1}
    )

    n_ruido = np.sum(
        etiquetas_dbscan == -1
    )

    porcentaje_ruido = (
        n_ruido / len(etiquetas_dbscan)
    ) * 100

    print(
        "\nClusters encontrados:",
        n_clusters
    )

    print(
        "Registros considerados ruido:",
        n_ruido
    )

    print(
        "Porcentaje de ruido:",
        round(porcentaje_ruido, 2),
        "%"
    )

    # --------------------------------------
    # EVALUAR DBSCAN
    # --------------------------------------

    print("\nEvaluando DBSCAN...")

    # Excluir registros considerados ruido
    mascara = etiquetas_dbscan != -1

    X_sin_ruido = X_scaled[mascara]

    etiquetas_sin_ruido = etiquetas_dbscan[
        mascara
    ]

    n_clusters_validos = len(
        set(etiquetas_sin_ruido)
    )

    silhouette_dbscan = np.nan

    if (
        2 <= n_clusters_validos
        < len(etiquetas_sin_ruido)
    ):

        silhouette_dbscan = silhouette_score(
            X_sin_ruido,
            etiquetas_sin_ruido,
            sample_size=min(
                2000,
                len(etiquetas_sin_ruido)
            ),
            random_state=42
        )

        print(
            "Silhouette Score DBSCAN:",
            round(silhouette_dbscan, 4)
        )

    else:

        print(
            "No es posible calcular "
            "Silhouette Score con los "
            "clusters obtenidos."
        )

    # Guardar modelo DBSCAN
    joblib.dump(
        dbscan,
        MODELS_DIR / "dbscan.pkl"
    )

    print(
        "\nModelo guardado en models/dbscan.pkl"
    )

    # --------------------------------------
    # COMPARACION FINAL
    # --------------------------------------

    comparacion = pd.DataFrame({
        "Modelo": [
            "K-Means",
            "DBSCAN"
        ],
        "Clusters": [
            k_optimo,
            n_clusters
        ],
        "Silhouette Score": [
            silhouette_kmeans,
            silhouette_dbscan
        ],
        "Registros ruido": [
            0,
            n_ruido
        ]
    })

    print(
        "\n========== COMPARACION FINAL =========="
    )

    print(
        comparacion.to_string(index=False)
    )

    comparacion.to_csv(
        PROCESSED_DIR / "comparacion_clustering.csv",
        index=False
    )

    print("\nArchivos guardados correctamente.")
    print("models/kmeans.pkl")
    print("models/dbscan.pkl")
    print("models/unsupervised_scaler.pkl")
    print("data/processed/metodo_codo.csv")
    print("data/processed/metodo_codo.png")
    print("data/processed/comparacion_clustering.csv")

    print("\nClase 7 ejecutada correctamente.")
