import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from data_loader import load_raw_data, preprocess
from sklearn.preprocessing import RobustScaler


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "sensor_readings_24.data"

MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ENTRENAMIENTO DE K-MEANS
def train_kmeans(X_scaled):
    
    """
    Agrupa las lecturas de los sensores del robot
    utilizando el algoritmo K-Means.
    """

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    etiquetas = model.fit_predict(X_scaled)

    return model, etiquetas

# ENTRENAMIENTO DE DBSCAN
def train_dbscan(X_scaled):
    """
    Agrupa las lecturas de los sensores utilizando
    DBSCAN y detecta posibles valores atipicos.
    """

    model = DBSCAN(
        eps=3.0,
        min_samples=10
    )

    etiquetas = model.fit_predict(X_scaled)

    return model, etiquetas


# PROGRAMA PRINCIPAL
if __name__ == "__main__":

    # 1. Cargar el dataset original
    df = load_raw_data(DATA_PATH)

    # 2. Separar los sensores de la clase objetivo
    X, y = preprocess(df)

    # 3. Escalar los 24 sensores
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("Datos preparados correctamente.")
    print("Registros:", X_scaled.shape[0])
    print("Sensores:", X_scaled.shape[1])

    # 4. Entrenar K-Means
    print("\nEntrenando K-Means...")

    kmeans, etiquetas_kmeans = train_kmeans(X_scaled)

    print("K-Means entrenado correctamente.")

    # 5. Mostrar cantidad de registros por grupo
    print("\nCantidad de registros por cluster:")

    grupos = pd.Series(etiquetas_kmeans).value_counts().sort_index()

    print(grupos)
    
        # 6. Evaluar la calidad de los clusters
    print("\nEvaluando K-Means...")

    silhouette = silhouette_score(
        X_scaled,
        etiquetas_kmeans
    )

    print("Silhouette Score:", round(silhouette, 4))   

    # 6. Guardar el modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        kmeans,
        MODELS_DIR / "kmeans.pkl"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "unsupervised_scaler.pkl"
    )

    print("\nModelo guardado en models/kmeans.pkl")

    # 7. Entrenar DBSCAN
    print("\nEntrenando DBSCAN...")

    dbscan, etiquetas_dbscan = train_dbscan(X_scaled)

    print("DBSCAN entrenado correctamente.")

    # 8. Mostrar resultados de DBSCAN
    print("\nCantidad de registros por cluster:")

    grupos_dbscan = pd.Series(
        etiquetas_dbscan
    ).value_counts().sort_index()

    print(grupos_dbscan)

    # 9. Contar clusters y valores atipicos
    n_clusters = len(
        set(etiquetas_dbscan) - {-1}
    )

    n_ruido = np.sum(etiquetas_dbscan == -1)

    print("\nClusters encontrados:", n_clusters)
    print("Registros considerados ruido:", n_ruido)
        # 11. Evaluar DBSCAN
    print("\nEvaluando DBSCAN...")

    # Excluir los registros considerados ruido
    mascara = etiquetas_dbscan != -1

    X_sin_ruido = X_scaled[mascara]
    etiquetas_sin_ruido = etiquetas_dbscan[mascara]

    # Comprobar que existan al menos dos clusters
    n_clusters_validos = len(set(etiquetas_sin_ruido))

    if 2 <= n_clusters_validos < len(etiquetas_sin_ruido):

        silhouette_dbscan = silhouette_score(
            X_sin_ruido,
            etiquetas_sin_ruido,
            sample_size=min(2000, len(etiquetas_sin_ruido)),
            random_state=42
        )

        print(
            "Silhouette Score DBSCAN:",
            round(silhouette_dbscan, 4)
        )

    else:
        print(
            "No es posible calcular Silhouette Score "
            "con los clusters obtenidos."
        )

    # 10. Guardar DBSCAN
    joblib.dump(
        dbscan,
        MODELS_DIR / "dbscan.pkl"
    )

    print("\nModelo guardado en models/dbscan.pkl")