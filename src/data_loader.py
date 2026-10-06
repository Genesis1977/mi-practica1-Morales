import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_raw_data(path):
    """
    Carga el dataset Wall-Following Robot Navigation
    utilizando la representación de 24 sensores.

    Parameters
    ----------
    path : str
        Ruta del archivo sensor_readings_24.data.

    Returns
    -------
    pandas.DataFrame
        Dataset con los nombres de las columnas asignados.
    """

    columnas = [f"sensor_{i}" for i in range(1, 25)] + ["clase"]

    df = pd.read_csv(
        path,
        header=None,
        names=columnas
    )

    return df

def preprocess(df):
    """
    Separa las variables predictoras de la variable objetivo.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset completo.

    Returns
    -------
    X : pandas.DataFrame
        Variables predictoras correspondientes a los 24 sensores.
    y : pandas.Series
        Variable objetivo con las clases de navegación.
    """

    X = df.drop(columns=["clase"])
    y = df["clase"]

    return X, y

def split_data(X, y):
    """
    Divide los datos en entrenamiento y prueba de forma estratificada
    y aplica RobustScaler evitando Data Leakage.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras.
    y : pandas.Series
        Variable objetivo.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
        Datos divididos, escalados y el scaler ajustado.
    """

    # División estratificada: 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Crear el escalador
    scaler = RobustScaler()

    # El scaler aprende ÚNICAMENTE de los datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)

    # Los datos de prueba se transforman con el scaler ya ajustado
    X_test_scaled = scaler.transform(X_test)

    # Recuperar los nombres de las columnas
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":

    # 1. Cargar dataset original
    df = load_raw_data("data/raw/sensor_readings_24.data")

    # 2. Separar variables predictoras y variable objetivo
    X, y = preprocess(df)

    # 3. Dividir y escalar los datos
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    # 4. Unir entrenamiento y prueba para guardar el dataset procesado
    train_data = X_train.copy()
    train_data["clase"] = y_train
    train_data["conjunto"] = "train"

    test_data = X_test.copy()
    test_data["clase"] = y_test
    test_data["conjunto"] = "test"

    dataset_processed = pd.concat([train_data, test_data])

    # 5. Guardar en formato Parquet
    dataset_processed.to_parquet(
        "data/processed/dataset.parquet",
        index=False
    )

    print("Dataset procesado correctamente.")
    print("Dimensiones:", dataset_processed.shape)
    print("Archivo generado: data/processed/dataset.parquet")