# Práctica 1 – Machine Learning aplicado a Robótica

**Estudiante:** Genesis Morales  
**Asignatura:** Machine Learning – RIA-805  
**Carrera:** Ingeniería en Robótica e Inteligencia Artificial  
**Universidad:** Universidad Católica de Cuenca

## 1. Descripción del proyecto

Este proyecto desarrolla un pipeline de Machine Learning aplicado al análisis de datos de sensores de un robot móvil.

Se utilizan técnicas de análisis exploratorio de datos (EDA), preprocesamiento, aprendizaje supervisado, aprendizaje no supervisado, evaluación de modelos y visualización de resultados.

El proyecto está organizado en nueve clases prácticas que permiten desarrollar un flujo de trabajo reproducible utilizando Python, scikit-learn, Jupyter Notebook y GitHub.

## 2. Dataset utilizado

**Nombre:** Wall-Following Robot Navigation  
**Fuente:** UCI Machine Learning Repository  
**Archivo:** `sensor_readings_24.data`

El conjunto de datos contiene:

- 5456 registros.
- 24 variables numéricas correspondientes a lecturas de sensores.
- Una variable objetivo que representa cuatro clases de navegación.

El objetivo es analizar los patrones de los sensores y aplicar algoritmos de clasificación y agrupamiento.

## 3. Objetivos

### Objetivo general

Desarrollar un pipeline de Machine Learning para procesar, analizar y modelar datos de sensores robóticos mediante técnicas de aprendizaje supervisado y no supervisado.

### Objetivos específicos

1. Realizar un análisis exploratorio para comprender las características del dataset.
2. Implementar un proceso de preparación, división y escalado de los datos.
3. Entrenar y evaluar algoritmos de clasificación supervisada.
4. Implementar técnicas de agrupamiento no supervisado.
5. Visualizar los resultados mediante PCA y t-SNE.
6. Medir la latencia de inferencia y documentar los resultados obtenidos.

## 4. Estructura del proyecto

```text
mi-practica1-Morales/
│
├── data/
│   ├── raw/
│   │   └── sensor_readings_24.data
│   └── processed/
│       ├── dataset.parquet
│       ├── eda_summary.json
│       ├── comparacion_modelos.csv
│       ├── metodo_codo.csv
│       ├── metodo_codo.png
│       ├── comparacion_clustering.csv
│       └── benchmark_latency.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_supervised.ipynb
│   └── 03_unsupervised.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── train_supervised.py
│   └── train_unsupervised.py
│
├── models/
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── svm.pkl
│   ├── robust_scaler.pkl
│   ├── kmeans.pkl
│   ├── dbscan.pkl
│   └── unsupervised_scaler.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

Los archivos de resultados y modelos se generan durante la ejecución de los programas. Su disponibilidad en GitHub depende de la configuración de `.gitignore`.

## 5. Tecnologías utilizadas

- Python: lenguaje principal de programación.
- Pandas y NumPy: manipulación y procesamiento de datos.
- Scikit-learn: entrenamiento y evaluación de algoritmos.
- Matplotlib: generación de visualizaciones.
- Joblib: almacenamiento de modelos entrenados.
- PyArrow: almacenamiento de datos en formato Parquet.
- Jupyter Notebook: experimentación y presentación de resultados.
- Git y GitHub: control de versiones.

## 6. Instalación y configuración

### Clonar el repositorio

```bash
git clone https://github.com/Genesis1977/mi-practica1-Morales.git
cd mi-practica1-Morales
```

### Crear un entorno virtual

```bash
python -m venv venv
```

### Activar el entorno virtual en Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Instalar las dependencias

```bash
python -m pip install -r requirements.txt
```

### Ejecutar los programas

Preparación de datos:

```bash
python src/data_loader.py
```

Entrenamiento supervisado:

```bash
python src/train_supervised.py
```

Entrenamiento no supervisado:

```bash
python src/train_unsupervised.py
```

El programa no supervisado muestra la gráfica del método del codo y solicita seleccionar un valor de `k` entre 2 y 10.

Para reproducir los experimentos realizados se seleccionó `k=4`.

Los notebooks deben abrirse en VS Code o Jupyter utilizando el entorno de Python configurado. Se recomienda ejecutar sus celdas en orden.

## 7. Desarrollo de las clases

| Clase | Actividad |
|---|---|
| 1 | Configuración del repositorio y estructura del proyecto |
| 2 | Carga y exploración inicial del dataset |
| 3 | Análisis exploratorio de datos y estadísticas |
| 4 | Preprocesamiento, división estratificada y escalado |
| 5 | Entrenamiento y evaluación de Random Forest |
| 6 | Comparación de Random Forest, KNN y SVM |
| 7 | Agrupamiento mediante K-Means y DBSCAN |
| 8 | Visualizaciones PCA, t-SNE y benchmark de latencia |
| 9 | Documentación, reproducibilidad y entrega final |

## 8. Preprocesamiento

Se implementó un proceso que incluye:

1. Carga del conjunto de datos original.
2. Separación de las 24 variables predictoras y la variable objetivo.
3. División estratificada del 80 % para entrenamiento y 20 % para prueba.
4. Aplicación de RobustScaler.
5. Almacenamiento del dataset procesado en formato Parquet.

Para los modelos supervisados, el escalador se ajusta únicamente con los datos de entrenamiento y después transforma los datos de prueba. Este procedimiento evita la fuga de información (*data leakage*) durante el preprocesamiento.

## 9. Aprendizaje supervisado

Se implementaron tres algoritmos de clasificación:

- Random Forest.
- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM).

### Resultados obtenidos

| Modelo | Accuracy | F1-score ponderado |
|---|---:|---:|
| Random Forest | 0.9927 | 0.9927 |
| KNN | 0.8791 | 0.8789 |
| SVM | 0.8690 | 0.8684 |

Random Forest presentó el mejor rendimiento predictivo de los tres modelos evaluados.

También se generaron matrices de confusión, una comparación gráfica y mediciones del tiempo de entrenamiento.

Los resultados completos, incluidos los tiempos medidos, se encuentran en `data/processed/comparacion_modelos.csv`.

## 10. Aprendizaje no supervisado

Se implementaron dos algoritmos de agrupamiento:

**K-Means:** permite dividir los datos en un número determinado de clústeres. Se aplicó el método del codo para estudiar diferentes valores de `k`.

**DBSCAN:** permite identificar agrupamientos basados en densidad y registros considerados ruido.

### Resultados obtenidos

| Métrica | K-Means | DBSCAN |
|---|---:|---:|
| Clústeres | 4 | 32 |
| Silhouette Score | 0.4709 | 0.1834 |
| Registros considerados ruido | 0 | 849 |

K-Means obtuvo un Silhouette Score superior en la configuración evaluada. DBSCAN identificó 849 registros considerados ruido, equivalentes aproximadamente al 15,56 % del dataset.

La comparación de Silhouette Score es orientativa, porque DBSCAN fue evaluado excluyendo los registros considerados ruido.

## 11. Visualización mediante PCA y t-SNE

Se utilizaron dos técnicas de reducción de dimensionalidad:

**PCA:** transforma los datos en componentes principales para representar su variabilidad en dos dimensiones.

**t-SNE:** genera una representación bidimensional orientada a explorar relaciones de proximidad entre observaciones.

Ambas técnicas permitieron visualizar y comparar los agrupamientos obtenidos mediante K-Means y DBSCAN.

Las visualizaciones se encuentran en `notebooks/03_unsupervised.ipynb`.

## 12. Benchmark de latencia

Se implementó una prueba de rendimiento para medir el tiempo de inferencia del modelo K-Means.

El procedimiento consistió en:

1. Seleccionar una observación del dataset procesado.
2. Realizar cinco predicciones de calentamiento.
3. Ejecutar 100 predicciones individuales.
4. Medir los tiempos mediante `time.perf_counter()`.
5. Calcular la latencia promedio, mínima y máxima.
6. Guardar los resultados en `benchmark_latency.json`.

Esta prueba mide la latencia de inferencia de K-Means en el entorno utilizado. No representa el tiempo total de respuesta de un robot físico.

## 13. Reproducibilidad

Para reproducir los experimentos se recomienda:

- Utilizar las dependencias indicadas en `requirements.txt`.
- Ejecutar los comandos desde la carpeta principal del proyecto.
- Mantener el dataset original sin modificaciones.
- Conservar `random_state=42` en los algoritmos que lo utilizan.
- Ejecutar los scripts antes de los notebooks que dependen de modelos guardados.
- Utilizar `k=4` y `eps=3.0` para reproducir la configuración no supervisada evaluada.

Los tiempos de entrenamiento e inferencia pueden variar dependiendo del equipo y del entorno de ejecución.

## 14. Conclusiones generales

Se desarrolló un pipeline de Machine Learning aplicado a datos reales de sensores robóticos, integrando las principales etapas de preparación, entrenamiento, evaluación y documentación.

En el aprendizaje supervisado, Random Forest presentó el mejor rendimiento predictivo entre los tres algoritmos evaluados, con un Accuracy y F1-score ponderado aproximados del 99,27 %.

En el aprendizaje no supervisado, K-Means permitió generar cuatro agrupamientos con una separación moderada, mientras que DBSCAN identificó estructuras basadas en densidad y registros considerados ruido.

Las técnicas PCA y t-SNE facilitaron la exploración visual de los resultados. Además, el benchmark de latencia permitió incorporar una medición básica del costo de inferencia de K-Means.

El proyecto demuestra la aplicación de técnicas de Machine Learning a datos de robótica. Los resultados corresponden a experimentos con un dataset y no constituyen una validación del sistema en un robot físico.

## 15. Referencia del dataset

UCI Machine Learning Repository. *Wall-Following Robot Navigation Data Set*.

https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data

## 16. Repositorio

GitHub: https://github.com/Genesis1977/mi-practica1-Morales

## Práctica 2 — Sesión 2: Matriz de costos

En esta sesión se evalúa el impacto económico simulado de los errores de clasificación de Random Forest y de un modelo trivial, utilizando el problema binario definido en la Sesión 1.

Se asigna un costo de **$50 por falso positivo (FP)** y **$500 por falso negativo (FN)**, conforme a los valores establecidos en la actividad.

Estos costos representan un escenario hipotético de navegación robótica. Un falso positivo podría provocar una respuesta preventiva innecesaria, mientras que un falso negativo podría representar la omisión de una condición relevante para la navegación. Por esta razón, se considera que el falso negativo tiene un impacto diez veces mayor.

El costo total se calcula mediante:

**Costo total = (FP × $50) + (FN × $500)**

La comparación permite determinar si Random Forest reduce los costos simulados respecto a un clasificador trivial que siempre predice la clase negativa.

## Práctica 2 — Evaluación, optimización y aprendizaje por refuerzo

### Descripción

La Práctica 2 continúa el trabajo realizado con el dataset *Wall-Following Robot Navigation*, incorporando evaluación de modelos en escenarios desbalanceados, análisis de costos, ajuste del umbral de clasificación y optimización de hiperparámetros.

Las sesiones 1 a 5 utilizan Random Forest y un clasificador trivial para estudiar el impacto de los errores de predicción. Las siguientes sesiones abordarán aprendizaje por refuerzo con Gymnasium.

### Sesión 1 — Métricas para clases desbalanceadas

Se transformó el problema original en una clasificación binaria, considerando como clase positiva a `Slight-Left-Turn`, identificada como clase minoritaria.

Se compararon Random Forest y un modelo trivial que predice siempre la clase negativa.

El modelo trivial obtuvo aproximadamente 93,96 % de Accuracy, pero un F1-score de 0 para la clase positiva. Esto demuestra que una exactitud elevada no garantiza un buen desempeño cuando las clases están desbalanceadas.

Se utilizaron Accuracy, Precision, Recall, F1-score y matriz de confusión para evaluar los modelos.

**Archivo:** `notebooks/notebooks/p2_01_metricas.ipynb`

### Sesión 2 — Matriz de costos

Se implementó una matriz de costos para evaluar las consecuencias simuladas de los errores de clasificación.

Se asignaron los siguientes valores:

- Falso positivo (FP): $50.
- Falso negativo (FN): $500.

**Fórmula:**

Costo total = (FP × 50) + (FN × 500)

Los costos representan supuestos académicos para un escenario hipotético de navegación robótica. No corresponden a costos económicos reales medidos en un robot.

Se asignó un costo mayor a los falsos negativos porque representan situaciones positivas que el clasificador no consigue detectar.

**Archivo:** `src/p2_02_costos.py`

### Sesión 3 — Threshold Tuning

Se implementó un procedimiento de ajuste del umbral de decisión para reducir el costo simulado de los errores de clasificación.

Los datos originales de entrenamiento se dividieron nuevamente en entrenamiento y validación mediante una partición estratificada.

Se evaluaron umbrales entre 0.10 y 0.90, con incrementos de 0.05. El umbral de menor costo se seleccionó utilizando exclusivamente el conjunto de validación.

Posteriormente, se entrenó un modelo con todo el conjunto original de entrenamiento y se evaluó el umbral seleccionado sobre el conjunto de prueba.

Este procedimiento evita utilizar los datos de prueba para seleccionar el umbral, reduciendo el riesgo de fuga de información.

**Archivos:**

- `src/p2_03_threshold.py`
- `reports/figures/threshold_tuning.png`
- `data/processed/p2_threshold_validacion.csv`
- `data/processed/p2_threshold_comparacion.csv`

### Sesión 4 — Optimización de hiperparámetros con Optuna

Se utilizó Optuna para optimizar los hiperparámetros de Random Forest mediante 20 pruebas y validación cruzada estratificada de tres particiones.

Los hiperparámetros evaluados fueron `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`.

**Resultados:**

- Pruebas completadas: 20.
- Mejor F1 promedio de validación cruzada: 0.9580.
- Número de árboles: 175.
- Profundidad máxima: 8.
- Mínimo de muestras para dividir: 6.
- Mínimo de muestras por hoja: 1.
- Variables consideradas por división: `log2`.

El estudio se almacena mediante SQLite y permite recuperar las pruebas existentes sin ejecutar nuevas pruebas innecesariamente.

**Archivo:** `src/p2_04_optuna.py`

### Sesión 5 — Análisis y comparación de modelos

Se analizaron los resultados obtenidos mediante Optuna, incluyendo el historial de optimización y la importancia relativa de los hiperparámetros.

Se compararon Random Forest original y Random Forest optimizado sobre el conjunto de prueba.

| Métrica | Random Forest original | Random Forest Optuna |
|---|---:|---:|
| Accuracy | 0.996337 | 0.997253 |
| Precision | 0.984375 | 0.984615 |
| Recall | 0.954545 | 0.969697 |
| F1-score | 0.969231 | 0.977099 |
| Falsos positivos | 1 | 1 |
| Falsos negativos | 3 | 2 |
| Costo simulado | $1,550 | $1,050 |

El modelo optimizado mejoró el F1-score y redujo de tres a dos los falsos negativos, manteniendo un falso positivo.

La reducción del costo simulado fue de aproximadamente 32.26 %, equivalente a $500.

Estos resultados corresponden a la evaluación experimental del dataset y no constituyen una validación sobre un robot físico.

**Archivos:**

- `src/p2_05_optuna_analisis.py`
- `config.json`
- `reports/figures/optuna_historial.png`
- `reports/figures/optuna_parametros.png`

### Ejecución de las sesiones

Los siguientes comandos deben ejecutarse desde la carpeta principal del repositorio, con el entorno virtual y las dependencias instaladas.

```powershell
.\venv\Scripts\python.exe src\p2_02_costos.py
.\venv\Scripts\python.exe src\p2_03_threshold.py
.\venv\Scripts\python.exe src\p2_04_optuna.py
.\venv\Scripts\python.exe src\p2_05_optuna_analisis.py
```

La Sesión 1 se ejecuta desde su notebook utilizando el intérprete de Python del entorno virtual.

Los scripts requieren el archivo `data/processed/dataset.parquet`, generado durante la Práctica 1.

Para reproducir la optimización de Optuna se recomienda ejecutar primero la Sesión 4 y después la Sesión 5.

### Estado del proyecto

Las sesiones 1 a 5 de la Práctica 2 se encuentran implementadas. Se comprobó la ejecución correcta de las sesiones 4 y 5, incluyendo la persistencia de las 20 pruebas de Optuna y la comparación final de modelos.

La siguiente etapa corresponde a la Sesión 6, dedicada a la introducción al aprendizaje por refuerzo mediante Gymnasium y el entorno CartPole.

### Sesión 6 — Introducción a Gymnasium y CartPole

**Objetivo:** implementar un agente aleatorio en el entorno `CartPole-v1` de Gymnasium y analizar las recompensas obtenidas durante cinco episodios.

Se utilizó Gymnasium para crear el entorno, reiniciar los episodios, seleccionar acciones aleatorias y registrar las recompensas acumuladas.

El agente no realiza aprendizaje. Su comportamiento sirve como referencia inicial para futuras comparaciones con agentes entrenados mediante aprendizaje por refuerzo.

**Resultados obtenidos:**

| Episodio | Pasos | Recompensa |
|---|---:|---:|
| 1 | 93 | 93 |
| 2 | 44 | 44 |
| 3 | 10 | 10 |
| 4 | 20 | 20 |
| 5 | 20 | 20 |

- Recompensa promedio: 37.40.
- Recompensa mínima: 10.
- Recompensa máxima: 93.
- Episodios ejecutados: 5.

Los resultados muestran variaciones entre episodios debido a la selección aleatoria de acciones. El agente no mantiene un rendimiento constante, ya que no dispone de una política aprendida.

**Archivos:**

- `src/p2_06_cartpole_random.py`
- `reports/figures/p2_cartpole_random.png`
- `data/processed/p2_cartpole_random.csv` (generado localmente).

### Sesión 7 — Entorno personalizado con Gymnasium

**Objetivo:** desarrollar un entorno personalizado compatible con Gymnasium, definiendo observaciones, acciones, recompensas y condiciones de finalización.

Se implementó `RobotLineEnv`, un entorno simplificado que representa el desplazamiento de un robot sobre una línea de posiciones.

**Características del entorno:**

- Posición inicial: 0.
- Posición objetivo: 10.
- Acción 0: desplazamiento hacia la izquierda.
- Acción 1: desplazamiento hacia la derecha.
- Recompensa por movimiento sin alcanzar la meta: -1.
- Recompensa al alcanzar la meta: +100.
- Límite máximo: 30 pasos por episodio.

El entorno utiliza los métodos `reset()`, `step()`, `render()` y `close()` de la interfaz de Gymnasium.

Se utilizó `check_env` de Stable Baselines3 para comprobar su compatibilidad.

Posteriormente, se evaluaron dos políticas: una aleatoria y otra basada en reglas que siempre avanza hacia la meta.

**Resultados obtenidos:**

| Métrica | Política aleatoria | Política basada en reglas |
|---|---:|---:|
| Episodios | 5 | 5 |
| Recompensa promedio | 15.0 | 91.0 |
| Pasos promedio | 25.4 | 10.0 |
| Tasa de éxito | 40 % | 100 % |

La política aleatoria alcanzó la meta en dos de los cinco episodios. La política basada en reglas alcanzó la meta en todos los episodios y necesitó únicamente diez pasos por ejecución.

Estos resultados permiten comprobar que el entorno responde correctamente a las acciones y que una política dirigida hacia el objetivo obtiene mejores resultados que una política aleatoria en esta simulación.

**Archivos:**

- `src/p2_07_robot_env.py`
- `src/p2_07_test_env.py`
- `data/processed/p2_robot_env_resultados.csv` (generado localmente).

### Conclusión de las sesiones 6 y 7

Se logró implementar un agente aleatorio en CartPole y desarrollar un entorno personalizado compatible con Gymnasium.

Las pruebas permitieron comprender el funcionamiento de los episodios, las observaciones, las acciones, las recompensas y las condiciones de terminación.

Estos experimentos establecen una base para el entrenamiento de agentes mediante algoritmos de aprendizaje por refuerzo, como PPO, en las siguientes sesiones.

### Sesión 8 — Entrenamiento de un agente PPO

**Objetivo:** entrenar un agente de aprendizaje por refuerzo utilizando el algoritmo Proximal Policy Optimization (PPO) en el entorno personalizado RobotLineEnv.

Se utilizó Stable Baselines3 con una política `MlpPolicy`, una tasa de aprendizaje de 0.0003, factor de descuento de 0.99 y un entrenamiento solicitado de 10.000 pasos.

El entorno representa un robot que debe desplazarse desde la posición 0 hasta la posición 10. El agente recibe una recompensa de +100 al alcanzar la meta y una penalización de -1 por cada paso sin alcanzarla.

Se implementó un callback para registrar la recompensa acumulada y la duración de los episodios. También se generó una gráfica del entrenamiento con un promedio móvil de 20 episodios.

**Archivos:**

- `src/p2_08_ppo_entrenamiento.py`
- `models/p2_ppo_robot.zip`
- `reports/figures/p2_ppo_entrenamiento.png`
- `data/processed/p2_ppo_entrenamiento.csv` (generado localmente)

**Comando de ejecución:**

```powershell
.\venv\Scripts\python.exe src\p2_08_ppo_entrenamiento.py
```

El modelo entrenado se utilizará en la Sesión 9 para comparar PPO con las políticas aleatoria y basada en reglas.

### Estado del proyecto

Las sesiones 1 a 8 de la Práctica 2 se encuentran implementadas.

Las sesiones 6 y 7 permitieron trabajar con Gymnasium y desarrollar un entorno personalizado. En la Sesión 8 se implementó el entrenamiento de un agente PPO.

La siguiente etapa corresponde a la Sesión 9, dedicada a evaluar y comparar las políticas PPO, aleatoria y basada en reglas.

### Sesión 9 — Evaluación comparativa de políticas

Se realizó una evaluación de tres políticas en el entorno personalizado RobotLineEnv: agente aleatorio, política basada en reglas y agente PPO entrenado.

Cada política fue evaluada durante 30 episodios utilizando las mismas condiciones del entorno.

Las métricas consideradas fueron:

- Recompensa acumulada promedio.
- Desviación estándar de las recompensas.
- Número promedio de pasos.
- Tasa de éxito al alcanzar la meta.

El modelo PPO se cargó desde `models/p2_ppo_robot.zip` y se evaluó sin realizar entrenamiento adicional.

Los resultados se almacenaron en archivos CSV y se representaron mediante gráficas comparativas.

**Archivos:**

- `src/p2_09_evaluacion_ppo.py`
- `data/processed/p2_09_episodios.csv`
- `data/processed/p2_09_comparacion.csv`
- `reports/figures/p2_09_recompensas.png`
- `reports/figures/p2_09_tasa_exito.png`

### Sesión 10 — Documentación de resultados

Se consolidaron los resultados de las sesiones 6 a 9 en un reporte reproducible.

El documento incluye las pruebas con CartPole, el funcionamiento del entorno personalizado, el entrenamiento PPO y la comparación de políticas.

Se desarrolló un script que recupera los resultados experimentales de los archivos CSV y genera un reporte en formato Markdown.

**Archivos:**

- `src/p2_10_reporte.py`
- `reports/p2_reporte_sesiones_6_9.md`

### Estado del proyecto

Las sesiones 1 a 10 de la Práctica 2 se encuentran desarrolladas, pendientes de la verificación final de ejecución de las sesiones 9 y 10.

Las siguientes actividades corresponden a la preparación de la presentación, el video demostrativo y la entrega final.

### Sesión 11 — Auditoría automática del proyecto

Se implementó un procedimiento de auditoría para verificar la presencia de los archivos de las sesiones anteriores, las dependencias de Python y el modelo PPO entrenado.

El script realiza una prueba de ejecución del modelo PPO sobre el entorno personalizado y registra los resultados.

También verifica la disponibilidad de los archivos CSV generados durante los experimentos.

**Archivos:**

- `src/p2_11_auditoria.py`
- `reports/p2_11_archivos.csv`
- `reports/p2_11_dependencias.csv`
- `reports/p2_11_resultados.csv`
- `reports/p2_11_auditoria.json`

**Ejecución:**

```powershell
.\venv\Scripts\python.exe src\p2_11_auditoria.py
```

### Sesión 12 — Informe y entrega final

Se implementó un procedimiento para consolidar los resultados experimentales y preparar la documentación final de la Práctica 2.

El informe reúne las actividades de clasificación supervisada, optimización de hiperparámetros y aprendizaje por refuerzo.

Se incluyen los resultados de la comparación de políticas y una revisión de las evidencias disponibles.

**Archivos:**

- `src/p2_12_entrega_final.py`
- `reports/p2_12_informe_final.md`
- `reports/p2_12_inventario.csv`

**Ejecución:**

```powershell
.\venv\Scripts\python.exe src\p2_12_entrega_final.py
```

### Estado final de la Práctica 2

Se desarrollaron las doce sesiones de la Práctica 2, integrando evaluación de clasificadores, análisis de costos, ajuste de umbrales, optimización mediante Optuna y aprendizaje por refuerzo con Gymnasium y Stable Baselines3.

En la evaluación de 30 episodios, el agente PPO obtuvo una recompensa promedio de 91,00 y una tasa de éxito del 100 %, igualando a la política basada en reglas en el entorno RobotLineEnv.

Las sesiones finales incorporan herramientas para auditar los archivos del proyecto, verificar el modelo entrenado y generar un informe consolidado.

La entrega queda sujeta a la ejecución satisfactoria de las verificaciones finales y a los requisitos establecidos por el docente.

## Práctica 2 — Evaluación de factibilidad técnica y optimización de algoritmos predictivos

### Objetivo

Evaluar y optimizar el pipeline de Machine Learning desarrollado en la Práctica 1, incorporando métricas para clases desbalanceadas, análisis económico de errores, optimización de hiperparámetros y aprendizaje por refuerzo.

La práctica utiliza el dataset **Wall-Following Robot Navigation** para la clasificación supervisada y el entorno **CartPole-v1 de Gymnasium** para los experimentos de aprendizaje por refuerzo.

### Tecnologías utilizadas

- Python, NumPy, Pandas y Matplotlib.
- Scikit-learn para clasificación y métricas.
- Optuna para optimización de hiperparámetros.
- Gymnasium para entornos de aprendizaje por refuerzo.
- Stable-Baselines3 para entrenamiento con PPO.
- Jupyter Notebook para experimentación.

### Desarrollo por sesiones

| Sesión | Actividad | Entregable |
|---|---|---|
| 1 | Evaluación de clases desbalanceadas | `p2_01_metricas.ipynb` |
| 2 | Matriz de costos FP/FN | `p2_02_costos.py` |
| 3 | Optimización del umbral de decisión | `p2_03_threshold.py` |
| 4 | Estudio de hiperparámetros con Optuna | `p2_04_optuna.py` |
| 5 | Análisis de Optuna y comparación de modelos | `p2_05_optuna_analisis.py` |
| 6 | Exploración de CartPole-v1 | `p2_06_gymnasium.ipynb` |
| 7 | Entorno personalizado RobotLineEnv | `p2_07_entorno_rl.py` |
| 8 | Entrenamiento de PPO | `p2_08_entrenamiento_rl.py` |
| 9 | Evaluación PPO frente a agente aleatorio | `p2_09_evaluacion_rl.py` |
| 10 | Comparación PPO frente a controlador clásico | `p2_10_comparacion.py` |
| 11 | Documentación y auditoría | `README.md` |
| 12 | Entrega y demostración | Repositorio y demo de 5 minutos |

### Instalación

Desde la raíz del proyecto, instalar las dependencias del archivo `requirements.txt` y las librerías utilizadas en la Práctica 2:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install optuna gymnasium stable-baselines3 matplotlib pandas pyarrow ipykernel
```

### Ejecución de las sesiones 2 a 5

```powershell
.\venv\Scripts\python.exe src\p2_02_costos.py
.\venv\Scripts\python.exe src\p2_03_threshold.py
.\venv\Scripts\python.exe src\p2_04_optuna.py
.\venv\Scripts\python.exe src\p2_05_optuna_analisis.py
```

Los notebooks de las sesiones 1 y 6 se ejecutan desde VS Code o Jupyter, seleccionando el entorno virtual del proyecto.

### Ejecución de las sesiones 7 a 10

```powershell
.\venv\Scripts\python.exe src\p2_07_entorno_rl.py
.\venv\Scripts\python.exe src\p2_08_entrenamiento_rl.py
.\venv\Scripts\python.exe src\p2_09_evaluacion_rl.py
.\venv\Scripts\python.exe src\p2_10_comparacion.py
```

**Nota:** La sesión 8 entrena el modelo PPO y guarda `models/ppo_cartpole.zip`. Las sesiones 9 y 10 utilizan ese archivo para evaluar la política aprendida sin repetir el entrenamiento.

### Evaluación económica

Se utiliza una matriz de costos académica:

- Falso positivo (FP): 50 USD.
- Falso negativo (FN): 500 USD.

**Costo total = FP × 50 + FN × 500**

El costo de los falsos negativos es mayor debido al supuesto de que omitir una condición relevante para la navegación podría tener consecuencias más graves que generar una alerta innecesaria.

Los valores representan un escenario simulado, no costos medidos en un robot físico.

### Threshold Tuning y Optuna

Se evalúan umbrales de clasificación entre 0.10 y 0.90, con incrementos de 0.05, seleccionando el umbral que minimiza el costo en validación.

Posteriormente se utiliza Optuna para optimizar los hiperparámetros de Random Forest mediante validación cruzada estratificada. El estudio contempla al menos 20 pruebas.

La configuración seleccionada se exporta a `config.json` y se compara con el modelo original utilizando métricas predictivas y costos.

### Aprendizaje por refuerzo

**Sesión 6:** exploración de CartPole-v1 mediante cinco episodios con acciones aleatorias.

**Sesión 7:** desarrollo de un entorno personalizado de navegación lineal, con estados, acciones, recompensas y condiciones de terminación.

**Sesión 8:** entrenamiento de PPO en CartPole-v1 con 20.000 pasos solicitados. Se guarda el modelo entrenado y se registra la evolución de las recompensas.

**Sesión 9:** evaluación de PPO y de un agente aleatorio durante diez episodios por agente, calculando recompensas promedio, máximas y mínimas.

**Sesión 10:** comparación de PPO con un controlador clásico basado en reglas durante diez episodios por controlador.

### Evidencias y resultados

Las evidencias se almacenan en las carpetas `reports/`, `reports/figures/`, `data/processed/` y `models/`.

Entre los principales archivos generados se encuentran:

- `reports/figures/threshold_tuning.png`
- `reports/figures/optuna_historial.png`
- `reports/figures/optuna_parametros.png`
- `data/processed/p2_comparacion_optuna.csv`
- `config.json`
- `models/ppo_cartpole.zip`
- `reports/figures/p2_08_recompensas.png`
- `reports/p2_08_entrenamiento.json`
- `reports/p2_09_evaluacion_rl.json`
- `reports/p2_09_evaluacion_rl.csv`
- `reports/figures/p2_09_comparacion_rl.png`
- `reports/p2_10_resultados.json`
- `reports/p2_10_comparacion.csv`
- `reports/figures/p2_10_comparacion.png`

Los valores experimentales deben consultarse en los reportes generados durante la ejecución.

### Interpretación y limitaciones

El análisis económico permite comparar modelos según las consecuencias simuladas de sus errores y no únicamente según Accuracy.

Optuna permite explorar distintas configuraciones de Random Forest, aunque la mejora obtenida debe comprobarse mediante las métricas y costos de evaluación.

La comparación de PPO, el agente aleatorio y el controlador clásico permite estudiar diferentes estrategias de control en CartPole-v1.

Los resultados corresponden a simulaciones y experimentos académicos. No constituyen una validación de navegación autónoma en un robot físico.

### Reproducibilidad

Se utilizan semillas fijas cuando corresponde, se separan los conjuntos de entrenamiento, validación y prueba, y se conservan los modelos y reportes generados.

El rendimiento y los tiempos de ejecución pueden variar según el equipo y las versiones de las dependencias.

# Práctica 3 — Despliegue, Profiling y Green AI

## 1. Descripción

La Práctica 3 implementa el despliegue de un modelo de clasificación Random Forest para el problema *Wall-Following Robot Navigation*. El modelo entrenado se exporta al formato ONNX y se integra en una API REST desarrollada con FastAPI.

La API utiliza Pydantic para validar las solicitudes y Docker para facilitar su ejecución en un entorno reproducible. También se realizan pruebas de rendimiento y un análisis estimado de sostenibilidad energética.

## 2. Objetivos

- Exportar un modelo Random Forest al formato ONNX.
- Implementar una API REST con los endpoints `/health` y `/predict`.
- Validar la estructura y los valores de las solicitudes mediante Pydantic.
- Desplegar la API en un contenedor Docker.
- Medir RAM y latencia mediante 100 solicitudes.
- Comparar el rendimiento de ONNX y Random Forest original.
- Estimar consumo energético y emisiones de CO₂ equivalente.

## 3. Tecnologías

Python 3.10, scikit-learn, ONNX Runtime, FastAPI, Pydantic, Docker, NumPy, Pandas, Matplotlib y GitHub.

## 4. Archivos principales

| Archivo | Función |
|---|---|
| `app/api.py` | API REST y validación de datos |
| `models/modelo.onnx` | Modelo exportado a ONNX |
| `models/random_forest.pkl` | Modelo original |
| `Dockerfile` | Construcción del contenedor |
| `requirements-api.txt` | Dependencias de la API |
| `src/p3_06_profiling.py` | Medición de RAM y latencia |
| `src/p3_07_green_ai.py` | Estimación energética y de emisiones |
| `reports/profiling.json` | Resultados numéricos de profiling |
| `reports/profiling.png` | Gráfico comparativo |
| `reports/green_ai.md` | Informe de sostenibilidad |

## 5. Instalación

Clonar el repositorio:

```powershell
git clone https://github.com/Genesis1977/mi-practica1-Morales.git
cd mi-practica1-Morales
```

Crear y activar el entorno virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instalar las dependencias generales:

```powershell
python -m pip install -r requirements.txt
```

Para ejecutar la API localmente:

```powershell
python -m pip install -r requirements-api.txt
python -m uvicorn app.api:app --reload
```

La documentación interactiva estará disponible en `http://localhost:8000/docs`.

## 6. Despliegue con Docker

Construir la imagen desde la raíz del proyecto:

```powershell
docker build -t ml-api .
```

Iniciar el contenedor:

```powershell
docker run --name practica3-api -p 8000:8000 ml-api
```

Comprobar su estado:

```powershell
docker ps
```

Verificar la API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

La ruta `POST /predict` recibe un objeto JSON con una lista `sensores` de exactamente 24 valores numéricos.

**Contrato de entrada:** los sensores deben encontrarse en el mismo orden y escala transformada utilizados para entrenar el modelo. La API no aplica automáticamente el escalador a lecturas sin procesar.

## 7. Validación Pydantic

La API comprueba la cantidad de sensores, los tipos numéricos, la ausencia de valores booleanos o cadenas, la finitud de los datos, su compatibilidad con `float32` y la ausencia de campos adicionales.

**Limitación:** los límites específicos de dominio para cada sensor aún requieren justificación. Los valores transformados del dataset pueden ser negativos y superiores a uno, por lo que no corresponde aplicar indiscriminadamente el intervalo `[0,1]`.

Las entradas inválidas producen respuestas HTTP 422.

## 8. Resultados de Profiling — Sesión 6

Se realizaron 100 peticiones HTTP a la API y 100 inferencias directas para cada uno de los dos modelos.

| Método | Latencia promedio |
|---|---:|
| API FastAPI en Docker | 48.0927 ms |
| ONNX directo | 0.0486 ms |
| Random Forest original | 37.6765 ms |

La memoria del contenedor pasó de 69.877 MB a 69.982 MB, una diferencia de 0.105 MB entre dos mediciones puntuales.

Las predicciones de los métodos coincidieron para la muestra evaluada.

Las pruebas directas se ejecutaron en el entorno local y las peticiones HTTP se realizaron contra Docker. Por tanto, los tiempos no representan una comparación aislada de los motores de inferencia bajo idénticas condiciones.

Reproducir el profiling:

```powershell
python -m src.p3_06_profiling
```

Resultados: `reports/profiling.json` y `reports/profiling.png`.

## 9. Green AI — Sesión 7

Se estimó el consumo energético mediante:

**Energía (Wh) = Potencia (W) × Tiempo (s) / 3600**

La huella de carbono se estimó utilizando la energía en kWh y un factor de emisión expresado en kg CO₂e/kWh.

Se emplearon dos supuestos ilustrativos: potencia constante de 50 W y factor de emisión de 0.4 kg CO₂e/kWh.

| Método | Energía estimada para 100 operaciones | CO₂e estimado |
|---|---:|---:|
| API Docker | 0.066795 Wh | 0.026718 g |
| ONNX directo | 0.0000675 Wh | 0.000027 g |
| Random Forest original | 0.052328 Wh | 0.020931 g |

Los resultados son estimaciones basadas en latencias y supuestos de potencia, no mediciones eléctricas directas.

### Propuestas de mejora

**Optimización 1 — Uso de ONNX Runtime.** Evaluar el uso de ONNX para reducir la latencia de inferencia y el tiempo activo de procesamiento, verificando equivalencia de predicciones y consumo energético bajo condiciones comparables.

**Optimización 2 — Reducción de recursos de despliegue.** Mantener una imagen Docker ligera, instalar únicamente las dependencias necesarias para inferencia y evitar procesos o servicios innecesarios. Evaluar el impacto en memoria, tiempo de arranque y consumo.

Reproducir el informe:

```powershell
python -m src.p3_07_green_ai
```

Entregable: `reports/green_ai.md`.

## 10. Conclusiones

La Práctica 3 permitió integrar un modelo Random Forest con ONNX, FastAPI, Pydantic y Docker, obteniendo un servicio funcional de clasificación a partir de 24 sensores transformados.

El profiling mostró una latencia directa de ONNX inferior a la del modelo original en las condiciones experimentales evaluadas. El análisis Green AI permitió relacionar tiempo de ejecución, energía estimada y emisiones hipotéticas.

Los resultados corresponden a pruebas sobre un dataset y una API local; no representan una validación física de un robot ni una medición directa de consumo eléctrico.