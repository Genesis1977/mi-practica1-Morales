# Práctica 3 — Sesión 7: Green AI

**Proyecto:** Robot Wall-Following — Random Forest / ONNX / FastAPI / Docker  
**Fecha de generación:** 2026-11-20 05:58 UTC  
**Fuente:** `reports/profiling.json` de la sesión 6.

## 1. Objetivo
Estimar la energía y la huella de carbono asociadas al tiempo de ejecución, comparar ONNX con el modelo original y proponer dos mejoras de sostenibilidad.

## 2. Datos y supuestos
- Operaciones analizadas por método: **100**.
- Potencia eléctrica atribuida a la carga: **50 W (hipótesis, NO medida)**.
- Factor de emisión: **0.4 kg CO₂e/kWh (hipótesis ilustrativa, NO factor oficial verificado)**.
- Latencias: medias registradas en el profiling de la sesión 6.
- Memoria Docker: **69.877 MB** al inicio y **69.982 MB** al final. Dos observaciones puntuales, no el máximo de RAM.
- La API se midió por HTTP en Docker; los dos modelos se midieron directamente en Windows. Las condiciones no son idénticas.

## 3. Metodología y ecuaciones
Para 100 operaciones: `tiempo_s = latencia_media_ms × 100 / 1000`.

`Energía (Wh) = potencia (W) × tiempo (s) / 3600`.

`Energía (kWh) = energía (Wh) / 1000`.

`Emisiones (g CO₂e) = energía (kWh) × factor (kg CO₂e/kWh) × 1000`.

El cálculo es un **modelo de potencia constante**: no mide el consumo real ni separa la energía en reposo del equipo. No incluye arranque, entrenamiento, espera ni costes de infraestructura.

## 4. Resultados estimados

| Método | Latencia media (ms) | Tiempo 100 ops (s) | Energía (Wh) | Emisiones (g CO₂e) |
|---|---:|---:|---:|---:|
| API Docker | 48.0927 | 4.809270 | 0.066795417 | 0.026718167 |
| ONNX directo | 0.0486 | 0.004860 | 0.000067500 | 0.000027000 |
| Random Forest original | 37.6765 | 3.767650 | 0.052328472 | 0.020931389 |

## 5. Comparación ONNX frente al original
La latencia media del modelo original fue **37.6765 ms** y la de ONNX directo **0.0486 ms**. La razón de tiempos es **775.24×**. Bajo la hipótesis de igual potencia constante, la energía calculada para ONNX resulta **99.8710% menor** que para el original.

**Limitación:** esta diferencia no demuestra una reducción energética real equivalente, porque no se midió potencia eléctrica y el profiling se ejecutó en condiciones distintas. La API Docker incluye comunicación HTTP y validación; no se debe interpretar su latencia como tiempo exclusivo de inferencia.

## 6. Dos mejoras viables
1. **Usar ONNX Runtime para la inferencia de producción**, después de validar equivalencia de predicciones sobre un conjunto representativo. Comparar el consumo y la latencia en el mismo entorno, con el mismo número de hilos y el mismo lote de datos.
2. **Reducir el trabajo y los recursos ociosos de la API**: mantener el modelo cargado una sola vez, evitar dependencias innecesarias en la imagen Docker y ajustar el número de workers e hilos según la carga. Evaluar la RAM y el consumo total incluyendo el tiempo en reposo.

## 7. Limitaciones y trabajo futuro
- Medir potencia real con instrumentación o telemetría apropiada y registrar consumo en reposo.
- Usar un factor de emisión verificable, fechado y específico del lugar o del proveedor eléctrico.
- Repetir ensayos, registrar variabilidad y probar múltiples muestras del dataset.
- Validar rangos de sensores transformados en la API con criterios documentados.
- No extrapolar las cifras de este informe a emisiones anuales sin un perfil real de utilización.

## 8. Conclusión
El modelo ONNX presentó menor latencia de inferencia en la prueba realizada. Bajo un escenario hipotético de potencia constante, esto se traduce en una menor energía y huella estimadas para las 100 operaciones. El resultado es orientativo y requiere medición eléctrica para cuantificar el ahorro ambiental real.
