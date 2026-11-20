"""Práctica 3, sesión 7: estimación Green AI basada en profiling.json.

Ejecutar desde la raíz: python -m src.p3_07_green_ai
No mide potencia eléctrica; documenta supuestos editables.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / 'reports'
INPUT = REPORTS / 'profiling.json'
OUTPUT = REPORTS / 'green_ai.md'

# ESCENARIO ILUSTRATIVO: sustituir por valores fundamentados si se dispone de ellos.
POTENCIA_W = 50.0
FACTOR_KG_CO2E_KWH = 0.4
NUM_OPERACIONES = 100

KEYS = {
    'API Docker': 'api_docker',
    'ONNX directo': 'onnx_directo',
    'Random Forest original': 'random_forest_original',
}

def calcular(latencia_ms):
    segundos = latencia_ms * NUM_OPERACIONES / 1000.0
    wh = POTENCIA_W * segundos / 3600.0
    kwh = wh / 1000.0
    gramos_co2e = kwh * FACTOR_KG_CO2E_KWH * 1000.0
    return segundos, wh, kwh, gramos_co2e

def main():
    if POTENCIA_W <= 0 or FACTOR_KG_CO2E_KWH < 0:
        raise ValueError('La potencia debe ser positiva y el factor no negativo.')
    if not INPUT.is_file():
        raise FileNotFoundError(f'Falta {INPUT}. Ejecuta primero la sesión 6.')
    datos = json.loads(INPUT.read_text(encoding='utf-8'))
    latencias = datos.get('latencia', {})
    filas = []
    calculos = {}
    for nombre, clave in KEYS.items():
        bloque = latencias.get(clave)
        if not isinstance(bloque, dict) or 'promedio_ms' not in bloque:
            raise ValueError(f'Falta latencia.{clave}.promedio_ms en profiling.json')
        lat = float(bloque['promedio_ms'])
        if not 0 < lat < float('inf'):
            raise ValueError(f'Latencia inválida para {nombre}: {lat}')
        seg, wh, kwh, g = calcular(lat)
        calculos[clave] = {'latencia_ms': lat, 'tiempo_s': seg, 'energia_wh': wh, 'carbono_g': g}
        filas.append(f'| {nombre} | {lat:.4f} | {seg:.6f} | {wh:.9f} | {g:.9f} |')

    original = calculos['random_forest_original']
    onnx = calculos['onnx_directo']
    factor_tiempo = original['latencia_ms'] / onnx['latencia_ms']
    reduccion = (1 - onnx['energia_wh'] / original['energia_wh']) * 100
    ram = datos.get('ram_contenedor', {})
    antes = ram.get('antes', {}).get('ram_mb', 'no disponible')
    despues = ram.get('despues', {}).get('ram_mb', 'no disponible')
    fecha = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    contenido = f'''# Práctica 3 — Sesión 7: Green AI

**Proyecto:** Robot Wall-Following — Random Forest / ONNX / FastAPI / Docker  
**Fecha de generación:** {fecha}  
**Fuente:** `reports/profiling.json` de la sesión 6.

## 1. Objetivo
Estimar la energía y la huella de carbono asociadas al tiempo de ejecución, comparar ONNX con el modelo original y proponer dos mejoras de sostenibilidad.

## 2. Datos y supuestos
- Operaciones analizadas por método: **{NUM_OPERACIONES}**.
- Potencia eléctrica atribuida a la carga: **{POTENCIA_W:g} W (hipótesis, NO medida)**.
- Factor de emisión: **{FACTOR_KG_CO2E_KWH:g} kg CO₂e/kWh (hipótesis ilustrativa, NO factor oficial verificado)**.
- Latencias: medias registradas en el profiling de la sesión 6.
- Memoria Docker: **{antes} MB** al inicio y **{despues} MB** al final. Dos observaciones puntuales, no el máximo de RAM.
- La API se midió por HTTP en Docker; los dos modelos se midieron directamente en Windows. Las condiciones no son idénticas.

## 3. Metodología y ecuaciones
Para {NUM_OPERACIONES} operaciones: `tiempo_s = latencia_media_ms × {NUM_OPERACIONES} / 1000`.

`Energía (Wh) = potencia (W) × tiempo (s) / 3600`.

`Energía (kWh) = energía (Wh) / 1000`.

`Emisiones (g CO₂e) = energía (kWh) × factor (kg CO₂e/kWh) × 1000`.

El cálculo es un **modelo de potencia constante**: no mide el consumo real ni separa la energía en reposo del equipo. No incluye arranque, entrenamiento, espera ni costes de infraestructura.

## 4. Resultados estimados

| Método | Latencia media (ms) | Tiempo 100 ops (s) | Energía (Wh) | Emisiones (g CO₂e) |
|---|---:|---:|---:|---:|
{chr(10).join(filas)}

## 5. Comparación ONNX frente al original
La latencia media del modelo original fue **{original['latencia_ms']:.4f} ms** y la de ONNX directo **{onnx['latencia_ms']:.4f} ms**. La razón de tiempos es **{factor_tiempo:.2f}×**. Bajo la hipótesis de igual potencia constante, la energía calculada para ONNX resulta **{reduccion:.4f}% menor** que para el original.

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
'''
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(contenido, encoding='utf-8')
    print('=== PRACTICA 3 - SESION 7: GREEN AI ===')
    for nombre, clave in KEYS.items():
        x = calculos[clave]
        print(f"{nombre}: {x['tiempo_s']:.6f} s | {x['energia_wh']:.9f} Wh | {x['carbono_g']:.9f} g CO2e")
    print(f'Informe generado: {OUTPUT}')
    print('IMPORTANTE: potencia y factor de emisión son supuestos, no mediciones.')

if __name__ == '__main__':
    main()
