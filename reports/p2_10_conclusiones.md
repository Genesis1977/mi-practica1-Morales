# Practica 2 - Sesion 10

## Objetivo

Comparar un agente entrenado con PPO y un
controlador clasico basado en reglas
en el entorno CartPole-v1.

## Metodologia

Se evaluaron 10 episodios por agente,
utilizando las mismas semillas iniciales.

El agente PPO emplea una politica aprendida.
El controlador clasico usa reglas sobre
el angulo, la velocidad angular, la posicion
y la velocidad del carrito.

## Resultados

| Metrica | PPO | Clasico |
|---|---:|---:|
| Promedio | 500.00 | 500.00 |
| Maximo | 500.00 | 500.00 |
| Minimo | 500.00 | 500.00 |
| Desviacion estandar | 0.00 | 0.00 |

Diferencia promedio PPO - Clasico: 0.00

## Interpretacion

PPO y el controlador clasico obtuvieron la misma recompensa promedio.

## Ventajas y desventajas

### PPO

Ventajas:
- Aprende mediante experiencia y recompensas.
- No requiere programar manualmente cada decision.
- Puede adaptarse a problemas complejos mediante entrenamiento.

Desventajas:
- Requiere tiempo y recursos de entrenamiento.
- Su politica es menos interpretable.
- Su rendimiento depende del entrenamiento y la configuracion.

### Controlador clasico

Ventajas:
- No necesita entrenamiento.
- Sus decisiones son explicitas e interpretables.
- Tiene un costo computacional bajo.

Desventajas:
- Depende de reglas disenadas manualmente.
- Puede requerir ajustes para otras condiciones.
- No aprende automaticamente de la experiencia.

## Conclusion

La comparacion permite determinar cual enfoque
obtuvo mayor recompensa en los episodios evaluados.
Los resultados corresponden al entorno simulado
y no garantizan el mismo rendimiento en otros escenarios.
