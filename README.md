
# El Predictor — Aprendizaje supervisado

## Descripción
El Predictor es un proyecto educativo de machine learning que estima
si un alumno aprueba o reprueba según sus horas de estudio, asistencia
y porcentaje de tareas entregadas.

## Tecnologías
- Python
- Scikit-learn
- Streamlit
- NumPy
- Pandas

## Archivos
- `modelo.py`: genera los datos, entrena el modelo y calcula las métricas.
- `app.py`: interfaz visual para ingresar datos y mostrar predicciones.
- `requirements.txt`: librerías necesarias.
- `capturas/`: imágenes del sistema funcionando.

## Instalación
Crear y activar un entorno virtual, y ejecutar:

```bash
python -m pip install -r requirements.txt
```

## Ejecución
Desde la carpeta del proyecto:

```bash
streamlit run app.py
```

## Evaluación
Se utiliza un 80% de los ejemplos para entrenamiento y un 20% para prueba.
Se reportan la exactitud y la matriz de confusión sobre datos no usados
durante el entrenamiento.

## Datos
Los ejemplos y las etiquetas son simulados con fines educativos.
El resultado no garantiza la calificación real de un estudiante.

## Integrantes
Escribir aquí los nombres de los integrantes del equipo.
