
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# 1. Crear ejemplos simulados de alumnos
rng = np.random.default_rng(42)
cantidad = 500

horas = rng.uniform(0, 10, cantidad)
asistencia = rng.uniform(0, 100, cantidad)
tareas = rng.uniform(0, 100, cantidad)

# Cada fila representa un alumno.
# Las columnas son horas, asistencia y tareas.
X = np.column_stack((horas, asistencia, tareas))

# 2. Crear las etiquetas conocidas:
# 0 = REPRUEBA, 1 = APRUEBA
puntaje = horas * 4.5 + asistencia * 0.25 + tareas * 0.20
y = (puntaje >= 60).astype(int)


# 3. Separar los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Crear y entrenar el modelo
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

modelo.fit(X_train, y_train)


# 5. Evaluar con datos que no vio durante el entrenamiento
predicciones = modelo.predict(X_test)

exactitud = accuracy_score(y_test, predicciones)

# Filas = clase real; columnas = clase predicha.
# Orden: 0 = REPRUEBA, 1 = APRUEBA.
matriz = confusion_matrix(
    y_test,
    predicciones,
    labels=[0, 1]
)


# 6. Función para predecir un alumno nuevo
def predecir_alumno(horas, asistencia, tareas):
    alumno = np.array([[horas, asistencia, tareas]])

    resultado = int(modelo.predict(alumno)[0])
    probabilidades = modelo.predict_proba(alumno)[0]

    return {
        "resultado": "APRUEBA" if resultado == 1 else "REPRUEBA",
        "clase": resultado,
        "probabilidad": float(probabilidades[resultado]) * 100
    }


# 7. Mostrar resultados en la terminal
if __name__ == "__main__":
    print("EL PREDICTOR")
    print("-" * 35)
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    print(f"Exactitud: {exactitud:.2%}")
    print("\nMatriz de confusión:")
    print(matriz)

    ejemplo = predecir_alumno(8, 80, 80)
    print("\nPredicción de ejemplo:")
    print(ejemplo)
