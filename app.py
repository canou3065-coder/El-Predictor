
import streamlit as st
import pandas as pd

from modelo import (
    predecir_alumno,
    exactitud,
    matriz,
    X_train,
    X_test
)


# Configuración de la página
st.set_page_config(
    page_title="El Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 El Predictor")
st.subheader("Sistema de aprendizaje supervisado")

st.write(
    "Introduce los datos de un alumno para estimar "
    "si aprueba o reprueba."
)


# Sliders para introducir los datos
horas = st.slider(
    "📚 Horas de estudio por semana",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5
)

asistencia = st.slider(
    "📅 Porcentaje de asistencia",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)

tareas = st.slider(
    "📝 Porcentaje de tareas entregadas",
    min_value=0,
    max_value=100,
    value=80,
    step=1
)


# Botón para realizar la predicción
if st.button("🔍 Predecir", use_container_width=True):
    resultado = predecir_alumno(horas, asistencia, tareas)

    st.session_state["ultimo_resultado"] = resultado
    st.session_state["ultimo_alumno"] = {
        "horas": horas,
        "asistencia": asistencia,
        "tareas": tareas
    }


# Mostrar el resultado guardado
if "ultimo_resultado" in st.session_state:
    resultado = st.session_state["ultimo_resultado"]
    alumno = st.session_state["ultimo_alumno"]

    st.divider()
    st.subheader("Resultado de la predicción")

    if resultado["clase"] == 1:
        st.success("✅ El modelo predice: APRUEBA")
    else:
        st.error("❌ El modelo predice: REPRUEBA")

    st.write(
        f"Probabilidad estimada de la clase elegida: "
        f"{resultado['probabilidad']:.1f}%"
    )

    st.caption(
        f"Datos evaluados: {alumno['horas']} horas de estudio, "
        f"{alumno['asistencia']}% de asistencia y "
        f"{alumno['tareas']}% de tareas."
    )


# Mostrar las métricas del modelo
st.divider()
st.header("📊 Evaluación del modelo")

st.metric("Exactitud en datos de prueba", f"{exactitud:.1%}")

st.write(f"Alumnos para entrenamiento: {len(X_train)}")
st.write(f"Alumnos para prueba: {len(X_test)}")

st.caption(
    "La exactitud se calcula sobre datos reservados "
    "que no se utilizaron para entrenar el modelo."
)


# Matriz de confusión
st.subheader("Matriz de confusión")

tabla = pd.DataFrame(
    matriz,
    index=["Real: REPRUEBA", "Real: APRUEBA"],
    columns=["Predijo: REPRUEBA", "Predijo: APRUEBA"]
)

st.dataframe(tabla, use_container_width=True)

st.markdown(
    """
    **Cómo interpretar la matriz**

    - **Arriba a la izquierda:** reprobados identificados correctamente.
    - **Arriba a la derecha:** alumnos que reprobaron, pero fueron
      clasificados como aprobados.
    - **Abajo a la izquierda:** alumnos que aprobaron, pero fueron
      clasificados como reprobados.
    - **Abajo a la derecha:** aprobados identificados correctamente.
    """
)

st.info(
    "Este proyecto utiliza datos simulados con fines educativos. "
    "La probabilidad del modelo no garantiza el resultado real "
    "de un alumno."
)
