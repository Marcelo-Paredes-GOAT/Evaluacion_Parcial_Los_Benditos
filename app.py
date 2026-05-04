import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Configuración de la página
st.set_page_config(page_title="Alerta Temprana de Deserción", page_icon="🎓", layout="wide")

# 2. Carga del Modelo y Scaler (Caché para que no recargue cada vez)
@st.cache_resource
def cargar_archivos():
    modelo = load_model('modelo_desercion.keras')
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    modelo, scaler = cargar_archivos()
    archivos_cargados = True
except Exception as e:
    st.error(f"Error al cargar el modelo o scaler: {e}")
    archivos_cargados = False

# 3. Interfaz de Usuario (UI)
st.title("🎓 Sistema de Alerta Temprana de Deserción Estudiantil")
st.markdown("""
Esta herramienta utiliza una Red Neuronal Artificial (MLP) para evaluar el riesgo de deserción de un estudiante
basado en sus factores académicos, comportamentales y socioeconómicos.
""")

st.divider()

if archivos_cargados:
    st.header("Datos del Estudiante")
    
    # Organizando los inputs en 3 columnas para que la página no sea infinitamente larga
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎓 Académicos")
        age = st.number_input("Edad", min_value=15, max_value=60, value=20)
        gpa = st.number_input("Promedio Ponderado (GPA 0-20)", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
        credits_completed = st.number_input("Créditos Aprobados", min_value=0, max_value=250, value=60)
        prior_failures = st.number_input("Cursos reprobados previos", min_value=0, max_value=10, value=0)
        
    with col2:
        st.subheader("📊 Comportamentales")
        attendance_rate = st.slider("Tasa de Asistencia (%)", 0.0, 100.0, 85.0)
        engagement_score = st.slider("Nivel de Compromiso (1-10)", 1, 10, 7)
        motivation_score = st.slider("Nivel de Motivación (1-10)", 1, 10, 8)
        stress_level = st.slider("Nivel de Estrés (1-10)", 1, 10, 5)
        
    with col3:
        st.subheader("🏠 Socioeconómicos")
        socioeconomic_index = st.slider("Índice Socioeconómico (1-5)", 1, 5, 3)
        commute_minutes = st.number_input("Minutos de viaje al campus", min_value=0, max_value=300, value=45)
        internet_access_score = st.slider("Acceso a Internet (1-10)", 1, 10, 8)
        financial_aid_amount = st.number_input("Ayuda Financiera / Beca ($)", min_value=0, max_value=5000, value=500)
        family_support_index = st.slider("Apoyo Familiar (1-10)", 1, 10, 8)
        work_hours_per_week = st.number_input("Horas de trabajo semanal", min_value=0, max_value=60, value=10)

    st.divider()
    
    # 4. Botón de Predicción y Lógica
    centro = st.columns([1, 2, 1])[1] # Centrar el botón
    if centro.button("🔍 Evaluar Riesgo de Deserción", use_container_width=True, type="primary"):
        
        # Recolectar datos con EXACTAMENTE los mismos nombres que el dataset de Colab
        datos_estudiante = pd.DataFrame({
            'age': [age],
            'gpa': [gpa],
            'attendance_rate': [attendance_rate],
            'credits_completed': [credits_completed],
            'socioeconomic_index': [socioeconomic_index],
            'commute_minutes': [commute_minutes],
            'internet_access_score': [internet_access_score],
            'motivation_score': [motivation_score],
            'stress_level': [stress_level],
            'prior_failures': [prior_failures],
            'engagement_score': [engagement_score],
            'financial_aid_amount': [financial_aid_amount],
            'family_support_index': [family_support_index],
            'work_hours_per_week': [work_hours_per_week]
        })
        
        # Mostrar barra de carga para darle realismo al procesamiento
        with st.spinner("Procesando datos a través de la Red Neuronal..."):
            # Escalar y Predecir
            datos_escalados = scaler.transform(datos_estudiante)
            probabilidad = modelo.predict(datos_escalados, verbose=0)[0][0]
            
        # 5. Mostrar Resultados
        st.header("Resultados de la Evaluación")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.metric(label="Probabilidad Matemática", value=f"{probabilidad * 100:.2f}%")
            
        with col_res2:
            if probabilidad >= 0.50:
                st.error("🚨 ALERTA: ESTUDIANTE EN RIESGO DE DESERCIÓN (Clase 1)")
                st.markdown("**Protocolo Recomendado:** Iniciar protocolo de retención inmediatamente. Coordinar con Bienestar Universitario y asignar tutoría.")
            elif probabilidad >= 0.35:
                st.warning("⚠️ PRECAUCIÓN: Riesgo Moderado")
                st.markdown("**Protocolo Recomendado:** Mantener en observación. Sugerir sesiones de orientación.")
            else:
                st.success("✅ SEGURO: Estudiante Estable (Clase 0)")
                st.markdown("**Protocolo Recomendado:** Ninguna acción extraordinaria requerida.")