import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import save_uploaded_file, export_results
import os

st.set_page_config(page_title="Emotion AI", layout="centered")
st.title("Detector de Emoções com DeepFace")

if "resultados" not in st.session_state:
    st.session_state["resultados"] = []

uploaded_file = st.file_uploader("Envie uma imagem facial (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Salvar imagem no diretório temporário
    image_path = save_uploaded_file(uploaded_file)

    # Mostrar imagem
    st.image(image_path, caption="Imagem carregada", use_column_width=True)

    # Escolher rótulo real para comparação
    rotulo_real = st.selectbox("Qual é a emoção real desta imagem?", 
                               ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])

    if st.button("Analisar Emoção"):
        with st.spinner("Analisando..."):
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            result["Rótulo Real"] = rotulo_real
            st.session_state["resultados"].append(result)

        st.success(f"Detectado: {result[0]['dominant_emotion'].upper()} ({result[0]['emotion'][result[0]['dominant_emotion']]:.2f}%)")
        st.write("Distribuição emocional:", result[0]["emotion"])

        # Exibir gráfico
        fig, ax = plt.subplots()
        ax.bar(result[0]["emotion"].keys(), result[0]["emotion"].values(), color='skyblue')
        plt.xticks(rotation=45)
        st.pyplot(fig)

if st.session_state["resultados"]:
    st.write("📊 Histórico de análises:")
    st.write(st.session_state["resultados"])

    if st.button("Exportar resultados para CSV"):
        export_results(st.session_state["resultados"])
        st.success("Arquivo 'emotionia_streamlit_resultados.csv' exportado com sucesso!")
