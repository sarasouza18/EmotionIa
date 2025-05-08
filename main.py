import streamlit as st
from analyzer import process_image, compare_emotions
from utils import save_uploaded_file, export_results
import pandas as pd
import os

st.set_page_config(page_title="Emotion AI", layout="centered")

st.title("Detector de Emoções com DeepFace")

if "resultados" not in st.session_state:
    st.session_state["resultados"] = []

uploaded_file = st.file_uploader("Envie uma imagem facial (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = save_uploaded_file(uploaded_file)
    st.image(image_path, caption="Imagem carregada", use_column_width=True)

    rotulo_real = st.selectbox("Qual é a emoção real desta imagem?", ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"])

    if st.button("Analisar Emoção"):
        with st.spinner("Analisando..."):
            result = process_image(image_path)
            result["Rótulo Real"] = rotulo_real
            st.session_state["resultados"].append(result)

            st.success(f"Detectado: {result['dominant_emotion'].upper()} ({result['confidence']}%)")
            st.write("Distribuição emocional:", result['distribution'])

if st.session_state["resultados"]:
    st.header("Resultados Acumulados")
    df = pd.DataFrame(st.session_state["resultados"])
    st.dataframe(df[["imagem", "Rótulo Real", "dominant_emotion", "confidence"]])

    if st.button("Gerar Análise e Relatório"):
        compare_emotions(df)
        export_results(df)
        st.success("Relatório exportado como 'relatorio_emocoes.csv'")
