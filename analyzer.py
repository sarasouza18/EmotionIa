from deepface import DeepFace
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def process_image(image_path):
    analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)[0]
    dominant = analysis["dominant_emotion"]
    confidence = round(analysis["emotion"][dominant], 2)
    return {
        "imagem": image_path.split("/")[-1],
        "dominant_emotion": dominant,
        "confidence": confidence,
        "distribution": analysis["emotion"]
    }

def compare_emotions(df):
    y_true = df["Rótulo Real"]
    y_pred = df["dominant_emotion"]
    labels = sorted(list(set(y_true) | set(y_pred)))

    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    acuracia = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(matriz, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Detectado")
    plt.ylabel("Rótulo Real")
    plt.title(f"Matriz de Confusão (Acurácia: {acuracia:.2%})")
    plt.tight_layout()
    plt.savefig("/mnt/data/matriz_confusao.png")
    plt.close()
