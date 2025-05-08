import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

base_path = "assets/test"
results = []

print("Iniciando processamento das imagens...")

for emotion in emotion_labels:
    emotion_dir = os.path.join(base_path, emotion)
    if not os.path.isdir(emotion_dir):
        print(f"⚠️ Pasta não encontrada: {emotion_dir}")
        continue

    for img_name in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_name)
        try:
            print(f"Analisando: {img_path}")
            prediction = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            detected = prediction[0]['dominant_emotion']
            results.append({
                'image': img_name,
                'expected': emotion,
                'predicted': detected
            })
        except Exception as e:
            print(f"Erro em {img_name}: {str(e)}")

if not results:
    print("Nenhuma imagem foi analisada com sucesso.")
    exit()

df = pd.DataFrame(results)
df.to_csv("emotionia_results.csv", index=False)
print("Resultados salvos em emotionia_results.csv")

y_true = df['expected'].str.lower()
y_pred = df['predicted'].str.lower()

acc = accuracy_score(y_true, y_pred)
print(f"Acurácia total: {acc:.2%}")

print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=emotion_labels))

cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predito")
plt.ylabel("Esperado")
plt.title("Matriz de Confusão - DeepFace vs FER2013")
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.show()
print("Matriz de confusão salva como matriz_confusao.png")