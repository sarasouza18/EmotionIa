from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("Uso: python3 test_emotionia.py caminho/para/imagem.jpg")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f" Arquivo não encontrado: {image_path}")
    sys.exit(1)

img = cv2.imread(image_path)
if img is None:
    print(f"Falha ao carregar a imagem: {image_path}")
    sys.exit(1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Imagem carregada")
plt.show()

print("Analisando com DeepFace...")
result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

emotions = result[0]['emotion']
print("Distribuição emocional:")
for emo, score in emotions.items():
    print(f" - {emo.capitalize()}: {score:.2f}%")

plt.figure(figsize=(8, 4))
plt.bar(emotions.keys(), emotions.values(), color='skyblue')
plt.title("Emoções Detectadas")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
