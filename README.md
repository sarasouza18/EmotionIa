# EmotionIa

Projeto de visão computacional com DeepFace + Streamlit para detecção e avaliação emocional em imagens faciais.

## Funcionalidades

- Análise de emoções com DeepFace
- Testes em imagens da base FER-2013
- Acurácia e matriz de confusão
- Interface interativa com Streamlit
- Exportação de resultados em CSV

## Como executar localmente

```bash
pip install -r requirements_colab.txt
streamlit run main.py
```

## Deploy na nuvem

Utilize `requirements.txt` (baseado no `requirements_streamlit.txt`) e conecte este repositório ao [Streamlit Cloud](https://share.streamlit.io).

## Estrutura

```
EmotionIa/
├── main.py
├── utils.py
├── emotionia_choose_image_test.py
├── emotionia_test.py
├── emotionia_avaliacao.py
├── requirements_local.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

Desenvolvido pela aluna Sara Paloma de Souza
