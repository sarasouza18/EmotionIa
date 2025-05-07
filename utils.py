import tempfile
import os

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def export_results(df):
    df.to_csv("/mnt/data/relatorio_emocoes.csv", index=False)
