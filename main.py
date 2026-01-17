from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ExifTags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import zipfile
import tempfile
import os
import io
import exifread
import json

app = FastAPI()

# Carrega modelo CNN
modelo = load_model("modelo_Coffee_melhorado.h5")
classes = ['Cercospora', 'Healthy', 'Leaf Rust', 'Miner', 'Phoma']


def get_exif_gps(path):
    """Extrai GPS de EXIF se existir"""
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f)

        def convert_to_degress(value):
            d = float(value.values[0].num) / float(value.values[0].den)
            m = float(value.values[1].num) / float(value.values[1].den)
            s = float(value.values[2].num) / float(value.values[2].den)
            return d + (m / 60.0) + (s / 3600.0)

        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat = convert_to_degress(tags['GPS GPSLatitude'])
            lon = convert_to_degress(tags['GPS GPSLongitude'])

            if 'GPS GPSLatitudeRef' in tags and tags['GPS GPSLatitudeRef'].values[0] == 'S':
                lat = -lat
            if 'GPS GPSLongitudeRef' in tags and tags['GPS GPSLongitudeRef'].values[0] == 'W':
                lon = -lon

            return lat, lon
    except:
        pass

    return None, None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = modelo.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))

    return {
        "classe": classes[idx],
        "confiança": f"{pred[idx]*100:.2f}%"
    }


@app.post("/predict-zip")
async def predict_zip(file: UploadFile = File(...)):
    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, file.filename)

    with open(zip_path, "wb") as f:
        f.write(await file.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp)

    csv_data = None
    csv_path = None

    # tenta achar o CSV (opcional)
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(root, f)
                csv_data = pd.read_csv(csv_path)
                break

    results = []

    for root, _, files in os.walk(tmp):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)

                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))
                arr = np.array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = modelo.predict(arr, verbose=0)[0]
                idx = int(np.argmax(pred))

                # GPS via EXIF
                lat_exif, lon_exif = get_exif_gps(img_path)
                lat, lon, alt, time = None, None, None, None
                origem = None

                # GPS via CSV > EXIF
                if csv_data is not None and 'file' in csv_data.columns:
                    row = csv_data[csv_data['file'] == fname]
                    if len(row) > 0:
                        lat = row['lat'].values[0]
                        lon = row['lon'].values[0]
                        alt = row['alt'].values[0] if 'alt' in row else None
                        time = row['time'].values[0] if 'time' in row else None
                        origem = "csv"
                    else:
                        if lat_exif and lon_exif:
                            lat, lon = lat_exif, lon_exif
                            origem = "exif"
                else:
                    if lat_exif and lon_exif:
                        lat, lon = lat_exif, lon_exif
                        origem = "exif"

                results.append({
                    "file": fname,
                    "classe": classes[idx],
                    "confiança": float(pred[idx]*100),
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "time": time,
                    "origem": origem
                })

    df = pd.DataFrame(results)

    # contagem para gráfico pizza
    contagem = df['classe'].value_counts()

    # gera pizza
    fig, ax = plt.subplots()
    contagem.plot.pie(autopct='%1.1f%%', ax=ax)
    plt.title("Distribuição das Classes")
    pizza_path = os.path.join(tmp, "pizza.png")
    plt.savefig(pizza_path)
    plt.close()

    # salva CSV
    csv_out = os.path.join(tmp, "relatorio.csv")
    df.to_csv(csv_out, index=False)

    # salva resumo JSON
    resumo = {
        "total_imagens": len(df),
        "contagem": contagem.to_dict()
    }
    resumo_path = os.path.join(tmp, "resumo.json")
    with open(resumo_path, "w") as f:
        json.dump(resumo, f, indent=4)

    # compacta ZIP para retorno
    output_zip = io.BytesIO()
    with zipfile.ZipFile(output_zip, "w") as z:
        z.write(csv_out, arcname="relatorio.csv")
        z.write(pizza_path, arcname="pizza.png")
        z.write(resumo_path, arcname="resumo.json")
    output_zip.seek(0)

    return StreamingResponse(
        output_zip,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition": f"attachment; filename=resultado.zip"
        }
    )
