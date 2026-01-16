from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Carrega o modelo .h5
modelo = tf.keras.models.load_model("modelo_Coffee_melhorado.h5")

# Defina suas classes (ajuste conforme seu dataset)
CLASSES = ["Cercospora", "Phoma", "Leaf Miner", "Leaf Rust", "Healthy"]

def preprocess(img: Image.Image):
    img = img.resize((128, 128))  # ajuste para sua resolução
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # abre a imagem
    img = Image.open(file.file).convert("RGB")
    data = preprocess(img)
    
    # previsao
    pred = modelo.predict(data)
    classe = CLASSES[np.argmax(pred)]
    conf = float(np.max(pred))

    return JSONResponse({
        "classe": classe,
        "confianca": round(conf, 3)
    })
