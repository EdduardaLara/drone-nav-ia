from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

modelo = load_model("modelo_Coffee_melhorado.h5")

classes = ['Cercospora', 'Healthy', 'Leaf Rust', 'Miner', 'Phoma']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr, verbose=0)[0]
    idx = int(np.argmax(pred))

    return {
        "classe_predita": classes[idx],
        "confiança": f"{pred[idx]*100:.2f}%",
        "raw": pred.tolist()
    }
