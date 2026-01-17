@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    from PIL import Image
    import numpy as np

    img = Image.open(file.file).convert("RGB")
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr)[0]

    classes = ['COFFEE']   # se só tem 1 classe no treinamento

    return {
        "raw_prediction": pred.tolist(),
        "classe_predita": classes[int(np.argmax(pred))]
    }
