import uvicorn
import io
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from keras.models import load_model
import tensorflow as tf
from PIL import Image
import base64

import os
import base64
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil

app = FastAPI()

# Defina a função de perda personalizada, se necessário
def dice_loss(y_true, y_pred):
    # Implemente a lógica da função de perda aqui
    pass

# Carregue o modelo
with tf.keras.utils.custom_object_scope({'dice_loss': dice_loss}):
    loaded_model = load_model('model/modelo_salvo.h5')

# Montando a pasta 'static' para servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Criando um caminho absoluto para os templates
templates_dir = os.path.join(os.path.dirname(__file__), "static/templates")
templates = Jinja2Templates(directory=templates_dir)

# Rota inicial para exibir o formulário HTML
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Endpoint para processar a imagem e fazer previsões
@app.post("/segmentation")
async def segmentation(file: UploadFile):
    try:
        contents = await file.read()  # Ler o conteúdo do arquivo
        image = Image.open(io.BytesIO(contents))  # Abrir a imagem usando PIL

        # Pré-processamento da imagem (redimensionamento, normalização, etc.)
        # Exemplo: Redimensionamento para 128x128 e normalização entre 0 e 1
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0  # Normalização entre 0 e 1
        image_array = np.expand_dims(image_array, axis=0)  # Adicionar dimensão batch

        # Realizar a previsão com o modelo carregado
        prediction = loaded_model.predict(image_array)

        # Converte a matriz de valores em uma imagem usando a biblioteca PIL
        segmented_image = Image.fromarray((prediction[0, ..., 0] * 255).astype(np.uint8))  # Use apenas um canal

        # Salvando a imagem temporariamente
        temp_path = "model/segmented_image.png"
        segmented_image.save(temp_path)

        # Retorna a imagem segmentada como resposta da API
        with open(temp_path, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')

        return {"segmented_image": encoded_image}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
