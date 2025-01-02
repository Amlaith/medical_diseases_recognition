import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header
from fastapi import UploadFile
from data_models_blueprint import *


def process_images(X) -> pd.DataFrame:
    """Turn images into a dataset"""

    def turn_str_to_image():
        """Turn string to image"""

        return None

    images_matrix = np.array([])

    for curr_img in X:
        # превратить строку в изображение
        # сжать изображени | сделать нужный размер
        # сложить изображение в матричку
        pass

    images_matrix = pd.DataFrame(images_matrix)
    return images_matrix


def gen_id():
    """Generate id"""
    id = "fnjk24r4"
    return id


app = FastAPI(title="Checkpoint 4")


@app.get("/", tags=["Root"])
def root() -> RootResponse:
    """Root of the app"""
    return {"status": "App is running"}


@app.post("/upload/", tags=["Upload new data"])
def upload_data(file: UploadFile = Header()):
    """Upload data for later use"""

    # Логиченее архивчиком принимать

    X = ...
    data = process_images(X)

    info = "bla-bla-bla"
    id = gen_id()  # id генерируем сами
    return f"Данные загружены {id} {info}"


@app.get("/predict/", tags=["Make predictions"])
def predict():
    """Return prediction on X"""
    pred = "dsdsf"
    return f"{pred}"


@app.post("/upfit/{model_params}", tags=["Upfit"])
def upfit(model_params) -> ModelFitResponse:
    """Дообучение"""
    id = gen_id()  # id генерируем сами
    return {"info": f"Мы дообучили, держи. {id}"}


@app.delete(path="/delete/{model_id}", tags=["Deletion"])
def delete_model():
    """Delete trained model"""
    pass


@app.delete(path="/delete/{data_id}", tags=["Deletion"])
def delete_data():
    """Delete data (images)"""
    pass


@app.delete(path="/delete_all/", tags=["Deletion"])
def delete_all():
    """Delete all history and all data"""
    pass


if __name__ == "__main__":
    uvicorn.run("fastapi_app_blueprint:app", host="0.0.0.0", port=8000, reload=True)
