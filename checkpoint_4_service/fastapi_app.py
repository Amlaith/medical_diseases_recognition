import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Header, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import List
from zipfile import ZipFile
import shutil
import os
import cv2
from data_models import *
from preprocessing import preprocess, train, create_default_dataset
import pickle
from sklearn.ensemble import RandomForestClassifier


TEMP_DIR = "temp_files"
PRETRAINED_MODEL_PATH = "../checkpoint_3_baseline/RF_model.pkl"
model_storage = MLModelStorage()
dataset_storage = DatasetStorage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    with open(PRETRAINED_MODEL_PATH, 'rb') as pretrained_RF:
        model_storage.add_model('Pretrained RandomForest', pickle.load(pretrained_RF))

    default_images, default_labels = create_default_dataset(seed=73)
    dataset_storage.add_dataset('Default Dataset', default_images, default_labels)
    
    yield

    shutil.rmtree(TEMP_DIR, ignore_errors=True)


app = FastAPI(title="Pneumonia Detection", lifespan=lifespan)

@app.get("/", tags=["Root"])
def root() -> RootResponse:
    """Root of the app"""
    return {"status": "App is running"}

@app.get("/list_models", tags=["Models"])
def list_models() -> Dict[str, List[str]]:
    """List all stored models"""
    return {"models": model_storage.list_models()}

@app.get("/list_datasets", tags=["Datasets"])
def list_datasets() -> Dict[str, List[str]]:
    """List all stored datasets"""
    return {"datasets": dataset_storage.list_datasets()}

@app.post("/predict", tags=["Prediction"])
def predict(file: UploadFile = File(...), model_name: str = "Pretrained RandomForest") -> Dict[str, Any]:
    """Endpoint to predict using a specific model"""
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Check if model exists
        try:
            model = model_storage.get_model(model_name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        # Read and preprocess the image directly from file stream
        file_bytes = np.frombuffer(file.file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        image = cv2.resize(image, (128, 128))
        image_array = np.array([image])  # Create a batch of one
        preprocessed_image = preprocess(image_array)

        # Predict using the model
        predicted_result = model.predict(preprocessed_image)
        params = model.get_params() if hasattr(model, "get_params") else "Parameters not available"

        return {
            "model_name": model_name,
            "parameters": params,
            "result": predicted_result.tolist()  # Ensure it's JSON serializable
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/upload", tags=["Datasets"])
def upload_dataset(file: UploadFile = File(...), dataset_name: str = "My Dataset") -> Dict[str, str]:
    """Endpoint to upload a zip dataset and store it in dataset storage"""
    try:
        if file.content_type not in ["application/zip", "application/x-zip-compressed"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only zip files are allowed.")

        if not dataset_name:
            raise HTTPException(status_code=400, detail="Dataset name is required.")

        zip_path = os.path.join(TEMP_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract zip file
        extract_dir = os.path.join(TEMP_DIR, dataset_name)
        os.makedirs(extract_dir, exist_ok=True)
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Process images and labels
        image_files = [f for f in os.listdir(extract_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        labels_path = os.path.join(extract_dir, "labels.csv")

        if not image_files or not os.path.exists(labels_path):
            raise HTTPException(status_code=400, detail="Zip file must contain images and a labels.csv file.")

        # Load labels
        labels_df = pd.read_csv(labels_path)
        images = []
        labels = []

        for _, row in labels_df.iterrows():
            patient_id = row["patientId"]
            label = row["Target"]
            image_path = os.path.join(extract_dir, f"{patient_id}.png")

            if os.path.exists(image_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128, 128))
                images.append(image)
                labels.append(label)

        images = np.array(images)
        images = preprocess(images)
        labels = np.array(labels)

        # Store dataset
        dataset_storage.add_dataset(dataset_name, images, labels)

        # Cleanup temporary files
        shutil.rmtree(extract_dir)
        os.remove(zip_path)

        return {"message": f"Dataset '{dataset_name}' added successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/add_model", tags=["Models"])
def add_model(
        model_name: str = "My Model", 
        n_estimators: int = 100, 
        max_depth: int = None, 
        random_state: int = 74,
        dataset_name: str = 'Default Dataset'
    ) -> Dict[str, str]:
    """Endpoint to add a RandomForest model to the model storage"""
    try:
        # Create RandomForest model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        model, acc_score, f2_score = train(model, dataset_storage.get_dataset(dataset_name))
    
        # Add the model to storage
        model_storage.add_model(model_name, model)
    
        return {"message": f"Model '{model_name}' trained and added successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="localhost", port=8012, reload=True)
