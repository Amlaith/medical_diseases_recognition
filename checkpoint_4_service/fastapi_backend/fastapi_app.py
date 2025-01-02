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
from preprocessing import preprocess, get_scores, create_default_dataset
import pickle
from sklearn.ensemble import RandomForestClassifier


TEMP_DIR = "temp_files"
PRETRAINED_MODEL_PATH = "models/RF_model.pkl"
model_storage = MLModelStorage()
dataset_storage = DatasetStorage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # default_images, default_labels = create_default_dataset(seed=73)
    default_images = pd.read_csv('data/train_images.csv')
    default_labels = np.array(pd.read_csv('data/train_labels.csv')).reshape(-1,)
    dataset_storage.add_dataset('Default Dataset', default_images, default_labels)

    with open(PRETRAINED_MODEL_PATH, 'rb') as pretrained_RF:
        RF_model = pickle.load(pretrained_RF)
        scores = get_scores(RF_model, 'Default Dataset', needs_fit=False)
        model_storage.add_model(
            'Pretrained RandomForest',
            *scores
            )

    
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
    
@app.post("/add_dataset", tags=["Datasets"])
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
        labels = np.array(labels).reshape(-1,)

        # Store dataset
        dataset_storage.add_dataset(dataset_name, images, labels)

        # Cleanup temporary files
        shutil.rmtree(extract_dir)
        os.remove(zip_path)

        return {"message": f"Dataset '{dataset_name}' added successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/demo_dataset", tags=["Datasets"])
def get_demo_dataset() -> FileResponse:
    """Serve a pre-built demo dataset."""
    demo_file_path = "data/demo_dataset.zip"
    try:
        return FileResponse(demo_file_path, media_type="application/zip", filename="demo_dataset.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/demo_picture", tags=["Prediction"])
def get_demo_dataset() -> FileResponse:
    """Serve a random demo picture for prediction."""
    demo_file_path = f"data/demo_images/image{str(np.random.randint(1, 6))}.png"
    try:
        return FileResponse(demo_file_path, media_type="image/png", filename="demo_image.png")
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
        dataset = dataset_storage.get_dataset(dataset_name)
        model, acc_score, f2_score, fpr, tpr, roc_auc  = get_scores(model, dataset)
        # Add the model to storage
        model_storage.add_model(model_name, model, acc_score, f2_score, fpr, tpr, roc_auc)
    
        return {"message": f"Model '{model_name}' trained and added successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/model_scores", tags=["Models"])
def get_model_scores(model_name: str = "Pretrained RandomForest") -> ModelScores:
    """Get model scores from the storage."""
    return model_storage.get_model_scores(model_name)

@app.get("/model_params", tags=["Models"])
def get_model_params(model_name: str = "Pretrained RandomForest") -> Dict[str, Any]:
    """Get model parameters from the storage."""
    try:
        return model_storage.get_model_params(model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="localhost", port=8000, reload=True)
