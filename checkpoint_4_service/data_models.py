from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from typing import List, Dict, Optional, Any
from fastapi import HTTPException
import numpy as np
import pandas as pd

class RootResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App is running"}]}
    )

class TrainResults(BaseModel):
    model: Any


class MLModelStorage(BaseModel):
    models: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
    
    def add_model(self, model_name: str, model_instance: Any):
        """Adds a model to the storage."""
        if model_name in self.models:
            raise ValueError(f"Model '{model_name}' already exists.")
        self.models[model_name] = model_instance

    def get_model(self, model_name: str) -> Any:
        """Retrieves a model from the storage."""
        model = self.models.get(model_name)
        if model is None:
            raise KeyError(f"Model '{model_name}' not found.")
        return model

    def list_models(self) -> List[str]:
        """Lists all stored models."""
        return list(self.models.keys())
    

class DatasetStorage(BaseModel):
    datasets: Dict[str, Dict[str, np.ndarray]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def add_dataset(self, name: str, images: np.ndarray, labels: np.ndarray):
        """Adds a dataset to the storage."""
        if name in self.datasets:
            raise ValueError(f"Dataset '{name}' already exists.")
        self.datasets[name] = {"images": images, "labels": labels}

    def get_dataset(self, name: str) -> Dict[str, np.ndarray]:
        """Retrieves a dataset from the storage."""
        dataset = self.datasets.get(name)
        if dataset is None:
            raise KeyError(f"Dataset '{name}' not found.")
        return dataset

    def list_datasets(self) -> List[str]:
        """Lists all stored datasets."""
        return list(self.datasets.keys())
    
# def create_dataset_enum(dataset_storage: DatasetStorage) -> Enum:
#     """Dynamically generate an Enum for dataset names."""
#     datasets = dataset_storage.list_datasets()
#     # if not datasets:
#     #     raise HTTPException(status_code=400, detail="No datasets available.")
#     return Enum("DatasetNames", {name: name for name in datasets})