from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional, Any


class ImageSchema(BaseModel):
    images: Any
    classes: Optional[str]


class ModelConfigSchema(BaseModel):
    ml_model_type: str = Field(choices=['SVM', 'DecisionTree'])
    hyperparameters: Optional[Dict[str, Any]] = None


class RootResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App is running"}]}
    )


class ModelFitResponse(BaseModel):
    info: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"info": "Model_id: wehhf3. Model_info: SVM(kernel='rbf', tol=1e-3)"}]}
    )
