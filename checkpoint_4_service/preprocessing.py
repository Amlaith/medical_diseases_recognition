import os
# from pathlib import Path
# import zipfile
# import gdown
import pydicom
import cv2
import numpy as np
import pandas as pd
from skimage.filters import sobel, rank
from skimage.feature import hog
from skimage import exposure
from skimage.morphology import disk
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, fbeta_score, roc_curve, roc_auc_score, PrecisionRecallDisplay
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


PRETRAINED_PCA_PATH = "../checkpoint_3_baseline/pca.pkl"
with open(PRETRAINED_PCA_PATH, 'rb') as pretrained_RF:
    pca = pickle.load(pretrained_RF)


def preprocess(image_array: np.array) -> np.array:
    image_array = image_array / 255.
    features = []
    for img in range(image_array.shape[0]):
        cur_image = image_array[img, :, :]
        img_features = np.array([])

        footprint = disk(30)
        img_features = np.concatenate((img_features, rank.equalize(cur_image, footprint=footprint).reshape(-1)))

        features.append(img_features)

    features_array = np.array(features)
    pca_features = pca.transform(image_array.reshape(image_array.shape[0], -1))

    return np.concatenate((features_array, pca_features), axis=1)

def create_default_dataset(seed: int = 73) -> tuple[np.array, np.array]:
    DICOM_FOLDER = "..\\rsna-pneumonia-dataset\\stage_2_train_images"  # Path to the folder with .dcm files
    LABELS_FILE = "..\\rsna-pneumonia-dataset\\stage_2_train_labels.csv"  # Path to the CSV file with labels
    NUM_IMAGES_PER_CLASS = 200

    # Load labels
    labels = pd.read_csv(LABELS_FILE)

    # Separate by class
    class_1 = labels[labels["Target"] == 1]
    class_0 = labels[labels["Target"] == 0]

    # Randomly sample 200 images from each class
    sampled_class_1 = class_1.sample(NUM_IMAGES_PER_CLASS, random_state=73)
    sampled_class_0 = class_0.sample(NUM_IMAGES_PER_CLASS, random_state=73)
    sampled_data = pd.concat([sampled_class_1, sampled_class_0])

    # Prepare the dataset
    output_labels = []
    output_images = []

    for _, row in sampled_data.iterrows():
        patient_id = row["patientId"]
        target = row["Target"]
        
        dicom_path = os.path.join(DICOM_FOLDER, f"{patient_id}.dcm")
        if not os.path.exists(dicom_path):
            print(f"File {dicom_path} not found. Skipping.")
            continue
        
        # Read .dcm file
        dcm = pydicom.dcmread(dicom_path)
        output_images.append(cv2.resize(dcm.pixel_array, (128, 128)))
    
    output_images = np.array(output_images)
    output_images = preprocess(output_images)
    output_labels = sampled_data['Target']

    return output_images, output_labels

def train(model, dataset):
    X = dataset["images"]
    y = dataset["labels"]
    X_train, X_test, y_train, y_test = train_test_split(dataset["images"], dataset["labels"], test_size=0.33, random_state=74)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = (pred_proba >= 0.5).astype('int')

    # Compute curves
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba, pos_label=1)
    roc_auc = roc_auc_score(y_test, pred_proba)

    # Draw curves to array
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    # Plot the ROC curve 
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc) 
    # Roc curve for tpr = fpr  
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend(loc="lower right")     
    fig.savefig('temp_files\\ROC_Curve.png')  # save the figure to file
    plt.close(fig)  # close the figure window

    display = PrecisionRecallDisplay.from_predictions(
        y_test, pred, name="RandomForest", plot_chance_level=True
    )
    _ = display.ax_.set_title("Precision-Recall curve")

    plt.savefig('temp_files\\PR_Curve.png')  # save the figure to file
    plt.close(fig)  # close the figure window
    
    return model, accuracy_score(y_test, pred), fbeta_score(y_test, pred, beta=2)