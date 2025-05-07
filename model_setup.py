#!/usr/bin/env python
"""
Model Setup Script for Tunisian Plate Detector
Downloads and prepares required models for deployment
"""
import os
import sys
import urllib.request
import shutil
import zipfile
import torch
import cv2

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, local_path):
    print(f"Downloading {url} to {local_path}")
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"Downloaded {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    print("Setting up models for Tunisian Plate Detector")
    
    # Download EasyOCR Arabic model
    arabic_model_url = "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/arabic.zip"
    arabic_model_zip = os.path.join(MODEL_DIR, "arabic.zip")
    
    if not os.path.exists(os.path.join(MODEL_DIR, "arabic.pth")):
        if download_file(arabic_model_url, arabic_model_zip):
            print("Extracting Arabic model")
            with zipfile.ZipFile(arabic_model_zip, 'r') as zip_ref:
                zip_ref.extractall(MODEL_DIR)
            os.remove(arabic_model_zip)
            print("Arabic model extracted successfully")
    else:
        print("Arabic model already exists")
        
    # Ensure cascade classifier is available
    cascade_path = os.path.join(MODEL_DIR, "haarcascade_russian_plate_number.xml")
    if not os.path.exists(cascade_path):
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
        download_file(cascade_url, cascade_path)
    else:
        print("Cascade classifier already exists")
    
    # Create a README file in model directory
    with open(os.path.join(MODEL_DIR, "README.txt"), "w") as f:
        f.write("This directory contains models used by the Tunisian Plate Detector.\n")
        f.write("arabic.pth - EasyOCR Arabic recognition model\n")
        f.write("haarcascade_russian_plate_number.xml - OpenCV cascade classifier for license plates\n")
    
    print("Model setup complete!")
    print("Files in model directory:")
    for file in os.listdir(MODEL_DIR):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
