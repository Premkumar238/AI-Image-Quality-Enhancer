# Architecture

This document describes the **planned** system architecture for the AI Image Quality Enhancer. None of the components described below have been implemented yet.

## System Flow

```
User
  ↓
React Frontend
  ↓
FastAPI Backend
  ↓
Image Preprocessing
  ↓
PyTorch SRCNN Model
  ↓
Image Postprocessing
  ↓
Enhanced Image
  ↓
Frontend
  ↓
User Download
```

## Layer Responsibilities

### User
The user interacts with the system through a web browser. They select a low-quality image to upload and later receive the enhanced result for preview and download.

### React Frontend
The frontend provides the user interface. It will handle image upload, display a preview of the original image, show the enhanced result, and allow a before/after comparison. It will also communicate with the backend to send images and receive processed results.

### FastAPI Backend
The backend acts as the bridge between the frontend and the machine learning pipeline. It will receive uploaded images from the frontend, pass them to the preprocessing and model stages, and return the enhanced image along with metadata such as resolution and processing time.

### Image Preprocessing
Before an image can be processed by the model, it must be prepared in the correct format. Preprocessing will include tasks such as resizing, normalising pixel values, and converting the image into a tensor that the PyTorch model can accept.

### PyTorch SRCNN Model
This is the core machine learning component. The SRCNN (Super-Resolution Convolutional Neural Network) model will take the preprocessed low-resolution image as input and predict a higher-resolution version with improved detail and clarity.

### Image Postprocessing
After the model produces its output, postprocessing will convert the result back into a standard image format. This may include denormalising pixel values, clipping values to a valid range, and saving the image in a format suitable for display and download.

### Enhanced Image
The final enhanced image is returned to the backend, which forwards it to the frontend. The user can then preview the result, compare it with the original, and download the improved image.

## Summary

Each layer in this architecture has a clear and separate responsibility. The frontend handles user interaction, the backend manages communication and orchestration, and the ML pipeline (preprocessing → model → postprocessing) performs the actual image enhancement. This separation makes the system easier to develop, test, and maintain.
