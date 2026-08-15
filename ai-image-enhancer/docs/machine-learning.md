# Machine Learning

This document describes the **planned** machine learning approach for the AI Image Quality Enhancer. No model has been implemented or trained yet.

## 1. What Is Image Super-Resolution?

Image super-resolution is the process of increasing the resolution and visual quality of a low-resolution image to produce a higher-resolution version. The goal is not simply to make the image larger, but to recover fine details — such as edges, textures, and patterns — that were lost or unclear in the original low-quality image.

## 2. Why Deep Learning for Image Enhancement?

Traditional image processing methods (such as bicubic interpolation) can enlarge an image, but they often produce blurry or smoothed results because they do not truly recover lost detail. Deep learning models, on the other hand, can learn complex patterns from large datasets of image pairs. By training on many examples of low-resolution and high-resolution image pairs, a neural network can learn to predict what the missing detail should look like, producing sharper and more realistic results.

## 3. What Is SRCNN?

SRCNN (Super-Resolution Convolutional Neural Network) is one of the earliest deep learning models designed specifically for image super-resolution. It uses a simple convolutional neural network with three layers:

1. **Patch extraction and representation** — extracts features from the low-resolution input.
2. **Non-linear mapping** — maps those features to a higher-dimensional representation.
3. **Reconstruction** — reconstructs the high-resolution output from the mapped features.

SRCNN is a good starting point for a university project because it is well-documented, relatively simple to understand, and has been widely used as a baseline in super-resolution research.

## 4. What Will the Input to SRCNN Be?

The input to the SRCNN model will be a low-resolution image. During training, this will be created by artificially degrading a high-resolution image (for example, by downsampling and adding blur). During inference (when a user uploads an image), the input will be the user's uploaded low-quality image, preprocessed into the correct size and format expected by the model.

## 5. What Will the Target Output Be?

The target output will be a high-resolution version of the input image with improved clarity and detail. During training, the target is the original high-resolution image from which the low-resolution input was created. During inference, the model will predict the enhanced version based on what it learned during training.

## 6. How Will Training Images Be Prepared?

Training images will be collected from a dataset of high-resolution photographs. Each high-resolution image will be processed to create a corresponding low-resolution version. This produces a **training pair**: a low-resolution input and a high-resolution target. The model will learn by comparing its predicted output against the high-resolution target and adjusting its internal parameters to reduce the difference.

## 7. Why Can a High-Resolution Image Be Degraded to Create a Training Pair?

In real-world super-resolution, we rarely have both a low-resolution image and its true high-resolution original. However, if we start with a high-resolution image and artificially reduce its quality (by downsampling, compressing, or blurring), we know exactly what the high-resolution version should look like — it is the original image. This allows us to create reliable training pairs without needing manually captured low/high-resolution pairs, which are difficult to obtain in practice.

## 8. What Is a Loss Function?

A loss function is a mathematical measure of how far the model's predicted output is from the expected target. During training, the model makes a prediction, the loss function calculates the error, and the model adjusts its weights to reduce that error in the next iteration. For image super-resolution, a common loss function is Mean Squared Error (MSE), which measures the average squared difference between predicted and target pixel values. A lower loss means the model's predictions are closer to the target.

## 9. What Is PSNR?

PSNR (Peak Signal-to-Noise Ratio) is a metric used to measure the quality of a reconstructed image compared to a reference (ground truth) image. It is expressed in decibels (dB), and higher values indicate better quality. PSNR compares pixel-level differences between the enhanced image and the original high-resolution image. While it is widely used, it does not always reflect how humans perceive image quality, which is why it is often used alongside SSIM.

## 10. What Is SSIM?

SSIM (Structural Similarity Index Measure) is another metric for evaluating image quality. Unlike PSNR, which compares pixels directly, SSIM considers structural information such as luminance, contrast, and texture patterns. SSIM values range from 0 to 1, where 1 means the enhanced image is identical to the reference. SSIM is often considered a better indicator of perceived visual quality because it aligns more closely with how the human eye judges similarity between images.

## Summary

This project will use the SRCNN model trained on artificially degraded image pairs to perform single-image super-resolution. Model performance will be evaluated using PSNR and SSIM metrics during testing. All of the above is planned — no model has been built or trained at this stage.
