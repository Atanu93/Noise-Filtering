# Noise Filtering & Image Restoration

A Python-based image restoration project for detecting,
modeling, and reducing common image noise using classical
spatial and adaptive filtering techniques.

## Project Objective

Images can be degraded by different types of noise...

## Noise Models

### Gaussian Noise

I_n(x,y) = I(x,y) + N(x,y)

### Salt-and-Pepper Noise

I_n(x,y) =
    0     → pepper
    255   → salt
    I(x,y) → otherwise

### Speckle Noise

I_n(x,y) = I(x,y)(1 + N(x,y))

## Filtering Techniques

### Mean Filter
### Median Filter
### Gaussian Filter
### Adaptive Gaussian Filter

## Automatic Noise Detection

Detect → Classify → Restore → Evaluate

## Pixel-Level Adaptive Restoration

Salt & Pepper → Median
Speckle       → Homomorphic restoration
Gaussian      → Non-Local Means
Clean         → Original pixel

## Quality Metrics

- PSNR
- SSIM
- IEF

## Project Structure

Noise-Filtering/
├── pixel.py
├── hybrid.py
├── Newfilter.py
├── filter.py
├── Gaussian Filter.py
├── adavtive gaussian filter.py
└── noisy_image.py

## Installation

```bash
pip install opencv-python numpy matplotlib scikit-image