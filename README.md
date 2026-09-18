# MediLens AI — Chest X-ray Model

## Overview

MediLens AI is a multimodal medical screening prototype.

This module provides the Chest X-ray AI component using a NORMAL-only anomaly detection approach.

## Dataset

The model was developed using the organizer-provided chest X-ray dataset.

- 678 NORMAL images initially identified
- 1 corrupted NORMAL image excluded
- 677 usable NORMAL images
- 390 PNEUMONIA images in the provided evaluation set

No external dataset was used.

## Model Pipeline

```text
Chest X-ray
    ↓
224 × 224 preprocessing
    ↓
Grayscale → RGB
    ↓
DenseNet121 feature extraction
    ↓
1024-dimensional feature vector
    ↓
NORMAL centroid
    ↓
Euclidean anomaly distance
    ↓
Threshold
    ↓
NORMAL_LIKE / ABNORMAL_PATTERN
