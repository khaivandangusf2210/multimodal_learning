# Multimodal Head and Neck Cancer Survival Prediction

## Overview
This project implements a multimodal machine learning pipeline for predicting head and neck cancer patient survival using AutoGluon. The model combines multiple data modalities to achieve clinically relevant performance in survival prediction.

## Project Goals & Outputs

### Primary Goal
Develop an **AutoGluon MultiModal AI model** that predicts **head and neck cancer patient survival status** (deceased vs. living) by integrating multiple types of medical data.

## Dataset
- **763 patients** with head and neck cancer
- **5 distinct data modalities** processed and integrated
- **76 total features** after multimodal fusion

## Data Modalities

### 1. Clinical Data (763 patients, 32 features)
- Demographics, survival outcomes, treatment history
- Patient baseline characteristics

### 2. Pathological Data (763 patients, 18 features)  
- Tumor staging (TNM classification)
- Histological grading
- HPV status and molecular markers

### 3. Text Data (742/763 patients)
- German surgery reports (~1928 characters average)
- English translations for cross-lingual analysis
- Processed using transformer/BERT encoders

### 4. TMA Cell Density Measurements (763 patients, 9 aggregated features)
- **6,332 raw measurements** aggregated per patient
- Tissue microarray cell density quantification
- Immune cell infiltration patterns

### 5. WSI Geometric Features (300/763 patients, 13 features)
- Whole slide image tumor annotations (39% coverage)
- Geometric shape analysis of primary tumors
- Spatial tumor characteristics

## Data Availability
All data has been converted to CSV format and is available in the `data_csv/` folder for easy analysis and model training.

