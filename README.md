# Multimodal Head and Neck Cancer Survival Prediction

## Overview
This project implements a multimodal machine learning pipeline for predicting head and neck cancer patient survival using AutoGluon. The model combines multiple data modalities to achieve clinically relevant performance in survival prediction.

## Project Goals & Outputs

### Primary Goal
Develop an **AutoGluon MultiModal AI model** that predicts **head and neck cancer patient survival status** (deceased vs. living) by integrating multiple types of medical data.

## Dataset - HANCOCK Head and Neck Cancer Dataset

This project uses the **HANCOCK dataset** from [hancock.research.fau.edu](https://hancock.research.fau.eu/download). 

### HANCOCK Dataset Components Used

From the full HANCOCK dataset, this project specifically uses:

1. ** Structured Data** 
   - `clinical_data.json` - Patient demographics and treatment history
   - `pathological_data.json` - Tumor staging and molecular markers
   - `blood_data.json` - Laboratory measurements
   - `blood_data_reference_ranges.json` - Reference ranges

2. ** Text Data** 
   - German surgery reports
   - English translations of surgery reports

3. ** TMA Cell Density Measurements** 
   - Tissue microarray cell density quantification
   - 6,332 individual measurements across patients

4. ** Primary Tumor Annotations** 
   - Geometric annotations of primary tumors from WSI
   - Used for extracting tumor shape and spatial features







