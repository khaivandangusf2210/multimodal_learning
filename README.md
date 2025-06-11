# Multimodal Head and Neck Cancer Survival Prediction

## Overview
This project implements a multimodal machine learning pipeline for predicting head and neck cancer patient survival using AutoGluon. The model combines multiple data modalities to achieve clinically relevant performance in survival prediction.

## Project Goals & Outputs

### Primary Goal
Develop an **AutoGluon MultiModal AI model** that predicts **head and neck cancer patient survival status** (deceased vs. living) by integrating multiple types of medical data.

## Dataset - HANCOCK Head and Neck Cancer Dataset

This project uses the **HANCOCK dataset** from [hancock.research.fau.edu](# Multimodal Head and Neck Cancer Survival Prediction

## Overview
This project implements a multimodal machine learning pipeline for predicting head and neck cancer patient survival using AutoGluon. The model combines multiple data modalities to achieve clinically relevant performance in survival prediction.

## Project Goals & Outputs

### Primary Goal
Develop an **AutoGluon MultiModal AI model** that predicts **head and neck cancer patient survival status** (deceased vs. living) by integrating multiple types of medical data.

## Dataset - HANCOCK Head and Neck Cancer Dataset

This project uses the **HANCOCK dataset** from [hancock.research.fau.edu](https://hancock.research.fau.eu/download). 

**Citation**: Dörrich et al, medrxiv 2024 (currently available as preprint)  
**License**: CC BY

### HANCOCK Dataset Components Used

From the full HANCOCK dataset, this project specifically uses:

1. ** Structured Data** (JSON | 4 files | 7 MB)
   - `clinical_data.json` - Patient demographics and treatment history
   - `pathological_data.json` - Tumor staging and molecular markers
   - `blood_data.json` - Laboratory measurements
   - `blood_data_reference_ranges.json` - Reference ranges

2. ** Text Data** (TXT | 5514 files | 100 MB) 
   - German surgery reports
   - English translations of surgery reports

3. ** TMA Cell Density Measurements** (CSV | 1 file | < 1 MB)
   - Tissue microarray cell density quantification
   - 6,332 individual measurements across patients

4. ** Primary Tumor Annotations** (GeoJSON | 709 files | 45 MB)
   - Geometric annotations of primary tumors from WSI
   - Used for extracting tumor shape and spatial features

**Total Used**: ~152 MB from the HANCOCK dataset  
**Note**: This project uses annotations and structured data, not the full slide images (SVS files)

### Dataset Statistics
- **763 patients** with head and neck cancer
- **5 distinct data modalities** processed and integrated  
- **76 total features** after multimodal fusion

**Citation**: Dörrich et al, medrxiv 2024 (currently available as preprint)  
**License**: CC BY

### HANCOCK Dataset Components Used

From the full HANCOCK dataset, this project specifically uses:

1. ** Structured Data** (JSON | 4 files | 7 MB)
   - `clinical_data.json` - Patient demographics and treatment history
   - `pathological_data.json` - Tumor staging and molecular markers
   - `blood_data.json` - Laboratory measurements
   - `blood_data_reference_ranges.json` - Reference ranges

2. ** Text Data** (TXT | 5514 files | 100 MB) 
   - German surgery reports
   - English translations of surgery reports

3. ** TMA Cell Density Measurements** (CSV | 1 file | < 1 MB)
   - Tissue microarray cell density quantification
   - 6,332 individual measurements across patients

4. ** Primary Tumor Annotations** (GeoJSON | 709 files | 45 MB)
   - Geometric annotations of primary tumors from WSI
   - Used for extracting tumor shape and spatial features

**Total Used**: ~152 MB from the HANCOCK dataset  
**Note**: This project uses annotations and structured data, not the full slide images (SVS files)



