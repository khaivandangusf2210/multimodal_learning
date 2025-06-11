import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

def create_autogluon_results_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"autogluon_results_{timestamp}"
    
    os.makedirs(results_folder, exist_ok=True)
    
    subfolders = [
        'models',
        'metrics', 
        'predictions',
        'data_splits',
        'visualizations'
    ]
    
    for subfolder in subfolders:
        os.makedirs(os.path.join(results_folder, subfolder), exist_ok=True)
    
    print(f"Created AutoGluon results folder: {results_folder}")
    print(f"Subfolders: {', '.join(subfolders)}")
    
    return results_folder

def setup_colab_environment():
    print("SETTING UP GOOGLE COLAB ENVIRONMENT")
    print("=" * 60)

    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("SUCCESS: Google Drive mounted successfully")
    except ImportError:
        print("WARNING: Not running in Google Colab - skipping drive mount")

    print("\nInstalling AutoGluon MultiModal...")
    import subprocess
    import sys

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "autogluon.multimodal[all]", "--quiet"
        ])
        print("SUCCESS: AutoGluon MultiModal installed")
    except:
        print("WARNING: AutoGluon installation may have issues - continuing...")

    additional_packages = ["shapely", "geopandas", "plotly", "wordcloud", "textstat"]

    for package in additional_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                package, "--quiet"
            ])
            print(f"SUCCESS: {package} installed")
        except:
            print(f"WARNING: {package} installation failed")

def load_multimodal_data(base_path="/content/drive/MyDrive/Multimodal"):
    base_path = Path(base_path)

    print("\nLOADING MULTIMODAL DATA")
    print("=" * 60)

    print("Loading structured data...")
    with open(base_path / 'StructuredData/clinical_data.json') as f:
        clinical_df = pd.DataFrame(json.load(f))

    with open(base_path / 'StructuredData/pathological_data.json') as f:
        pathological_df = pd.DataFrame(json.load(f))

    df = clinical_df.merge(pathological_df, on='patient_id', how='inner')
    print(f"   Base dataset: {len(df)} patients")
    print(f"   Clinical features: {list(clinical_df.columns)}")
    print(f"   Pathological features: {list(pathological_df.columns)}")

    print("Loading text data...")

    def load_text_content(patient_id, text_type):
        templates = {
            'surgery_report': f'TextData/reports/SurgeryReport_{patient_id:03d}.txt',
            'surgery_report_en': f'TextData/reports_english/SurgeryReport_{patient_id:03d}.txt'
        }

        if text_type in templates:
            file_path = base_path / templates[text_type]
            try:
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            content = ' '.join(content.split())
                            content = content[:2000]
                            return content
                return ""
            except:
                return ""
        return ""

    df['surgery_report_text'] = df['patient_id'].apply(
        lambda x: load_text_content(int(x), 'surgery_report')
    )
    df['surgery_report_en_text'] = df['patient_id'].apply(
        lambda x: load_text_content(int(x), 'surgery_report_en')
    )

    de_available = (df['surgery_report_text'] != "").sum()
    en_available = (df['surgery_report_en_text'] != "").sum()
    print(f"   German reports: {de_available}/{len(df)} available")
    print(f"   English reports: {en_available}/{len(df)} available")

    if de_available > 0:
        avg_de_length = df[df['surgery_report_text'] != ""]["surgery_report_text"].str.len().mean()
        print(f"   Average German text length: {avg_de_length:.0f} characters")
    if en_available > 0:
        avg_en_length = df[df['surgery_report_en_text'] != ""]["surgery_report_en_text"].str.len().mean()
        print(f"   Average English text length: {avg_en_length:.0f} characters")

    df['combined_text'] = df.apply(lambda row:
        f"German Report: {row['surgery_report_text']} " +
        f"English Report: {row['surgery_report_en_text']}", axis=1
    )

    print("Loading TMA measurements...")
    tma_file = base_path / 'TMA_CellDensityMeasurements/TMA_celldensity_measurements.csv'
    if tma_file.exists():
        tma_data = pd.read_csv(tma_file)
        tma_data['patient_id'] = tma_data['Case ID'].astype(str)

        tma_agg = tma_data.groupby('patient_id').agg({
            'Num Positive per mm^2': ['mean', 'max', 'std'],
            'Positive %': ['mean', 'max', 'std'],
            'Num Detections': ['sum', 'mean', 'std']
        }).round(4)

        tma_agg.columns = [f'tma_{col[0].lower().replace(" ", "_").replace("%", "pct")}_{col[1]}'
                          for col in tma_agg.columns]
        tma_agg = tma_agg.reset_index().fillna(0)

        df = df.merge(tma_agg, on='patient_id', how='left')
        print(f"   TMA features: {len(tma_agg.columns)-1} added")
        print(f"   TMA feature names: {list(tma_agg.columns)}")

    print("Loading WSI annotations...")

    def extract_wsi_geometric_features(patient_id):
        try:
            from shapely.geometry import Polygon
        except ImportError:
            print("WARNING: Shapely not available, WSI features will be zeros")
            return {
                'wsi_total_area': 0.0, 'wsi_total_perimeter': 0.0, 'wsi_num_regions': 0,
                'wsi_avg_region_area': 0.0, 'wsi_largest_region_area': 0.0,
                'wsi_smallest_region_area': 0.0, 'wsi_area_std': 0.0,
                'wsi_convex_hull_area': 0.0, 'wsi_solidity': 0.0, 'wsi_complexity': 0.0,
                'wsi_compactness': 0.0, 'wsi_eccentricity': 0.0, 'wsi_has_annotation': False
            }

        annotation_file = base_path / f'WSI_PrimaryTumor_Annotations/PrimaryTumor_HE_{patient_id:03d}.geojson'

        features = {
            'wsi_total_area': 0.0, 'wsi_total_perimeter': 0.0, 'wsi_num_regions': 0,
            'wsi_avg_region_area': 0.0, 'wsi_largest_region_area': 0.0,
            'wsi_smallest_region_area': 0.0, 'wsi_area_std': 0.0,
            'wsi_convex_hull_area': 0.0, 'wsi_solidity': 0.0, 'wsi_complexity': 0.0,
            'wsi_compactness': 0.0, 'wsi_eccentricity': 0.0, 'wsi_has_annotation': False
        }

        try:
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    geojson_data = json.load(f)

                polygons = []
                areas = []

                for feature in geojson_data.get('features', []):
                    if feature['geometry']['type'] == 'Polygon':
                        coords = feature['geometry']['coordinates'][0]
                        try:
                            polygon = Polygon(coords)
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                                areas.append(polygon.area)
                        except:
                            continue

                if polygons and areas:
                    total_area = sum(areas)
                    total_perimeter = sum([p.length for p in polygons])

                    features.update({
                        'wsi_total_area': round(total_area, 4),
                        'wsi_total_perimeter': round(total_perimeter, 4),
                        'wsi_num_regions': len(polygons),
                        'wsi_avg_region_area': round(np.mean(areas), 4),
                        'wsi_largest_region_area': round(max(areas), 4),
                        'wsi_smallest_region_area': round(min(areas), 4),
                        'wsi_area_std': round(np.std(areas), 4),
                        'wsi_has_annotation': True
                    })

                    try:
                        from shapely.ops import unary_union
                        union_polygon = unary_union(polygons)
                        convex_hull = union_polygon.convex_hull

                        features['wsi_convex_hull_area'] = round(convex_hull.area, 4)
                        if convex_hull.area > 0:
                            features['wsi_solidity'] = round(total_area / convex_hull.area, 4)

                        if total_area > 0:
                            features['wsi_complexity'] = round((total_perimeter ** 2) / total_area, 4)
                            features['wsi_compactness'] = round(total_area / (total_perimeter ** 2), 4)
                    except:
                        pass

        except Exception:
            pass

        return features

    wsi_features_list = []
    for idx, row in df.iterrows():
        patient_id = int(row['patient_id'])
        wsi_features = extract_wsi_geometric_features(patient_id)
        wsi_features['patient_id'] = str(patient_id)
        wsi_features_list.append(wsi_features)

    wsi_df = pd.DataFrame(wsi_features_list)
    df = df.merge(wsi_df, on='patient_id', how='left')

    wsi_available = df['wsi_has_annotation'].sum()
    print(f"   WSI annotations: {wsi_available}/{len(df)} available")
    print(f"   WSI geometric features: {len([col for col in wsi_df.columns if col != 'patient_id'])} added")
    print(f"   WSI feature names: {list(wsi_df.columns)}")

    print("Feature engineering...")

    df = df.fillna({
        'age_at_initial_diagnosis': df['age_at_initial_diagnosis'].median() if 'age_at_initial_diagnosis' in df.columns else 50,
        'combined_text': "No report available",
        'surgery_report_text': "",
        'surgery_report_en_text': ""
    })

    numerical_cols_to_fix = ['closest_resection_margin_in_cm']
    for col in numerical_cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    tma_cols = [col for col in df.columns if col.startswith('tma_')]
    wsi_cols = [col for col in df.columns if col.startswith('wsi_') and col != 'wsi_has_annotation']

    for col in tma_cols + wsi_cols:
        df[col] = df[col].fillna(0)

    categorical_features = [
        'sex', 'smoking_status', 'primary_tumor_site', 'pT_stage', 'pN_stage',
        'grading', 'hpv_association_p16', 'perinodal_invasion', 'lymphovascular_invasion_L',
        'vascular_invasion_V', 'perineural_invasion_Pn', 'resection_status',
        'resection_status_carcinoma_in_situ', 'carcinoma_in_situ', 'histologic_type',
        'primarily_metastasis', 'first_treatment_intent', 'first_treatment_modality',
        'adjuvant_treatment_intent', 'adjuvant_radiotherapy', 'adjuvant_radiotherapy_modality',
        'adjuvant_systemic_therapy', 'adjuvant_systemic_therapy_modality',
        'adjuvant_radiochemotherapy', 'recurrence', 'progress_1', 'progress_2'
    ]
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')

    df = df.dropna(subset=['survival_status'])

    print(f"   Final dataset: {len(df)} patients, {len(df.columns)} features")
    print(f"   Missing values by column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"     {col}: {count} missing")

    return df

def prepare_multimodal_format(df):
    print("\nPREPARING MULTIMODAL FORMAT")
    print("=" * 60)

    text_columns = ['combined_text', 'surgery_report_text', 'surgery_report_en_text']

    numerical_columns = [
        'age_at_initial_diagnosis',
        'year_of_initial_diagnosis',
        'days_to_last_information',
        'days_to_first_treatment',
        'days_to_recurrence',
        'days_to_progress_1',
        'days_to_progress_2',
        'days_to_metastasis_1',
        'days_to_metastasis_2',
        'days_to_metastasis_3',
        'days_to_metastasis_4',
        'number_of_positive_lymph_nodes',
        'number_of_resected_lymph_nodes',
        'closest_resection_margin_in_cm',
        'infiltration_depth_in_mm'
    ]

    numerical_columns.extend([col for col in df.columns if col.startswith('tma_')])
    numerical_columns.extend([col for col in df.columns if col.startswith('wsi_') and col != 'wsi_has_annotation'])

    categorical_columns = [
        'sex', 'smoking_status', 'primary_tumor_site', 'pT_stage', 'pN_stage',
        'grading', 'hpv_association_p16', 'perinodal_invasion', 'lymphovascular_invasion_L',
        'vascular_invasion_V', 'perineural_invasion_Pn', 'resection_status',
        'resection_status_carcinoma_in_situ', 'carcinoma_in_situ', 'histologic_type',
        'primarily_metastasis', 'first_treatment_intent', 'first_treatment_modality',
        'adjuvant_treatment_intent', 'adjuvant_radiotherapy', 'adjuvant_radiotherapy_modality',
        'adjuvant_systemic_therapy', 'adjuvant_systemic_therapy_modality',
        'adjuvant_radiochemotherapy', 'recurrence', 'progress_1', 'progress_2',
        'wsi_has_annotation'
    ]

    keep_columns = text_columns + numerical_columns + categorical_columns + ['survival_status']
    available_columns = [col for col in keep_columns if col in df.columns]
    df_multimodal = df[available_columns].copy()

    text_features_available = [col for col in text_columns if col in df.columns]
    numerical_features_available = [col for col in numerical_columns if col in df.columns]
    categorical_features_available = [col for col in categorical_columns if col in df.columns]

    print(f"Text features: {len(text_features_available)}")
    print(f"   Names: {text_features_available}")
    print(f"Numerical features: {len(numerical_features_available)}")
    print(f"   Names: {numerical_features_available}")
    print(f"Categorical features: {len(categorical_features_available)}")
    print(f"   Names: {categorical_features_available}")

    print(f"\nData types summary:")
    for col in df_multimodal.columns:
        print(f"   {col}: {df_multimodal[col].dtype}")

    return df_multimodal

def train_autogluon_multimodal(df, target_column='survival_status', results_folder=None):
    print(f"\nTRAINING AUTOGLUON MULTIMODAL MODEL")
    print("=" * 60)
    
    if results_folder is None:
        results_folder = create_autogluon_results_folder()
    
    model_path = os.path.join(results_folder, 'models', 'autogluon_multimodal_model')
    model_path_fallback = os.path.join(results_folder, 'models', 'autogluon_multimodal_model_fallback')
    model_path_final = os.path.join(results_folder, 'models', 'autogluon_multimodal_model_final')

    model_paths = [
        model_path,
        model_path_fallback, 
        model_path_final,
        './autogluon_multimodal_model',
        './autogluon_multimodal_model_fallback',
        './autogluon_multimodal_model_final',
        '/content/autogluon_multimodal_model',
        '/content/autogluon_multimodal_model_fallback',
        '/content/autogluon_multimodal_model_final'
    ]

    for path in model_paths:
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Cleaned up existing model path: {path}")
        except Exception as e:
            print(f"Warning: Could not clean up {path}: {e}")

    try:
        from autogluon.multimodal import MultiModalPredictor
        print("SUCCESS: AutoGluon MultiModal imported successfully")
    except ImportError as e:
        print(f"ERROR: AutoGluon MultiModal import failed: {e}")
        print("SOLUTION: Try: pip install autogluon.multimodal[all]")
        return None, None, None, None

    from sklearn.model_selection import train_test_split

    train_val_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df[target_column] if df[target_column].value_counts().min() > 1 else None
    )

    train_data, val_data = train_test_split(
        train_val_data, test_size=0.125, random_state=42,
        stratify=train_val_data[target_column] if train_val_data[target_column].value_counts().min() > 1 else None
    )

    print(f"Train set: {len(train_data)} patients ({len(train_data)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_data)} patients ({len(val_data)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_data)} patients ({len(test_data)/len(df)*100:.1f}%)")

    print(f"\nTarget distribution across splits:")

    for split_name, split_data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
        print(f"\n{split_name} set:")
        for label, count in split_data[target_column].value_counts().items():
            pct = count / len(split_data) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")

    print(f"\nInitializing MultiModal Predictor...")

    predictor = MultiModalPredictor(
        label=target_column,
        path=model_path,
        problem_type='binary',
        eval_metric='accuracy',
        verbosity=3
    )

    hyperparameters = {
        'optim.max_epochs': 20,
        'optim.lr': 1e-4,
    }

    print(f"Training hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")

    print(f"\nData summary before training:")
    print(f"   Text columns: {len([col for col in train_data.columns if 'text' in col])}")
    print(f"   Numerical columns: {len([col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64']])}")
    print(f"   Categorical columns: {len([col for col in train_data.columns if train_data[col].dtype in ['category', 'object'] and col != target_column])}")
    print(f"   Total features: {len(train_data.columns) - 1}")

    print(f"\nSetting up validation data for training...")

    print(f"\nStarting multimodal training with validation...")
    try:
        predictor.fit(
            train_data=train_data,
            tuning_data=val_data,
            hyperparameters=hyperparameters,
            time_limit=1800,
        )
        print("SUCCESS: Training completed successfully!")

    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        print("FALLBACK: Creating new predictor with minimal hyperparameters...")

        try:
            predictor_fallback = MultiModalPredictor(
                label=target_column,
                path=model_path_fallback,
                problem_type='binary',
                eval_metric='accuracy',
                verbosity=1
            )

            minimal_hyperparameters = {
                "optim.max_epochs": 2
            }
            predictor_fallback.fit(
                train_data=train_data,
                tuning_data=val_data,
                hyperparameters=minimal_hyperparameters,
                time_limit=900,  
            )
            print("SUCCESS: Training completed with minimal settings!")
            predictor = predictor_fallback 

        except Exception as e2:
            print(f"ERROR: Minimal hyperparameters failed: {e2}")
            print("FINAL FALLBACK: Trying with absolutely no hyperparameters...")

            try:
                predictor_final = MultiModalPredictor(
                    label=target_column,
                    path=model_path_final,
                    problem_type='binary',
                    verbosity=1
                )
                predictor_final.fit(
                    train_data=train_data,
                    tuning_data=val_data,
                    time_limit=600,  
                )
                print("SUCCESS: Training completed with default AutoGluon settings!")
                predictor = predictor_final 

            except Exception as e3:
                print(f"ERROR: Training failed completely: {e3}")
                return None, train_data, test_data, None, results_folder

    print(f"\nEvaluating on test set...")
    try:
        test_score = predictor.evaluate(test_data)
        print(f"Test Score: {test_score}")
    except Exception as e:
        print(f"WARNING: Evaluation failed: {e}")
        test_score = None
    
    try:
        train_data.to_csv(os.path.join(results_folder, 'data_splits', 'train_data.csv'), index=False)
        val_data.to_csv(os.path.join(results_folder, 'data_splits', 'val_data.csv'), index=False)
        test_data.to_csv(os.path.join(results_folder, 'data_splits', 'test_data.csv'), index=False)
        print(f"Data splits saved to {results_folder}/data_splits/")
    except Exception as e:
        print(f"WARNING: Could not save data splits: {e}")

    return predictor, train_data, val_data, test_data, test_score, results_folder

def analyze_results(predictor, train_data, val_data, test_data, target_column='survival_status', results_folder=None):
    print(f"\nCOMPREHENSIVE ANALYSIS")
    print("=" * 60)

    if predictor is None:
        print("ERROR: No trained predictor available")
        return {}

    results = {}

    
    print(f"\nData Split Summary:")
    print(f"   Training: {len(train_data)} patients ({len(train_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"   Validation: {len(val_data)} patients ({len(val_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"   Test: {len(test_data)} patients ({len(test_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")

    try:
        print(f"\nPERFORMANCE ON ALL SPLITS:")
        print("=" * 40)

        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, classification_report,
            roc_auc_score, confusion_matrix, f1_score
        )

        for split_name, split_data in [("Training", train_data), ("Validation", val_data), ("Test", test_data)]:
            print(f"\n{split_name} Set Performance:")
            print("-" * 25)

            y_true_split = split_data[target_column]
            y_pred_split = predictor.predict(split_data)

            accuracy_split = accuracy_score(y_true_split, y_pred_split)
            balanced_acc_split = balanced_accuracy_score(y_true_split, y_pred_split)
            f1_macro_split = f1_score(y_true_split, y_pred_split, average='macro')

            print(f"   Accuracy: {accuracy_split:.4f}")
            print(f"   Balanced Accuracy: {balanced_acc_split:.4f}")
            print(f"   F1 Macro: {f1_macro_split:.4f}")

            if split_name == "Training":
                results['train_accuracy'] = round(accuracy_split, 4)
                results['train_balanced_accuracy'] = round(balanced_acc_split, 4)
            elif split_name == "Validation":
                results['val_accuracy'] = round(accuracy_split, 4)
                results['val_balanced_accuracy'] = round(balanced_acc_split, 4)
            elif split_name == "Test":
                results['test_accuracy'] = round(accuracy_split, 4)
                results['test_balanced_accuracy'] = round(balanced_acc_split, 4)
                results['test_f1_macro'] = round(f1_macro_split, 4)

        print(f"\nDETAILED TEST SET ANALYSIS:")
        print("=" * 35)

        y_true = test_data[target_column]
        y_pred = predictor.predict(test_data)

        try:
            y_pred_proba = predictor.predict_proba(test_data)
            if hasattr(y_pred_proba, 'values'):
                y_pred_proba = y_pred_proba.values
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_prob_deceased = y_pred_proba[:, 1]
            else:
                y_prob_deceased = y_pred_proba
        except:
            y_prob_deceased = None

        from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, cohen_kappa_score

        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        micro_precision = precision_score(y_true, y_pred, average='micro')
        micro_recall = recall_score(y_true, y_pred, average='micro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')

        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        weighted_precision = precision_score(y_true, y_pred, average='weighted')
        weighted_recall = recall_score(y_true, y_pred, average='weighted')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        roc_auc_ovr = None
        roc_auc_ovo = None
        loss_value = None

        if y_prob_deceased is not None:
            try:
                y_true_binary = (y_true == 'deceased').astype(int)
                roc_auc = roc_auc_score(y_true_binary, y_prob_deceased)
                roc_auc_ovr = roc_auc 
                roc_auc_ovo = roc_auc
                print(f"ROC AUC: {roc_auc:.4f}")

                from sklearn.metrics import log_loss
                try:
                    y_prob_matrix = np.column_stack([1 - y_prob_deceased, y_prob_deceased])
                    loss_value = log_loss(y_true_binary, y_prob_matrix)
                except Exception as e:
                    print(f"WARNING: Log loss calculation failed: {e}")

            except Exception as e:
                print(f"WARNING: ROC AUC calculation failed: {e}")

        json_results = {
            "model": "autogluon_multimodal",
            "test_metrics": {
                "accuracy": round(accuracy, 4),
                "balanced_accuracy": round(balanced_acc, 4),
                "micro": {
                    "avg_recall": round(micro_recall, 4),
                    "avg_precision": round(micro_precision, 4),
                    "avg_f1": round(micro_f1, 4)
                },
                "macro": {
                    "avg_recall": round(macro_recall, 4),
                    "avg_precision": round(macro_precision, 4),
                    "avg_f1": round(macro_f1, 4)
                },
                "weighted": {
                    "avg_recall": round(weighted_recall, 4),
                    "avg_precision": round(weighted_precision, 4),
                    "avg_f1": round(weighted_f1, 4)
                },
                "loss": round(loss_value, 4) if loss_value is not None else None,
                "roc_auc_ovr": round(roc_auc_ovr, 4) if roc_auc_ovr is not None else None,
                "roc_auc_ovo": round(roc_auc_ovo, 4) if roc_auc_ovo is not None else None,
                "mcc": round(mcc, 4),
                "quadratic_kappa": round(kappa, 4)
            }
        }

        results['json_format'] = json_results

        print(f"\nCLASS IMBALANCE ANALYSIS:")
        print("=" * 30)
        class_counts = test_data[target_column].value_counts()
        for label, count in class_counts.items():
            pct = count / len(test_data) * 100
            print(f"{label}: {count} samples ({pct:.1f}%)")

    
        pred_counts = pd.Series(y_pred).value_counts()
        print(f"\nPREDICTION DISTRIBUTION:")
        print("=" * 25)
        for label, count in pred_counts.items():
            pct = count / len(y_pred) * 100
            print(f"Predicted {label}: {count} ({pct:.1f}%)")

        print(f"\nTEST SET DETAILED METRICS:")
        print("=" * 30)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f} (accounts for class imbalance)")
        print(f"F1 Macro: {macro_f1:.4f}")
        print(f"F1 Weighted: {weighted_f1:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        if loss_value is not None:
            print(f"Log Loss: {loss_value:.4f}")

        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred))

        
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print("Predicted →")
        print(f"Actual ↓    deceased  living")
        labels = sorted(y_true.unique())
        for i, true_label in enumerate(labels):
            row_str = f"{true_label:8s}"
            for j, pred_label in enumerate(labels):
                row_str += f"{cm[i,j]:8d}"
            print(row_str)

        
        try:
            
            print(f"\nMODEL ARCHITECTURE ANALYSIS:")
            print("=" * 30)
            print(f"Model path: {predictor.path}")
            print(f"Model type: {type(predictor)}")

            
            if hasattr(predictor, '_feature_importance'):
                feature_importance = predictor._feature_importance
                results['feature_importance'] = feature_importance
                print("Feature importance available through _feature_importance")
            else:
                print("Feature importance not available in this AutoGluon version")

        except Exception as e:
            print(f"Model inspection failed: {e}")

        if y_prob_deceased is not None:
            print(f"\nTHRESHOLD ANALYSIS (for imbalanced data):")
            print("=" * 40)

            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            print("Threshold | Accuracy | Balanced Acc | F1 Score")
            print("----------|----------|--------------|----------")

            for threshold in thresholds:
                y_pred_thresh = (y_prob_deceased >= threshold).astype(int)
                y_pred_thresh_labels = ['living' if p == 0 else 'deceased' for p in y_pred_thresh]

                acc_thresh = accuracy_score(y_true, y_pred_thresh_labels)
                bal_acc_thresh = balanced_accuracy_score(y_true, y_pred_thresh_labels)
                f1_thresh = f1_score(y_true, y_pred_thresh_labels, average='weighted')

                print(f"   {threshold:.1f}    |   {acc_thresh:.3f}  |    {bal_acc_thresh:.3f}     |   {f1_thresh:.3f}")

            print(f"\nRECOMMENDATION: Try threshold ~0.6-0.7 for better balance")

        print(f"\nJSON FORMATTED RESULTS:")
        print("=" * 25)
        import json
        json_output = json.dumps(json_results, indent=2)
        print(json_output)

        try:
            if results_folder:
                metrics_dir = os.path.join(results_folder, 'metrics')
                base_filename = 'model_metrics'
            else:
                metrics_dir = '.'
                base_filename = 'model_metrics'
            
            counter = 1
            while os.path.exists(os.path.join(metrics_dir, f'{base_filename}_{counter}.json')):
                counter += 1

            filename = os.path.join(metrics_dir, f'{base_filename}_{counter}.json')
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to '{filename}'")
            
            if results_folder:
                try:
                    predictions_file = os.path.join(results_folder, 'predictions', 'test_predictions.csv')
                    predictions_df = test_data.copy()
                    predictions_df['predicted'] = predictor.predict(test_data)
                    if y_prob_deceased is not None:
                        predictions_df['probability_deceased'] = y_prob_deceased
                    predictions_df.to_csv(predictions_file, index=False)
                    print(f"Predictions saved to '{predictions_file}'")
                except Exception as pred_e:
                    print(f"WARNING: Could not save predictions: {pred_e}")
            
        except Exception as e:
            print(f"WARNING: Could not save JSON file: {e}")

    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    return results

def main():
    print("HEAD & NECK CANCER MULTIMODAL PREDICTION")
    print("AutoGluon MultiModal Training Pipeline")
    print("=" * 80)

    try:
        results_folder = create_autogluon_results_folder()
        
        setup_colab_environment()

        df = load_multimodal_data()

        df_multimodal = prepare_multimodal_format(df)

        predictor, train_data, val_data, test_data, basic_score, results_folder = train_autogluon_multimodal(df_multimodal, results_folder=results_folder)

        results = analyze_results(predictor, train_data, val_data, test_data, results_folder=results_folder)

        print(f"\nTRAINING PIPELINE COMPLETED!")
        print(f"All results saved to: {results_folder}")

        json_results = results.get('json_format', {})
        
        try:
            summary_file = os.path.join(results_folder, 'RESULTS_SUMMARY.txt')
            with open(summary_file, 'w') as f:
                f.write("AutoGluon MultiModal Head & Neck Cancer Survival Prediction\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Results folder: {results_folder}\n\n")
                
                if json_results and 'test_metrics' in json_results:
                    f.write("Test Set Performance:\n")
                    f.write("-" * 20 + "\n")
                    metrics = json_results['test_metrics']
                    f.write(f"Accuracy: {metrics.get('accuracy', 'N/A')}\n")
                    f.write(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A')}\n")
                    f.write(f"ROC AUC: {metrics.get('roc_auc_ovr', 'N/A')}\n")
                    f.write(f"F1 Macro: {metrics.get('macro', {}).get('avg_f1', 'N/A')}\n")
                
                f.write(f"\nFolder Structure:\n")
                f.write(f"├── models/           # Trained AutoGluon models\n")
                f.write(f"├── metrics/          # Performance metrics (JSON)\n") 
                f.write(f"├── predictions/      # Test set predictions\n")
                f.write(f"├── data_splits/      # Train/validation/test splits\n")
                f.write(f"├── visualizations/   # Future: plots and charts\n")
                f.write(f"└── RESULTS_SUMMARY.txt # This summary file\n")
            
            print(f"Summary saved to: {summary_file}")
        except Exception as e:
            print(f"WARNING: Could not create summary file: {e}")

        return predictor, results, json_results, results_folder

    except Exception as e:
        print(f"ERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    predictor, results, json_results, results_folder = main()

    if json_results:
        print(f"\nFINAL JSON RESULTS:")
        print("=" * 40)
        import json
        print(json.dumps(json_results, indent=2))

        print(f"\nInput/Output Summary:")
        print(f"INPUT MODALITIES:")
        print(f"   - Text: Surgery reports (German + English)")
        print(f"   - Structured: Clinical + Pathological data")
        print(f"   - Image-derived: TMA cell density + WSI geometric features")
        print(f"OUTPUT:")
        print(f"   - Binary prediction: Survival status (deceased/alive)")
        print(f"   - Confidence scores for each prediction")
        print(f"   - JSON formatted metrics matching standard format")
        
        if results_folder:
            print(f"\nALL RESULTS ORGANIZED IN: {results_folder}")
            print(f"Copy this folder to your Google Drive to preserve all results!")
        
    print(f"\n{'='*60}")
    print(f"AUTOGLUON MULTIMODAL TRAINING COMPLETE")
    print(f"{'='*60}")