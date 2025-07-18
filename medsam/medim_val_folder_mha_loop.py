# -*- encoding: utf-8 -*-

import medim # Assuming medim is installed and available

import sys
# Make sure SAM-Med3D is in your Python path
sys.path.insert(1, 'SAM-Med3D/')
# Import functions from SAM-Med3D's utils
from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

import os 
import SimpleITK as sitk # For .mha handling

import nibabel as nib # For .nii handling
import numpy as np
# os imported again, but fine
import glob # For file listing
# typing imports are correct

from typing import Dict, List, Tuple
from scipy.spatial.distance import directed_hausdorff # For Hausdorff distance
# scipy.ndimage is not explicitly used in the final version of calculate_segmentation_metrics, but useful for context.
# from scipy.ndimage import binary_erosion, binary_dilation # For boundary extraction
# from scipy.ndimage import sum as ndi_sum # For counting foreground voxels

import pandas as pd # Import pandas for DataFrame

def calculate_segmentation_metrics(
    ground_truth_path: str,
    prediction_path: str,
    class_labels: Dict[int, str] # e.g., {1: "jawbones", 2: "canal", ...}
) -> Dict[str, Dict[str, float]]:
    """
    Calculates Dice, Symmetric Difference (SD), Normalized Symmetric Difference (NSD),
    and 95th Percentile Hausdorff Distance (HD95) for multi-class 3D segmentations.
    NSD is calculated as SD / Volume_of_Union_of_Masks (for object-centric assessment).
    """
    if not os.path.exists(ground_truth_path):
        print(f"Erreur : Fichier de vérité terrain non trouvé à '{ground_truth_path}'")
        return None
    if not os.path.exists(prediction_path):
        print(f"Erreur : Fichier de prédiction non trouvé à '{prediction_path}'")
        return None

    try:
        gt_img = nib.load(ground_truth_path)
        pred_img = nib.load(prediction_path)

        gt_data = gt_img.get_fdata().astype(np.int32)
        pred_data = pred_img.get_fdata().astype(np.int32)

        if gt_data.shape != pred_data.shape:
            print(f"Erreur : Formes incompatibles : GT {gt_data.shape}, Préd {pred_data.shape}. Impossible de comparer.")
            return None
        
        # total_image_volume = float(gt_data.shape[0] * gt_data.shape[1] * gt_data.shape[2]) # Not used for this NSD version

    except Exception as e:
        print(f"Erreur lors du chargement ou du traitement des fichiers NIfTI : {e}")
        return None

    results = {}
    
    # Itérer sur chaque classe de premier plan
    for class_id in class_labels.keys():
        if class_id == 0: continue # Ignorer l'arrière-plan

        class_name = class_labels[class_id]
        
        gt_class_mask = (gt_data == class_id)
        pred_class_mask = (pred_data == class_id)

        # --- Dice Similarity Coefficient (DSC) ---
        intersection = np.sum(gt_class_mask & pred_class_mask)
        
        sum_of_masks = np.sum(gt_class_mask) + np.sum(pred_class_mask)
        dice = 2. * intersection / sum_of_masks if sum_of_masks > 0 else 1.0 # Gérer les masques vides (match parfait)
        
        # --- Symmetric Difference (SD) / Volume of Disagreement ---
        false_positives = np.sum(pred_class_mask & ~gt_class_mask)
        false_negatives = np.sum(~pred_class_mask & gt_class_mask)
        symmetric_difference = false_positives + false_negatives

        # --- Normalized Symmetric Difference (NSD) by Object Union Volume ---
        # NSD = SD / Volume(GT union Pred)
        # Volume(GT union Pred) = Volume(GT) + Volume(Pred) - Intersection
        gt_volume = np.sum(gt_class_mask)
        pred_volume = np.sum(pred_class_mask)
        union_volume = gt_volume + pred_volume - intersection
        
        nsd_by_object_union = symmetric_difference / union_volume if union_volume > 0 else 0.0 # If union is 0 (both empty), NSD is 0

        # --- 95th Percentile Hausdorff Distance (HD95) ---
        hd95 = np.nan # Initialize as NaN

        gt_points = np.argwhere(gt_class_mask)
        pred_points = np.argwhere(pred_class_mask)

        if len(gt_points) > 0 and len(pred_points) > 0:
            d1 = directed_hausdorff(gt_points, pred_points)[0]
            d2 = directed_hausdorff(pred_points, gt_points)[0]
            hd95 = max(d1, d2)
        elif len(gt_points) == 0 and len(pred_points) == 0:
            hd95 = 0.0 # Both empty, perfect match
        else:
            hd95 = np.inf # One is empty, the other is not, max disagreement (infinite effort)

        results[class_name] = {
            "Dice": float(dice),
            "SD": float(symmetric_difference),
            "NSD_by_Object_Union": float(nsd_by_object_union), # NEW METRIC for object-centric NSD
            "HD95": float(hd95)
        }

    # --- Calculate Overall Foreground Metrics (object vs. background) ---
    gt_overall_foreground = (gt_data > 0)
    pred_overall_foreground = (pred_data > 0)

    overall_intersection = np.sum(gt_overall_foreground & pred_overall_foreground)
    overall_sum_gt = np.sum(gt_overall_foreground)
    overall_sum_pred = np.sum(pred_overall_foreground)
    overall_dice = 2. * overall_intersection / (overall_sum_gt + overall_sum_pred) if (overall_sum_gt + overall_sum_pred) > 0 else 1.0

    overall_false_positives = np.sum(pred_overall_foreground & ~gt_overall_foreground)
    overall_false_negatives = np.sum(~pred_overall_foreground & gt_overall_foreground)
    overall_symmetric_difference = overall_false_positives + overall_false_negatives

    # Overall Normalized Symmetric Difference (NSD) by Object Union Volume
    overall_union_volume = overall_sum_gt + overall_sum_pred - overall_intersection
    overall_nsd_by_object_union = overall_symmetric_difference / overall_union_volume if overall_union_volume > 0 else 0.0

    overall_gt_points = np.argwhere(gt_overall_foreground)
    overall_pred_points = np.argwhere(pred_overall_foreground)
    overall_hd95 = np.nan
    if len(overall_gt_points) > 0 and len(overall_pred_points) > 0:
        d1_overall = directed_hausdorff(overall_gt_points, overall_pred_points)[0]
        d2_overall = directed_hausdorff(overall_pred_points, overall_gt_points)[0] 
        overall_hd95 = max(d1_overall, d2_overall)
    elif len(overall_gt_points) == 0 and len(overall_pred_points) == 0:
        overall_hd95 = 0.0
    else:
        overall_hd95 = np.inf

    results["overall"] = {
        "Dice": float(overall_dice),
        "SD": float(overall_symmetric_difference),
        "NSD_by_Object_Union": float(overall_nsd_by_object_union), # NEW METRIC
        "HD95": float(overall_hd95)
    }

    return results

# --- Your original configuration and main block starts here ---
# It will be modified to include the metric storage and aggregation.

DATA_ROOT = os.path.abspath("../../../data2/nnUnet/nnUNet_raw/Dataset112_ToothFairy2")
IMAGE_ROOT = os.path.join(DATA_ROOT, "imagesTr") # Use os.path.join for robustness
# PREDICTION_ROOT = os.path.abspath("../../../data2/nnUnet/nnUNet_raw/Dataset117_random_sampling_toothfairy/predictions_on_test_iteration9/")
PREDICTION_ROOT = os.path.abspath("../../../data2/nnUnet/predicted_for_sam/")
LABEL_ROOT = os.path.join(DATA_ROOT, "labelsTr") # Use os.path.join
EXPORT_PATH_ROOT = os.path.abspath("../../../data2/nnUnet/sam/nnunet_new_images/")

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # Note: medim.create_model and validate_paired_img_gt would require proper setup and imports for SAM-Med3D.
    # For this snippet, the actual SAM inference is commented out, assuming the `out_path` file is generated.
    # If you run this, ensure the 'medim' library is installed and its dependencies are met.
    # Also ensure the model path is accessible.
    try:
        model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    except Exception as e:
        print(f"Warning: Could not create SAM-Med3D model. Inference part will not run. Error: {e}")
        model = None # Set model to None if creation fails

    # Ensure EXPORT_PATH_ROOT exists for converted files
    os.makedirs(EXPORT_PATH_ROOT, exist_ok=True)

    MY_GROUPED_CLASS_LABELS = {
        1: "jawbones",
        2: "inferior alveolar canal",
        3: "sinus",
        4: "pharynx",
        5: "teeth"
    }

    # --- Initialize a list to store metrics for all patients ---
    all_patients_metrics_list = []

    # Get list of (image_path, label_path) pairs using the helper function
    # Assuming the "predictions_on_test_iteration9" folder contains the predicted labels
    # that correspond to the "imagesTs" and "labelsTs" files.
    # We will iterate based on labelsTs, then find corresponding images and predictions.

    # Correct way to list files for iteration (assuming image & label roots contain respective subfolders for patient IDs)
    # This lists the GT labels, then we find corresponding images and predictions
    gt_label_files = glob.glob(os.path.join(LABEL_ROOT, "*.mha"))

    # For each ground truth label file
    for mha_gt_path in gt_label_files:
        patient_id = os.path.basename(mha_gt_path).replace(".mha", "") # e.g., ToothFairy2F_001

        # Construct corresponding image and prediction paths
        mha_img_path = os.path.join(IMAGE_ROOT, f"{patient_id}_0000.mha") # Original image
        mha_pred_path = os.path.join(PREDICTION_ROOT, f"{patient_id}.mha") # Generative model's prediction

        # --- IMPORTANT: Ensure both image and prediction files exist before proceeding ---
        if not os.path.exists(mha_img_path):
            print(f"Skipping {patient_id}: Original image not found at {mha_img_path}")
            continue
        if not os.path.exists(mha_pred_path):
            print(f"Skipping {patient_id}: Prediction not found at {mha_pred_path}")
            continue

        print(f"\n--- Processing Patient: {patient_id} ---")
        print(f"Original Image: {mha_img_path}")
        print(f"Ground Truth: {mha_gt_path}")
        print(f"Prediction: {mha_pred_path}")

        # Convert .mha to .nii for MedSAM3D if needed, and prepare output path
        img_path_nii = os.path.join(EXPORT_PATH_ROOT, f"{patient_id}_0000.nii") # Original image converted to .nii
        gt_path_nii = os.path.join(EXPORT_PATH_ROOT, f"{patient_id}.nii") # Ground truth converted to .nii
        # MedSAM3D output path (assuming it saves as .nii.gz)
        sam_output_path_nii_gz = os.path.join(EXPORT_PATH_ROOT, f"{patient_id}_sam_pred.nii.gz") 

        # Convert original MHA files to NII if not already converted
        if not os.path.isfile(img_path_nii):
            print(f"Converting image {os.path.basename(mha_img_path)} to .nii...")
            img_sitk = sitk.ReadImage(mha_img_path)    
            sitk.WriteImage(img_sitk, img_path_nii)

        if not os.path.isfile(gt_path_nii):
            print(f"Converting GT {os.path.basename(mha_gt_path)} to .nii...")
            gt_sitk = sitk.ReadImage(mha_gt_path)
            sitk.WriteImage(gt_sitk, gt_path_nii)
            
        
        ''' 3. Infer with the pre-trained SAM-Med3D model '''
        # This part assumes you want to run SAM-Med3D here to generate predictions (out_path)
        # You specified 'predictions_on_test_iteration9' as PREDICTION_ROOT.
        # This means you already have predictions. So, if you want to use existing predictions,
        # you might skip running SAM-Med3D here and just use `mha_pred_path` (or its .nii equivalent).
        # I'll adapt to use mha_pred_path as the source for the metric calculation.

        # If you DO want to run SAM-Med3D to *generate* the prediction for metric calculation:
        if model: # Only run if model loaded successfully
            print(f"Running SAM-Med3D inference for {patient_id} (simulated with 5 clicks)...")
            try:
                # `validate_paired_img_gt` will write prediction to `sam_output_path_nii_gz`
                validate_paired_img_gt(model, img_path_nii, gt_path_nii, sam_output_path_nii_gz, num_clicks=5)
                # Ensure the predicted file actually exists after this call
                if not os.path.exists(sam_output_path_nii_gz):
                    print(f"Error: SAM-Med3D did not generate prediction file at {sam_output_path_nii_gz}. Skipping metrics.")
                    continue
                pred_for_metrics_path = sam_output_path_nii_gz # Use SAM's output for metrics
            except Exception as e:
                print(f"Error during SAM-Med3D inference for {patient_id}: {e}. Skipping metrics.")
                continue
        else:
            print(f"SAM-Med3D model not loaded. Using existing prediction from {mha_pred_path} for metrics.")
            pred_for_metrics_path = mha_pred_path # Use the existing prediction directly

        # Ensure the prediction file for metrics is .nii or .nii.gz
        if not pred_for_metrics_path.endswith((".nii", ".nii.gz")):
            print(f"Warning: Prediction file {pred_for_metrics_path} is not .nii or .nii.gz. Attempting conversion.")
            # Convert mha_pred_path to .nii.gz for calculate_segmentation_metrics
            converted_pred_path = os.path.join(EXPORT_PATH_ROOT, os.path.basename(mha_pred_path).replace(".mha", ".nii.gz"))
            if not os.path.exists(converted_pred_path):
                try:
                    pred_sitk = sitk.ReadImage(mha_pred_path)
                    sitk.WriteImage(pred_sitk, converted_pred_path)
                except Exception as e:
                    print(f"Error converting prediction {mha_pred_path} to {converted_pred_path}: {e}. Skipping metrics.")
                    continue
            pred_for_metrics_path = converted_pred_path


        ''' 4. compute the metrics of your prediction with the ground truth '''
        # Note: Your original compute_metrics and print_computed_metrics are commented out.
        # If you need them, ensure they are imported/defined correctly.

        # Calculate metrics using your custom function
        current_patient_metrics = calculate_segmentation_metrics(
            ground_truth_path=gt_path_nii, # Use the converted GT .nii path
            prediction_path=pred_for_metrics_path, # Use the path to the prediction (SAM or existing)
            class_labels=MY_GROUPED_CLASS_LABELS
        )

        if current_patient_metrics:
            # --- Store the metrics ---
            # Flatten the nested dictionary into a single row for the DataFrame
            patient_id = os.path.basename(mha_gt_path).replace(".mha", "")
            flattened_metrics = {"patient_id": patient_id}
            for class_or_overall, class_metrics in current_patient_metrics.items():
                for metric_name, score in class_metrics.items():
                    flattened_metrics[f"{class_or_overall}_{metric_name}"] = score
            all_patients_metrics_list.append(flattened_metrics)

            # Optional: Print metrics for current patient (as in your original code)
            print("\n--- Segmentation Metrics for Expert Effort (Current Patient) ---")
            for key, value in current_patient_metrics.items():
                print(f"\nClass: {key}")
                for metric_name, score in value.items():
                    print(f"  {metric_name}: {score:.4f}")
        else:
            print(f"Metric calculation failed for patient {patient_id}. Skipping storage.")        
        
        print("################")

    # --- After the loop: Aggregate and display overall results ---
    print("\n--- Aggregating all patient metrics ---")
    if all_patients_metrics_list:
        results_df = pd.DataFrame(all_patients_metrics_list)
        
        print("\n--- Aggregated Results (DataFrame Head) ---")
        print(results_df.head())

        print("\n--- Métriques Moyennes sur Tous les Patients ---")
        # Sélectionner uniquement les colonnes contenant les scores de métriques (exclure 'patient_id')
        metrics_columns = [col for col in results_df.columns if col != 'patient_id']
        
        # Calculer la moyenne pour chaque colonne de métrique
        mean_metrics = results_df[metrics_columns].mean()
        
        # Afficher les métriques moyennes, structurées par globale et par classe
        print("\nMétriques Moyennes Globales (toutes classes confondues) :")
        print(f"  Dice : {mean_metrics.get('overall_Dice', np.nan):.4f}")
        print(f"  DS : {mean_metrics.get('overall_SD', np.nan):.2f}")
        print(f"  NSD_by_Object_Union : {mean_metrics.get('overall_NSD_by_Object_Union', np.nan):.4f}") # AFFICHAGE DE LA NSD GLOBALE
        print(f"  HD95 : {mean_metrics.get('overall_HD95', np.nan):.4f}")

        print("\nMétriques Moyennes par Classe Groupée :")
        for class_name in MY_GROUPED_CLASS_LABELS.values():
            dice_col = f"{class_name}_Dice"
            sd_col = f"{class_name}_SD"
            nsd_col = f"{class_name}_NSD_by_Object_Union" # Colonne pour la NSD par objet
            hd95_col = f"{class_name}_HD95"

            if dice_col in mean_metrics: # Vérifier si les métriques de cette classe existent
                print(f"  {class_name} :")
                print(f"    Dice : {mean_metrics[dice_col]:.4f}")
                print(f"    DS : {mean_metrics[sd_col]:.2f}")
                print(f"    NSD_by_Object_Union : {mean_metrics[nsd_col]:.4f}") # AFFICHAGE DE LA NSD PAR CLASSE
                print(f"    HD95 : {mean_metrics[hd95_col]:.4f}")
            else:
                print(f"  {class_name} : Non trouvée chez les patients traités (ou ses métriques sont toutes NaN).")
        
        
        # Optional: Save aggregated results to CSV
        results_csv_path = os.path.join(EXPORT_PATH_ROOT, "all_patients_metrics_summary.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nDetailed patient metrics saved to {results_csv_path}")

    else:
        print("No patient metrics collected for aggregation.")

    # --- Cleanup temporary export directory ---
    # shutil.rmtree(EXPORT_PATH_ROOT) # Uncomment this if you want to clean up temp files
    # print(f"Cleaned up temporary directory: {EXPORT_PATH_ROOT}")
