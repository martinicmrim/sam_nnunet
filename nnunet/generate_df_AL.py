import os
import shutil
import random
import json
from glob import glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk # Ensure you have SimpleITK installed: pip install SimpleITK
from collections import OrderedDict # To keep order of labels in JSON
from typing import List, Dict, Tuple, Set

# --- 1. Embedded Original Dataset Info (from your dataset (1).json) ---
ORIGINAL_DATASET_INFO_JSON = {
 "name": "ToothFairy 2",
 "description": "Segmentation of maxillofacial CBCT volumes",
 "reference": "https://ditto.ing.unimore.it/toothfairy2/",
 "license": "CC-BY-SA 4.0",
 "release": "20/04/2024",
 "latestUpdate": "07/07/2024",
 "tensorImageSize": "4D",
 "labels": {
  "Background": 0,
  "Lower Jawbone": 1,
  "Upper Jawbone": 2,
  "Left Inferior Alveolar Canal": 3,
  "Right Inferior Alveolar Canal": 4,
  "Left Maxillary Sinus": 5,
  "Right Maxillary Sinus": 6,
  "Pharynx": 7,
  "Bridge": 8,
  "Crown": 9,
  "Implant": 10,
  "Upper Right Central Incisor": 11,
  "Upper Right Lateral Incisor": 12,
  "Upper Right Canine": 13,
  "Upper Right First Premolar": 14,
  "Upper Right Second Premolar": 15,
  "Upper Right First Molar": 16,
  "Upper Right Second Molar": 17,
  "Upper Right Third Molar (Wisdom Tooth)": 18,
  "NA": 40,
  "Upper Left Central Incisor": 21,
  "Upper Left Lateral Incisor": 22,
  "Upper Left Canine": 23,
  "Upper Left First Premolar": 24,
  "Upper Left Second Premolar": 25,
  "Upper Left First Molar": 26,
  "Upper Left Second Molar": 27,
  "Upper Left Third Molar (Wisdom Tooth)": 28,
  "Lower Left Central Incisor": 31,
  "Lower Left Lateral Incisor": 32,
  "Lower Left Canine": 33,
  "Lower Left First Premolar": 34,
  "Lower Left Second Premolar": 35,
  "Lower Left First Molar": 36,
  "Lower Left Second Molar": 37,
  "Lower Left Third Molar (Wisdom Tooth)": 38,
  "Lower Right Central Incisor": 41,
  "Lower Right Lateral Incisor": 42,
  "Lower Right Canine": 43,
  "Lower Right First Premolar": 44,
  "Lower Right Second Premolar": 45,
  "Lower Right First Molar": 46,
  "Lower Right Second Molar": 47,
  "Lower Right Third Molar (Wisdom Tooth)": 48
 }
}

# --- 2. Define Your Grouping Decision for Remapping ---
# This dictionary maps ORIGINAL numerical IDs to their NEW GROUPED numerical IDs.
# Classes are: Background (0), Jawbones (1), Canal (2), Sinus (3), Pharynx (4), Teeth (5).
# Others (Bridge, Crown, Implant, NA) are mapped to Background (0).
GROUPED_LABELS_MAPPING_DECISION = {
    # Background (remains 0)
    0: 0, # Background

    # Jawbones (new ID 1)
    1: 1, # Lower Jawbone
    2: 1, # Upper Jawbone

    # Inferior Alveolar Canal (new ID 2)
    3: 2, # Left Inferior Alveolar Canal
    4: 2, # Right Inferior Alveolar Canal

    # Sinus (new ID 3)
    5: 3, # Left Maxillary Sinus
    6: 3, # Right Maxillary Sinus

    # Pharynx (new ID 4)
    7: 4, # Pharynx

    # Teeth (new ID 5) - All individual tooth IDs from original dataset
    11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, # Upper Right
    21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5, # Upper Left
    31: 5, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, # Lower Left
    41: 5, 42: 5, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,  # Lower Right

    # Other classes (Bridge, Crown, Implant, NA) - mapped to Background (0)
    8: 0, # Bridge
    9: 0, # Crown
    10: 0, # Implant
    40: 0 # NA
}

# --- 3. Updated `remap_segmentation_labels` Function ---
def remap_segmentation_labels(
    input_mask_path: str,
    output_mask_path: str,
    label_mapping: Dict[int, int], # This mapping now includes your grouping decisions
    background_value: int = 0
):
    """
    Remaps labels in a 3D segmentation mask according to a provided mapping.
    Supports .nii, .nii.gz, and .mha input/output formats.

    Args:
        input_mask_path (str): Path to the input segmentation mask file (.nii/.nii.gz/.mha).
        output_mask_path (str): Path where the remapped mask will be saved (.nii.gz or .mha).
        label_mapping (dict): A dictionary where keys are original label values
                              and values are the new, consecutive label values.
        background_value (int): The value representing background in the input mask.
                                These will always be mapped to 0 in the output.
    """
    input_ext = os.path.splitext(input_mask_path)[1].lower()
    if input_ext == '.gz': 
        input_ext = os.path.splitext(os.path.splitext(input_mask_path)[0])[1].lower() + input_ext

    output_ext = os.path.splitext(output_mask_path)[1].lower()
    if output_ext == '.gz': 
        output_ext = os.path.splitext(os.path.splitext(output_mask_path)[0])[1].lower() + output_ext

    original_image_metadata = None
    mask_data = None

    if input_ext in ['.nii', '.nii.gz']:
        nifti_img = nib.load(input_mask_path)
        mask_data = nifti_img.get_fdata().astype(np.int32)
        original_image_metadata = {"affine": nifti_img.affine, "header": nifti_img.header}
    elif input_ext == '.mha':
        original_sitk_img = sitk.ReadImage(input_mask_path)
        mask_data = sitk.GetArrayFromImage(original_sitk_img).astype(np.int32)
        original_image_metadata = {"sitk_image": original_sitk_img}
    else:
        print(f"Unsupported input file format: {input_ext}. Supported: .nii, .nii.gz, .mha")
        return

    remapped_mask_data = np.zeros_like(mask_data, dtype=np.int32)
    
    for original_label_id in np.unique(mask_data):
        if original_label_id == background_value:
            remapped_mask_data[mask_data == original_label_id] = 0
        elif original_label_id in label_mapping:
            remapped_mask_data[mask_data == original_label_id] = label_mapping[original_label_id]
        else:
            print(f"Warning: Original label {original_label_id} found in mask {os.path.basename(input_mask_path)} but not in `label_mapping`. Mapping to background (0).")


    unique_remapped_labels = np.unique(remapped_mask_data)
    unique_remapped_labels = unique_remapped_labels[unique_remapped_labels >= 0] 

    expected_labels = np.arange(len(unique_remapped_labels))
    if not np.array_equal(np.sort(unique_remapped_labels), expected_labels):
        print(f"ERROR: Remapped labels {np.sort(unique_remapped_labels)} are NOT perfectly consecutive 0,1,2... for nnU-Net. Please double check your `label_mapping` and ensure all active original labels are correctly grouped.")
        print(f"Expected labels: {expected_labels}")
    
    if output_ext in ['.nii', '.nii.gz']:
        remapped_nifti_img = nib.Nifti1Image(remapped_mask_data, original_image_metadata["affine"], original_image_metadata["header"])
        nib.save(remapped_nifti_img, output_mask_path)
    elif output_ext == '.mha':
        remapped_sitk_img = sitk.GetImageFromArray(remapped_mask_data)
        if "sitk_image" in original_image_metadata:
            remapped_sitk_img.CopyInformation(original_image_metadata["sitk_image"])
        else:
            print("Warning: Input was NIfTI. Spacing, Origin, Direction might not be accurately copied to MHA if not explicitly handled from NIfTI header.")

        sitk.WriteImage(remapped_sitk_img, output_mask_path)

# --- Helper to get unique labels from a set of masks ---
def get_unique_labels_from_masks(mask_paths: List[str]):
    all_unique_labels = set()
    for mask_path in mask_paths:
        try:
            if mask_path.endswith(('.nii', '.nii.gz')):
                mask_data = nib.load(mask_path).get_fdata().astype(np.int32)
            elif mask_path.endswith('.mha'):
                mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.int32)
            else:
                continue
            all_unique_labels.update(np.unique(mask_data[mask_data > 0]))
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
    return sorted(list(all_unique_labels))

# --- 4. Updated `setup_nnunet_al_directories` Function ---
def setup_nnunet_al_directories(
    raw_data_source_dir: str,          # E.g., "/path/to/your_nnunet_format_data/" (containing imagesTr/ and labelsTr/)
    nnunet_base_dir: str,              # E.g., "/path/to/nnUNet_raw_data_base/"
    al_experiment_root_dir: str,       # E.g., "/path/to/al_experiments/"
    task_id: int,                      # Your nnU-Net Task ID (e.g., 112)
    experiment_name: str,              # E.g., "entropy_sampling", "random_sampling"
    test_set_ratio: float = 0.15,      # Proportion of data for the fixed test set
    initial_seed_ratio: float = 0.05,  # Proportion of remaining data for initial labeled pool
    random_seed: int = 42,             # Seed for reproducibility
    image_ext: str = ".mha",        # Extension of your image files
    label_ext: str = ".mha"         # Extension of your label files
) -> Tuple[str, str, str, str]:
    """
    Sets up the directory structure for an nnU-Net Active Learning experiment.
    Includes remapping and grouping of labels.

    Args:
        raw_data_source_dir (str): Path to the base directory of your *original*
                                   nnU-Net formatted data (containing imagesTr/ and labelsTr/).
                                   E.g., /path/to/nnUNet_raw_data/TaskXXX_OriginalDataset/
        nnunet_base_dir (str): The root directory for nnU-Net raw data
                                (e.g., where nnUNet_raw_data/ will be created).
        al_experiment_root_dir (str): The root directory where AL-specific pools
                                      (unlabeled_pool_images/labels) will reside.
        task_id (int): The nnU-Net Task ID (e.g., 112).
        experiment_name (str): A descriptive name for this AL run (e.g., "entropy_sampling").
        test_set_ratio (float): Proportion of total data to reserve for the fixed test set.
        initial_seed_ratio (float): Proportion of remaining data for the initial labeled pool.
        random_seed (int): Seed for random operations to ensure reproducibility.
        image_ext (str): File extension for image files (e.g., ".nii.gz", ".mha").
        label_ext (str): File extension for label files (e.g., ".nii.gz", ".mha").

    Returns:
        Tuple[str, str, str, str]: Paths to (
            current_nnunet_task_dir,
            current_labeled_images_tr,
            current_unlabeled_images_pool,
            current_unlabeled_labels_pool
        )
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # --- Define paths ---
    task_name = f"Dataset{task_id}_{experiment_name}"
    nnunet_raw_data_task_dir = os.path.join(nnunet_base_dir, "nnUNet_raw", task_name)
    remapped_data_temp_dir = os.path.join(al_experiment_root_dir, "temp_remapped_data_for_setup")

    al_run_dir = os.path.join(al_experiment_root_dir, f"run_{experiment_name}")
    current_labeled_images_tr = os.path.join(nnunet_raw_data_task_dir, "imagesTr")
    current_labeled_labels_tr = os.path.join(nnunet_raw_data_task_dir, "labelsTr")
    current_test_images_ts = os.path.join(nnunet_raw_data_task_dir, "imagesTs")
    current_test_labels_ts = os.path.join(nnunet_raw_data_task_dir, "labelsTs")
    current_unlabeled_images_pool = os.path.join(al_run_dir, "unlabeled_pool_images")
    current_unlabeled_labels_pool = os.path.join(al_run_dir, "unlabeled_pool_labels")

    # --- Clean up and Create Directories ---
    print(f"Setting up directories for AL run: {experiment_name}")
    print("Cleaning up existing temporary and nnU-Net task directories...")
    if os.path.exists(nnunet_raw_data_task_dir): shutil.rmtree(nnunet_raw_data_task_dir)
    if os.path.exists(al_run_dir): shutil.rmtree(al_run_dir)
    if os.path.exists(remapped_data_temp_dir): shutil.rmtree(remapped_data_temp_dir)

    os.makedirs(current_labeled_images_tr)
    os.makedirs(current_labeled_labels_tr)
    os.makedirs(current_test_images_ts)
    os.makedirs(current_test_labels_ts)
    os.makedirs(current_unlabeled_images_pool)
    os.makedirs(current_unlabeled_labels_pool)
    os.makedirs(remapped_data_temp_dir) # Temp dir to hold all remapped data before splitting

    print("Directories created/cleaned.")

    # --- Collect all patient IDs from original source (nnU-Net format) ---
    # CORRECTED: Globbing from imagesTr subdirectory
    all_image_files_original = glob(os.path.join(raw_data_source_dir, "imagesTr", f"*{image_ext}"))
    all_patient_ids = sorted([os.path.basename(f).replace(f"_0000{image_ext}", "") for f in all_image_files_original])
    
    if not all_patient_ids:
        raise ValueError(f"No image files found in {os.path.join(raw_data_source_dir, 'imagesTr')} with extension {image_ext}. Please check path and naming.")

    print(f"\nFound {len(all_patient_ids)} total patient IDs in source directory.")

    # --- Step 2a: Get Actual Unique Labels from Original Masks ---
    print("Scanning original masks to determine actual unique label IDs present for remapping...")
    # CORRECTED: Globbing from labelsTr subdirectory
    all_original_label_paths = [os.path.join(raw_data_source_dir, "labelsTr", f"{p_id}{label_ext}") for p_id in all_patient_ids]
    
    # Filter to ensure only existing paths are passed to get_unique_labels_from_masks
    existing_original_label_paths = [p for p in all_original_label_paths if os.path.exists(p)]
    
    actual_unique_original_labels = get_unique_labels_from_masks(existing_original_label_paths)
    print(f"Actual unique non-zero labels found across all original masks: {actual_unique_original_labels}")

    # --- Step 2b: Construct `my_label_mapping` and `new_labels_dict_for_json` ---
    # This logic now correctly combines the grouping decision with *only* the labels present in your data.
    
    # Start with background mapping
    final_label_mapping = {0: 0} # Background always maps to 0
    
    # Build new_labels_dict_for_json for dataset.json, ensuring lowercase names and consecutive IDs
    new_labels_dict_for_json = OrderedDict()
    new_labels_dict_for_json["background"] = 0

    # Intermediate map: New ID -> List of original IDs that map to it
    new_id_to_original_ids_map = {}
    for orig_id, new_id in GROUPED_LABELS_MAPPING_DECISION.items():
        if orig_id in actual_unique_original_labels or orig_id == 0: # Only consider original labels actually present (or background)
            if new_id not in new_id_to_original_ids_map:
                new_id_to_original_ids_map[new_id] = []
            new_id_to_original_ids_map[new_id].append(orig_id)

    # Process new IDs in consecutive order starting from 1
    # Iterate through target New IDs in their desired order (1, 2, 3, 4, 5)
    desired_new_ids_order = sorted(list(set(GROUPED_LABELS_MAPPING_DECISION.values()) - {0}))
    
    # This manually links the desired new IDs (1-5) to their names for the dataset.json
    new_grouped_id_to_name_map = {
        1: "jawbones",
        2: "inferior alveolar canal",
        3: "sinus",
        4: "pharynx",
        5: "teeth"
    }

    current_new_id_counter = 1
    for new_id_group in desired_new_ids_order:
        new_grouped_name = new_grouped_id_to_name_map.get(new_id_group, f"unknown_group_{new_id_group}")
        new_labels_dict_for_json[new_grouped_name] = current_new_id_counter

        for orig_id_candidate in new_id_to_original_ids_map.get(new_id_group, []):
            if orig_id_candidate in actual_unique_original_labels: # Only if this original ID is actually present
                if orig_id_candidate not in final_label_mapping: # Add only if not already added (e.g., if 0 was added for background)
                    final_label_mapping[orig_id_candidate] = current_new_id_counter
        
        current_new_id_counter += 1

    # Ensure original labels that map to Background (0) are included in final_label_mapping
    # This specifically covers 'other_to_background_ids' if they were present in masks.
    for orig_id, mapped_id in GROUPED_LABELS_MAPPING_DECISION.items():
        if mapped_id == 0 and orig_id != 0 and orig_id in actual_unique_original_labels: # orig_id != 0 to avoid redundant 0:0
            final_label_mapping[orig_id] = 0

    # Verify that all 'actual_unique_original_labels' (non-background) are now in final_label_mapping
    for orig_id in actual_unique_original_labels:
        if orig_id not in final_label_mapping:
            raise ValueError(f"ERROR: Original label ID {orig_id} found in actual masks but not mapped by GROUPED_LABELS_MAPPING_DECISION. Please review the mapping and ensure it covers all actual labels.")
    
    print("\nFinal `my_label_mapping` for remap_segmentation_labels:")
    print(final_label_mapping)
    print("\nNew `labels` dictionary for dataset.json:")
    print(new_labels_dict_for_json)


    # --- Step 3: Apply Global Remapping to ALL Original Masks ---
    print("\nApplying global remapping to all original masks...")
    remapped_count = 0
    for p_id in all_patient_ids:
        print(p_id)
        # CORRECTED: Input masks are from labelsTr subdirectory of raw_data_source_dir
        input_mask_path = os.path.join(raw_data_source_dir, "labelsTr", f"{p_id}{label_ext}")
        output_mask_path = os.path.join(remapped_data_temp_dir, f"{p_id}{label_ext}")
        
        if os.path.exists(input_mask_path): # Only remap if the label file exists
            remap_segmentation_labels(
                input_mask_path=input_mask_path,
                output_mask_path=output_mask_path,
                label_mapping=final_label_mapping # Use the dynamically generated mapping
            )
            remapped_count += 1
        else:
            # This can happen if imagesTr has files that labelsTr doesn't, which is unusual for nnU-Net format
            print(f"Warning: Label file {input_mask_path} not found. Skipping remapping for {p_id}. Ensure imagesTr and labelsTr are consistent.")
    print(f"Remapped {remapped_count} masks to temporary directory.")

    # --- Step 4 & 5: Split Remapped Data & Create `dataset.json` ---
    print("\nSplitting remapped data into Test, Initial Labeled, and Unlabeled pools...")
    
    all_remapped_patient_ids = sorted([os.path.basename(f).replace(f"{label_ext}", "") for f in glob(os.path.join(remapped_data_temp_dir, f"*{label_ext}"))])

    random.shuffle(all_remapped_patient_ids)
    
    num_test_samples = int(len(all_remapped_patient_ids) * test_set_ratio)
    test_ids = all_remapped_patient_ids[:num_test_samples]
    al_pool_ids = all_remapped_patient_ids[num_test_samples:]

    num_initial_seed_samples = int(len(al_pool_ids) * initial_seed_ratio)
    initial_seed_ids = al_pool_ids[:num_initial_seed_samples]
    unlabeled_pool_ids = al_pool_ids[num_initial_seed_samples:]

    print(f"Total remapped volumes: {len(all_remapped_patient_ids)}")
    print(f"Test set: {len(test_ids)} volumes ({test_set_ratio*100:.1f}%)")
    print(f"Initial seed labeled: {len(initial_seed_ids)} volumes ({initial_seed_ratio*100:.1f}% of AL pool)")
    print(f"Remaining unlabeled pool: {len(unlabeled_pool_ids)} volumes")

    # Copy files to final destinations
    for p_id in test_ids:
        # CORRECTED: Image source is imagesTr subdirectory of raw_data_source_dir
        shutil.copy(os.path.join(raw_data_source_dir, "imagesTr", f"{p_id}_0000{image_ext}"), current_test_images_ts)
        shutil.copy(os.path.join(remapped_data_temp_dir, f"{p_id}{label_ext}"), current_test_labels_ts)
    print(f"Copied {len(test_ids)} volumes to Test Set.")

    for p_id in initial_seed_ids:
        # CORRECTED: Image source is imagesTr subdirectory of raw_data_source_dir
        shutil.copy(os.path.join(raw_data_source_dir, "imagesTr", f"{p_id}_0000{image_ext}"), current_labeled_images_tr)
        shutil.copy(os.path.join(remapped_data_temp_dir, f"{p_id}{label_ext}"), current_labeled_labels_tr)
    print(f"Copied {len(initial_seed_ids)} volumes to Initial Labeled Training Pool.")

    for p_id in unlabeled_pool_ids:
        # CORRECTED: Image source is imagesTr subdirectory of raw_data_source_dir
        shutil.copy(os.path.join(raw_data_source_dir, "imagesTr", f"{p_id}_0000{image_ext}"), current_unlabeled_images_pool)
        shutil.copy(os.path.join(remapped_data_temp_dir, f"{p_id}{label_ext}"), current_unlabeled_labels_pool)
    print(f"Copied {len(unlabeled_pool_ids)} volumes to Unlabeled Pool.")

    # Create dataset.json
    dataset_json_path = os.path.join(nnunet_raw_data_task_dir, "dataset.json")
    dataset_json_content = {
        "channel_names": { "0": "CT" },
        "labels": new_labels_dict_for_json, # Use the generated dict with grouped, consecutive labels
        "numTraining": len(initial_seed_ids),
        "numTest": len(test_ids),
        "file_ending": label_ext,
        "name": task_name,
        "description": "Active Learning Experiment for Dental Segmentation with Grouped Classes"
    }
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json_content, f, indent=4)
    print(f"Created dataset.json at: {dataset_json_path}")

    # Clean up temporary remapped data
    shutil.rmtree(remapped_data_temp_dir)
    print(f"Cleaned up temporary remapped data directory: {remapped_data_temp_dir}")

    print("\nDirectory setup complete. You can now run nnUNetv2_plan_and_preprocess.")
    print(f"nnUNet Task Directory: {nnunet_raw_data_task_dir}")
    print(f"Unlabeled Pool Images: {current_unlabeled_images_pool}")
    print(f"Unlabeled Pool Labels: {current_unlabeled_labels_pool}")

    return nnunet_raw_data_task_dir, current_labeled_images_tr, current_unlabeled_images_pool, current_unlabeled_labels_pool

# --- Example Usage ---
if __name__ == "__main__":
    # Define your paths - REPLACE THESE WITH YOUR ACTUAL PATHS!
    # IMPORTANT: raw_data_source_dir should point to the base of your *original*
    # nnU-Net formatted dataset. So, if your data is in:
    # /my_raw_data/TaskXXX_OriginalDataset/imagesTr/
    # /my_raw_data/TaskXXX_OriginalDataset/labelsTr/
    # Then RAW_DATA_SOURCE should be: /my_raw_data/TaskXXX_OriginalDataset/
    RAW_DATA_SOURCE = "../data/Dataset112_ToothFairy2/"
    NNUNET_BASE = "../data/nnUnet/"
    AL_EXPERIMENT_ROOT = "../data/Dataset112_ToothFairy2/my_al_runs/"

    try:
        task_dir, labeled_tr_dir, unlabeled_img_pool, unlabeled_lbl_pool = setup_nnunet_al_directories(
            raw_data_source_dir=RAW_DATA_SOURCE,
            nnunet_base_dir=NNUNET_BASE,
            al_experiment_root_dir=AL_EXPERIMENT_ROOT,
            task_id=114, # Your desired nnU-Net Task ID for this new experiment
            experiment_name="toothFairy_grouped_classes_al_run",
            test_set_ratio=0.15,
            initial_seed_ratio=0.05,
            random_seed=42,
            image_ext=".mha", # Ensure this matches your original image files
            label_ext=".mha"  # Ensure this matches your original label files
        )
        print("\nSetup successful! Now you can proceed with nnU-Net planning and training.")
        print(f"To verify, run: nnUNetv2_plan_and_preprocess -d {114} --verify_dataset_integrity") # Use the task_id you defined (114)

    except ValueError as e:
        print(f"Error during setup: {e}")
    except FileNotFoundError as e:
        print(f"File/Directory not found error: {e}. Please ensure '{RAW_DATA_SOURCE}/imagesTr' and '{RAW_DATA_SOURCE}/labelsTr' exist and contain data.")