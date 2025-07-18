# Active Learning Pipeline with working nnU-Net training/prediction steps
# Paths and parameters based on your functional version

import os
import shutil
import subprocess
import numpy as np
import random
from glob import glob
import json
import calculate_uncertainty_scores
import sys
import math

RANDOM_SEED = 2025
random.seed(RANDOM_SEED)

# sampling_methods = ["random_sampling", "least_confidence_sampling"]
# sampling_tasks = [115, 116]

sampling_methods = ["random_sampling"]
sampling_tasks = [115]

cluster = "gcp"  # or "gcp"

# ---- PATHS ----
if cluster == "gcp":
    RAW_UNLABELLED_FOLDER = "../data/Dataset112_ToothFairy2/my_al_runs/run_toothFairy_grouped_classes_al_run"
    NNUNET_RAW = "../data/nnUnet/nnUNet_raw"
    NNUNET_RESULTS = "../data/nnUnet/nnUNet_results"
    NNUNET_RESULTS_PROCESSED = "../data/nnUnet/nnUNet_resultsprocessed"
    PATH_TO_DATA = "../data"
elif cluster == "lig":
    RAW_UNLABELLED_FOLDER = "../../../data2/toothfairy/Dataset112_ToothFairy2/my_al_runs/"
    NNUNET_RAW = "../../../data2/nnUnet/nnUNet_raw"
    NNUNET_RESULTS = "../../../data2/nnUnet/nnUNet_results"
    NNUNET_RESULTS_PROCESSED = "../../../data2/nnUnet/nnUNet_resultsprocessed"
    PATH_TO_DATA = "../../../data2/"

NNUNET_MODEL_CONFIGURATION = "3d_lowres"

AL_ROUNDS = 10
ACQUISITION_BUDGET_PER_ROUND = 5
INITIAL_SEED_RATIO = 0.05
TEST_SET_RATIO = 0.15
MAX_TOTAL_LABELED_VOLUMES = None

IMAGE_EXT = ".mha"
LABEL_EXT = ".mha"


def run_nnunet_command(command, cwd=None, env=None):
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(command, cwd=cwd, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with return code {return_code}")


def disable_next_stage_in_plan(plans_json_path):
    with open(plans_json_path, 'r') as f:
        plans = json.load(f)

    # Vérifier si la clé 'configurations' et '3d_lowres' existent
    if "configurations" in plans and "3d_lowres" in plans["configurations"]:
        if "next_stage" in plans["configurations"]["3d_lowres"]:
            print(f"Original next_stage: {plans['configurations']['3d_lowres']['next_stage']}")
            # Supprimer la clé next_stage
            del plans["configurations"]["3d_lowres"]["next_stage"]
            print("Clé 'next_stage' supprimée dans la configuration '3d_lowres'.")
        else:
            print("Clé 'next_stage' non trouvée dans '3d_lowres'. Rien à faire.")
    else:
        print("Structure 'configurations' -> '3d_lowres' non trouvée. Vérifie le fichier JSON.")

    # Réécrire le fichier modifié
    with open(plans_json_path, 'w') as f:
        json.dump(plans, f, indent=4)


def prepare_working_unlabeled_pool(original_unlabeled_dir, working_dir):
    if not os.path.exists(working_dir):
        print("Copying unlabeled images pool once...")
        shutil.copytree(original_unlabeled_dir, working_dir)
    else:
        print("Working unlabeled pool already exists — skipping copy.")


def remove_selected_images_from_pool(working_dir, selected_ids, image_ext=".nii.gz"):
    for patient_id in selected_ids:
        filename = f"{patient_id}_0000{image_ext}"
        filepath = os.path.join(working_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print(f"Warning: {filepath} not found — maybe already removed?")


def copy_patient_to_training(patient_id, image_src_dir, label_src_dir, dst_dir, image_ext=".mha"):
    shutil.copy(
        os.path.join(image_src_dir, f"{patient_id}_0000{image_ext}"),
        os.path.join(dst_dir, "imagesTr", f"{patient_id}_0000{image_ext}")
    )
    shutil.copy(
        os.path.join(label_src_dir, f"{patient_id}{image_ext}"),
        os.path.join(dst_dir, "labelsTr", f"{patient_id}{image_ext}")
    )            


# --- ACTIVE LEARNING MAIN LOOP ---
for sampling_name, sampling_number in zip(sampling_methods, sampling_tasks):
    print("###################", sampling_name, "###################")

    new_experiment_folder = os.path.join(NNUNET_RAW, f"Dataset{sampling_number}_{sampling_name}_toothfairy")
    if os.path.isdir(new_experiment_folder):
        shutil.rmtree(new_experiment_folder)
    shutil.copytree(os.path.join(NNUNET_RAW, "Dataset114_toothFairy_grouped_classes_al_run"), new_experiment_folder)
    shutil.copytree(os.path.join(NNUNET_RESULTS, "Dataset114_toothFairy_grouped_classes_al_run"),
                    os.path.join(NNUNET_RESULTS, f"Dataset{sampling_number}_{sampling_name}_toothfairy"))

    all_image_files_original = glob(os.path.join(RAW_UNLABELLED_FOLDER, "unlabeled_pool_images", f"*{IMAGE_EXT}"))
    all_patient_ids = sorted([os.path.basename(f).replace(f"_0000{IMAGE_EXT}", "") for f in all_image_files_original])
    if not all_patient_ids:
        raise ValueError(f"No image files found in {os.path.join(new_experiment_folder, 'imagesTr')}")

    remaining_patients = all_patient_ids.copy()

    # Répertoire où tu veux sauvegarder tes checkpoints versionnés
    checkpoint_backup_dir = os.path.join(NNUNET_RESULTS, "checkpoint_backups")
    os.makedirs(checkpoint_backup_dir, exist_ok=True)

    selected_ids_total = []
    initial_ids = sorted([
        os.path.basename(f).replace(f"_0000{IMAGE_EXT}", "")
        for f in glob(os.path.join(new_experiment_folder, "imagesTr", f"*{IMAGE_EXT}"))
    ])
    selected_ids_total = [initial_ids]    
    # selected_ids_total.append(glob(os.path.join(new_experiment_folder, "imagesTr", f"*{IMAGE_EXT}")))  #append the already annotated images

    if sampling_name == "least_confidence_sampling":
        # create the folder copy of unannotated images only for leat confidence which need to perform prediction on it
        tmp_unlabeled_dir = os.path.join(PATH_TO_DATA, "toothfairy/tmp")
        prepare_working_unlabeled_pool(os.path.join(RAW_UNLABELLED_FOLDER, "unlabeled_pool_images"), tmp_unlabeled_dir)

    for iteration in range(AL_ROUNDS):
        print(f"--- Iteration {iteration} ---")

        if sampling_name == "random_sampling":
            selected_ids = [remaining_patients.pop(random.randrange(len(remaining_patients)))
                            for _ in range(ACQUISITION_BUDGET_PER_ROUND)]

        elif sampling_name == "least_confidence_sampling":

            # Liste de toutes les images candidates
            all_tmp_images = sorted(glob(os.path.join(tmp_unlabeled_dir, f"*{IMAGE_EXT}")))
            print(f"Number of unannotated images: {len(all_tmp_images)}")

            # Paramètre : taille du batch pour éviter surcharge mémoire
            BATCH_SIZE = 25

            output_prediction_path_unnatotated = os.path.join(
                NNUNET_RESULTS, "predictions", f"Dataset{sampling_number}_{sampling_name}_toothfairy",
                f'predictions_on_unlabeled_iteration{iteration}'
            )
            os.makedirs(output_prediction_path_unnatotated, exist_ok=True)

            print(f"\nRunning batched nnUNetv2_prediction for iteration {iteration}")
            print("################################################################")

            # Lancer les prédictions par batch
            for batch_idx in range(math.ceil(len(all_tmp_images) / BATCH_SIZE)):
                print(f"--- Batch {batch_idx + 1}/{math.ceil(len(all_tmp_images) / BATCH_SIZE)} ---")

                batch_files = all_tmp_images[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

                # Crée un dossier temporaire pour les fichiers batch
                batch_tmp_dir = os.path.join(tmp_unlabeled_dir, f"batch_tmp_{batch_idx}")
                os.makedirs(batch_tmp_dir, exist_ok=True)

                for f in batch_files:
                    shutil.copy(f, os.path.join(batch_tmp_dir, os.path.basename(f)))

                # Appel nnUNetv2_predict sur le batch
                run_nnunet_command([
                    "nnUNetv2_predict",
                    "-i", batch_tmp_dir,
                    "-o", output_prediction_path_unnatotated,
                    "-d", str(sampling_number),
                    "-f", "0",
                    "-c", NNUNET_MODEL_CONFIGURATION,
                    "-chk", "checkpoint_best.pth",
                    "-device", "cuda",
                    "--save_probabilities",
                    "--disable_tta"
                ])

                shutil.rmtree(batch_tmp_dir)  # Nettoyage du dossier temporaire

            # Une fois toutes les prédictions faites, calcul des scores
            lc_scores = calculate_uncertainty_scores.calculate_uncertainty_scores(
                output_prediction_path_unnatotated,
                method="least_confidence"
            )

            print(lc_scores)

            # Ne garder que les patients encore éligibles
            lc_scores = {k: v for k, v in lc_scores.items() if k in remaining_patients}

            # Trier par incertitude décroissante
            sorted_patients = sorted(lc_scores.items(), key=lambda x: x[1], reverse=True)

            # Sélectionner les plus incertains
            selected_ids = [patient_id for patient_id, _ in sorted_patients[:ACQUISITION_BUDGET_PER_ROUND]]
            # selected_ids_total.append(selected_ids)

            remove_selected_images_from_pool(tmp_unlabeled_dir, selected_ids, image_ext=IMAGE_EXT)


        else:
            print("sampling method not implemented")
            sys.exit(1)

        selected_ids_total.append(selected_ids)

        for selected_id in selected_ids:
            copy_patient_to_training(selected_id,
                                    os.path.join(RAW_UNLABELLED_FOLDER, 'unlabeled_pool_images'),
                                    os.path.join(RAW_UNLABELLED_FOLDER, 'unlabeled_pool_labels'),
                                    new_experiment_folder,
                                    image_ext=IMAGE_EXT)

        # for selected_id in selected_ids:
        #     shutil.copy(os.path.join(RAW_UNLABELLED_FOLDER, 'unlabeled_pool_images', f"{selected_id}_0000{IMAGE_EXT}"),
        #                 os.path.join(new_experiment_folder, 'imagesTr', f"{selected_id}_0000{IMAGE_EXT}"))
        #     shutil.copy(os.path.join(RAW_UNLABELLED_FOLDER, 'unlabeled_pool_labels', f"{selected_id}{IMAGE_EXT}"),
        #                 os.path.join(new_experiment_folder, 'labelsTr', f"{selected_id}{IMAGE_EXT}"))

        nb_of_annotated_img = len(glob(os.path.join(new_experiment_folder, "imagesTr", f"*{IMAGE_EXT}")))
        print("nb images annotated:", nb_of_annotated_img)

        path_dataset_json = os.path.join(new_experiment_folder, "dataset.json")
        with open(path_dataset_json, "r") as jsonFile:
            data = json.load(jsonFile)
        data["numTraining"] = nb_of_annotated_img
        with open(path_dataset_json, "w") as jsonFile:
            json.dump(data, jsonFile)

        split_path = os.path.join(NNUNET_RESULTS_PROCESSED, f"Dataset{sampling_number}_{sampling_name}_toothfairy", "splits_final.json")
        if os.path.exists(split_path):
            os.remove(split_path)

        print(f"\nRunning nnUNetv2_plan_and_preprocess for {sampling_number}...")
        print("################################################################")
        try:
            run_nnunet_command([
                "nnUNetv2_plan_and_preprocess",
                "-d", str(sampling_number),
                "-c", NNUNET_MODEL_CONFIGURATION,
                "--verify_dataset_integrity"
            ])
        except Exception as e:
            print(f"CRITICAL ERROR in planning: {e}")
            continue

        disable_next_stage_in_plan(os.path.join(NNUNET_RESULTS_PROCESSED, f"Dataset{sampling_number}_{sampling_name}_toothfairy", "nnUNetPlans.json"))

        print(f"\nRunning nnUNetv2_train for {sampling_number}...")
        print("################################################################")
        if iteration == 0:
            path_best_checkpoint = os.path.join(
                NNUNET_RESULTS, 
                'Dataset114_toothFairy_grouped_classes_al_run/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth'
            )
        else:
            path_best_checkpoint = os.path.join(
                NNUNET_RESULTS,
                f"Dataset{sampling_number}_{sampling_name}_toothfairy",
                'nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth'
            )

        try:
            run_nnunet_command([
                "nnUNetv2_train",
                "-pretrained_weights", path_best_checkpoint,
                "-device=cuda",
                str(sampling_number),
                NNUNET_MODEL_CONFIGURATION,
                "0"
            ])
        except Exception as e:
            print(f"CRITICAL ERROR in training: {e}")
            continue

        # Chemin où on sauvegarde ce checkpoint avec la version itération
        backup_checkpoint_path = os.path.join(checkpoint_backup_dir, f"checkpoint_best_iter_{iteration}.pth")

        # ensure to take the last best checkpoint
        path_best_checkpoint = os.path.join(NNUNET_RESULTS, f"Dataset{sampling_number}_{sampling_name}_toothfairy", 'nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_best.pth')
        
        # Copier le checkpoint vers le dossier de sauvegarde
        shutil.copy(path_best_checkpoint, backup_checkpoint_path)            

        print(f"\nRunning nnUNetv2_predict for {sampling_number}...")
        print("################################################################")
        try:
            output_prediction_path = os.path.join(new_experiment_folder, f'predictions_on_test_iteration{iteration}')
            os.makedirs(output_prediction_path, exist_ok=True)
            run_nnunet_command([
                "nnUNetv2_predict",
                "-i", os.path.join(new_experiment_folder, "imagesTs"),
                "-o", output_prediction_path,
                "-d", str(sampling_number),
                "-f", "0",
                "-c", NNUNET_MODEL_CONFIGURATION,
                "-chk", "checkpoint_best.pth",
                "-device", "cuda"
            ])
        except Exception as e:
            print(f"CRITICAL ERROR in prediction: {e}")
            continue
