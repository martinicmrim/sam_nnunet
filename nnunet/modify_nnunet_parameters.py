# pip show nnunetv2

import fileinput
import sys

def modify_trainer_config(filepath, new_num_epochs=250, new_nb_iteration=0.001):
    found_epochs = False
    found_iteration = False
    
    # fileinput.input permet de lire le fichier et d'écrire sur place
    # inplace=True : modifie le fichier original
    # backup='.bak' : crée une sauvegarde
    with fileinput.input(filepath, inplace=True, backup='.bak') as f:
        for line in f:
            # sys.stdout.write écrit dans le fichier (car inplace=True)
            if "self.num_epochs =" in line and not found_epochs:
                indentation = line.split("self.num_epochs =")[0]
                sys.stdout.write(f"{indentation}self.num_epochs = {new_num_epochs}\n")
                found_epochs = True
            elif "self.num_iterations_per_epoch =" in line and not found_iteration:
                indentation = line.split("self.num_iterations_per_epoch =")[0]
                sys.stdout.write(f"{indentation}self.num_iterations_per_epoch = {new_nb_iteration}\n")
                found_iteration = True
            else:
                sys.stdout.write(line) # Écrire la ligne originale si elle n'est pas modifiée

    if not found_epochs:
        print(f"Warning: 'self.num_epochs' not found or already modified in {filepath}")
    if not found_iteration:
        print(f"Warning: 'self.initial_lr' not found or already modified in {filepath}")
    print(f"Configuration of {filepath} updated.")

# --- Utilisation ---
if __name__ == "__main__":
    NNUNET_TRAINER_FILE = "/opt/conda/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py" 
    modify_trainer_config(NNUNET_TRAINER_FILE, new_num_epochs=50, new_nb_iteration=50)