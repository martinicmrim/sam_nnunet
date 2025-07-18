# sam_nnunet
Experiments on combination of SAM and nnU-net

# TO DO 
- Improve comments in code + Path configuration
- Create script for automatic training / evaluation

# Description
The code is composed of two independant part due to incompatible requirements between nnU-net and MedSAM-3D.

## nnUnet 
1. Generate the datasplit + group the classes (generate_df_AL.py)
2. Configuration of nnU-Net (nn-Unet paths + modify_nnunet_parameters.py)
3. Train a first model (cold-start) - This step is currently made manually
4. Run the AL experiment (AL_experiments.py)
5. Evaluate the prediction (currently manually done) (function nnUNetv2_evaluate_folder)

## medSAM
1. Run the evaluation: medim_val_folder_mha_loop.py

