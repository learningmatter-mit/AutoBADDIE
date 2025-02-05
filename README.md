# AutoBADDIE
Autonomous fitting of classical force field parameters

## Installation

`pip install -e .`

Will compile autobaddie into your conda/pip environment. We recommend creating a new environment named autobaddie before running the pip install to avoid conflicts. 

## Use Tutorial


To use the code, please follow along with the below tutorial. Please open and read through the first sections of the 1. create_template.py, 2. assign_params.py, 3. run_full_training.sh files and update necessary paths and variables. Each of these codes begins with a description of inputs and outputs for each code. 

1. Enter the AutoBADDIE/autobaddie directory and update the  **create_template.py** file.
    - The create_template.py code is used to set which atomic environments will be parameterized for a given AutoBADDIE run.

    The user must provide a "template_smiles" string representing a set of molecules that encompass all atomic environments for training run that will be referred to by a "job_name". After a successful run of this code, a labelled image for each atom type such as the one below will be created in AutoBADDIE/training_results/_job\_name_/template/atom_types.png. The Lennard-Jones dispersion parameters for each of these atom types will need to be provided in the next code before training.
   
   <p align="center">
     <img src="publication_additional_files/atom_types_example.png" "width="750" />
   </p>

2. Update and run the **autobaddie/assign_params.py** file.
     - The user must input a "custom_parameter_dict" which houses the Lennard-Jones parameter for each atomic environment enumerated by the "template_smiles" in the create_template.py file above. After a successful run of this code, a csv file named "template.params" will be saved in the AutoBADDIE/training_results/_job\_name_/template directory. AutoBADDIE will use these Lennard-Jones parameters to optimize all other force field parameters.
  
3. Use the run_full_training.sh to take the example training data and learn an AutoBADDIE potential!
      - Update the following 3 variables: **WORKDIR_BASE**, **JOB_NAME**, **DATE**
      - The job_name must match the previous two examples. The date will be used as a general label to separate this particular optimization run from others for the same chemical system described by the "template_smiles". The training_data_path must contain all the atomic positions, forces, partial charges, and pose energies in a format as displayed in the examples provided (such as AutoBADDIE/training_data/class1).

      Once these changes are made, execute the run_full_training.sh code.
   
      If this code is completed successfully, a lammps output for each chemistry escribed in the to_param_list will be saved in the AutoBADDIE/training_results/_job\_name_/date_{EXTRA_TRAINING_DETAILS} directory. Additional information about the training and test performance can be found within this document as well. 
