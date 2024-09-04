import sys, subprocess, os, importlib, shutil, socket, copy
import random, itertools, logging, argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from munch import Munch
import _pickle as pickle
import pandas as pd
import json
import argparse
import self_contained.utils.train as train
import self_contained.utils.parameterize as parameterize
import self_contained.utils.param_compare_utils as param_compare_utils

parser = argparse.ArgumentParser(description="Parse for AuTopology training")
parser.add_argument("--job_name", type=str, action="store", nargs="?")
parser.add_argument("--condition", type=str, action="store", nargs="?")
parser.add_argument("--date", type=str, action="store", nargs="?")
parser.add_argument("--param_json", type=str, action="store", nargs="?")
parser.add_argument("--E_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--dih_reg_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--top_reg_hyp", type=float, default=100.0, nargs="?")
parser.add_argument("--no_train_flag", dest="train_flag", action="store_false")
parser.add_argument("--train_flag", dest="train_flag", action="store_true")
parser.add_argument("--glymecond", type=str, action="store", nargs="?")
parser.add_argument("--tfsicond", type=str, action="store", nargs="?")
parser.add_argument("--licond", type=str, action="store", nargs="?")
parser.add_argument("--training_ff", type=str, action="store", nargs="?")
parser.add_argument("--other_cond_for_dih", type=str, action="store", nargs="?")
parser.add_argument("--dih_label", type=str, action="store", nargs="?")
parser.add_argument("--other_dih_label", type=str, action="store", nargs="?")
parser.add_argument("--self_contained_base", type=str, action="store", nargs="?")
args = parser.parse_args()

train_flag = args.train_flag
SELF_CONTAINED_BASE = args.self_contained_base
job_name = args.job_name
condition = args.condition
print("condition:", condition)

with open(
    f"{SELF_CONTAINED_BASE}/../train/{job_name}/{condition}/job_details.json", "r"
) as jsonFile:
    job_details = json.load(jsonFile)
    try:
        job_details.description
    except:
        job_details["description"] = None
logtext = ""

# begin program
print("beginning the program")
print("job name:", job_details["job_name"])
#             #-----------------------------------------------train and val-----------------------------------------------------------

job_details, condition = train.assign_job_details(job_details, dataset_flag=False)

device = torch.device(job_details.device)
torch.set_default_device(job_details.device)
job_details.mode = "train"

print("condition:", condition)

# getting single datasets
print("recover the datasets")
dataset_path_base = os.path.join(job_details.WORKDIR, condition)
dataset = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset.pkl", "rb")
)
dataset_val = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset_val.pkl", "rb")
)
dataset_test = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset_test.pkl", "rb")
)
dataset, dataset_val, dataset_test = (
    dataset.to(device),
    dataset_val.to(device),
    dataset_test.to(device),
)
single_solv_dataset = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset_solv.pkl", "rb")
)
single_anion_dataset = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset_anion.pkl", "rb")
)
single_cation_dataset = pickle.load(
    open(f"{dataset_path_base}/{condition}_{job_details.mode}_dataset_cation.pkl", "rb")
)
path = os.path.join(job_details.TEMPLATEDIR, "template.py")
template_dataset = torch.load(path)
dataset, dataset_val, dataset_test = (
    dataset.to(device),
    dataset_val.to(device),
    dataset_test.to(device),
)

df = pd.read_csv(
    "/home/pleon/projects/repos/ForceFieldNet/training_data/240830_AuTo_detach_test_from_220928/08302024_train/file_contents.csv"
)

for i, cur_dataset in enumerate(single_solv_dataset):
    single_solv_dataset[i] = cur_dataset.to(device)
for i, cur_dataset in enumerate(single_anion_dataset):
    single_anion_dataset[i] = cur_dataset.to(device)
for i, cur_dataset in enumerate(single_cation_dataset):
    single_cation_dataset[i] = cur_dataset.to(device)

# #begin actual training
for rep in [None]:
    if rep != None:
        job_details.rep = reps
        print("beginning repetition:", rep)
    else:
        job_details.rep = None
    if train_flag:
        if "onlytfsi" in job_details.date:
            single_dataset = single_anion_dataset
        elif "onlyglyme" in job_details.date:
            single_dataset = single_solv_dataset
        else:
            single_dataset = None
        forcefield = train.create_forcefield(
            job_details, condition, dataset, template_dataset, single_dataset
        )
        if not "notrain" in job_details.date:
            # perform training
            print("beginning training")
            # Fs, F_targets, L_val, L_val_MAE are all from the validation set
            (
                forcefield,
                Es,
                E_targets,
                ensembles,
                Fs,
                F_targets,
                ps,
                p_targets,
                geom_ids,
                atomic_num,
                node_types,
                nums,
                L,
                L_MSE,
                L_MAE,
                L_val,
                L_val_MSE,
                L_val_MAE,
                batch,
                logtext,
            ) = train.loop(
                forcefield=forcefield,
                dataset=dataset,
                dataset_val=dataset_val,
                job_details=job_details,
                condition=condition,
                logtext=logtext,
            )

            print("done with training")
            if job_details.num_epoch > 0:
                train.plot_loss_evolution(
                    forcefield, job_details, condition, L_MAE, L_val_MAE
                )
                logtext = train.plot_force_parity(
                    job_details,
                    condition,
                    Fs,
                    F_targets,
                    L_MAE,
                    L_val_MAE,
                    ensembles,
                    geom_ids,
                    atomic_num,
                    node_types,
                    nums,
                    df,
                    logtext,
                    job_details.outlier_cut,
                )
                logtext = train.plot_energy_parity(
                    job_details,
                    condition,
                    Es,
                    E_targets,
                    ensembles,
                    geom_ids,
                    df,
                    logtext,
                )
                if job_details.learn_charge_flag:
                    print("plotting charge evolution")
                    train.plot_charge_evolution(
                        forcefield,
                        job_details,
                        condition,
                        dataset,
                        single_anion_dataset,
                        single_cation_dataset,
                    )
            if "pcff" in job_details.job_name:
                param_compare_utils.plot_param_evolution(
                    job_details, condition, forcefield
                )

        #                     #------------------------------------------------------test-------------------------------------------------------------
        print("\n\nbeginning test")
        job_details.mode = "test"
        print("test condition:", condition)
        # perform test
        (
            forcefield,
            Es,
            E_targets,
            ensembles,
            Fs,
            F_targets,
            ps,
            p_targets,
            geom_ids,
            atomic_num,
            node_types,
            nums,
            L,
            L_MSE,
            L_MAE,
            batch,
            logtext,
        ) = train.loop_test(
            forcefield=forcefield,
            dataset_test=dataset_test,
            job_details=job_details,
            condition=condition,
            logtext=logtext,
        )
        logtext = train.plot_force_parity(
            job_details=job_details,
            condition=condition,
            Fs=Fs,
            F_targets=F_targets,
            L_MAE=L_MAE,
            L_val_MAE=None,
            ensembles=ensembles,
            geom_ids=geom_ids,
            atomic_num=atomic_num,
            node_types=node_types,
            nums=nums,
            df_data=df,
            logtext=logtext,
            outlier_cut=job_details.outlier_cut,
        )
        logtext = train.plot_energy_parity(
            job_details, condition, Es, E_targets, ensembles, geom_ids, df, logtext
        )
        print("done with test\n\n")
        print("logtext", logtext)

    # ------------------------------------------------------parameterize----------------------------------------------------
    job_details.TEMPLATEDIR = os.path.join(
        os.path.abspath("."), job_details.job_name, "template"
    )
    path = os.path.join(job_details.TEMPLATEDIR, "template.py")
    template_dataset = torch.load(path)
    template_dataset = template_dataset.to(device)
    print("to param list:", job_details.to_param_list)
    for to_parameterize, param_name in job_details.to_param_list:
        # param_name is the name of the molecule: ex 'tfsi'
        # to_parameterize is the smiles ex. '[Li+]'

        job_details.to_parameterize = to_parameterize
        job_details.param_name = param_name
        if to_parameterize[-3:] == "xyz":
            param_dataset, rdkit_mol = parameterize.get_dataset_from_xyz(
                to_parameterize,
                job_details,
                condition,
                template_dataset,
                job_details.device,
                return_rdkit=True,
            )
        else:
            try:
                param_dataset, rdkit_mol = parameterize.get_dataset_from_smiles(
                    to_parameterize,
                    job_details,
                    condition,
                    template_dataset,
                    job_details.device,
                    return_rdkit=True,
                )
            except:
                print(to_parameterize)
                import ipdb

                ipdb.set_trace()
        param_dataset = param_dataset.to(device)
        path = os.path.join(
            job_details.WORKDIR, condition, "forcefield_{}.py".format(condition)
        )
        forcefield = torch.load(path)
        # forcefield.to_device(device)
        if "litstart" in job_details.date:
            forcefield.train(device, stop_flag=True)
        else:
            forcefield.train(device)
        print("loaded the forcefield again")
        # calculate forces once to fill the batch._data[TOPOLOGY] dictionaries with the learned parameters
        # (ex. batch._data['bond']['b0'])
        batch = param_dataset.batches[0]
        F = 0.0
        F = F + forcefield.bond(batch, job_details)
        F = F + forcefield.angle(batch, job_details)
        F = F + forcefield.dihedral(batch, job_details)
        F = F + forcefield.improper(batch, job_details)
        F = F + forcefield.pair(batch, job_details)
        scaled_ff, logtext = parameterize.scale_charges_v2(
            job_details,
            to_parameterize,
            param_dataset,
            forcefield,
            condition,
            single_anion_dataset,
            single_cation_dataset,
            single_solv_dataset,
            job_details.anion_smiles,
            job_details.cation_smiles,
            logtext,
        )
        print("param_name:", param_name, "to_parameterize:", to_parameterize)
        parameterize.write_mol_and_pdb_files_for_single_dataset(
            job_details, condition, param_name, rdkit_mol
        )
        print("going to write data file")
        logtext = parameterize.write_lammps_data_files(
            param_name, job_details, condition, batch, forcefield, scaled_ff, logtext
        )

        # perform  with scaled data
        job_details.mode = "test"
        (
            scaled_ff,
            Es,
            E_targets,
            ensembles,
            Fs,
            F_targets,
            ps,
            p_targets,
            geom_ids,
            atomic_num,
            node_types,
            nums,
            L,
            L_MSE,
            L_MAE,
            batch,
            logtext,
        ) = train.loop_test(
            forcefield=scaled_ff,
            dataset_test=dataset_test,
            job_details=job_details,
            condition=condition,
            logtext=logtext,
        )

        # logtext = train.plot_force_parity(job_details=job_details, condition=condition, Fs=Fs, F_targets=F_targets,
        #                                     L_MAE=L_MAE, L_val_MAE=None, ensembles=ensembles, geom_ids=geom_ids, atomic_num=atomic_num,
        #                                     node_types=node_types, nums=nums, logtext=logtext, outlier_cut=job_details.outlier_cut, scaled_flag=True)
        # logtext = train.plot_energy_parity(job_details, condition, Es, E_targets, ensembles, geom_ids, logtext, scaled_flag=True)
        print("done with test\n\n")
        print("logtext", logtext)

    print("done with parameterization\n\n\n\n")
    # if "pcff" in job_details.job_name:
    #     param_compare_utils.param_compare(job_details, condition)
    #     print("done with analysis")
