import os
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from munch import Munch
import _pickle as pickle
import pandas as pd
from rdkit.Chem import AllChem as Chem
from sklearn.model_selection import train_test_split
import scipy.stats
import self_contained.utils.training_utils as training_utils
import self_contained.forcefields.Forcefield as Forcefield

import self_contained.utils.param_compare_utils as param_compare_utils
from self_contained.utils.constants import ELEMENT_TO_NUM

import json
from self_contained.forcefields.Forcefield_pcff_linHar_nocrossnoanhar import (
    MIN_BOND_DIS,
    MAX_BOND_DIS,
    MIN_ANGLE_DIS,
    MAX_ANGLE_DIS,
)
from materialbuilder.matbuilder import Z_TO_SYMBOLS

HARTREE_TO_KCALMOL = 627.50947415
AU_TO_ANGSTROM = 1 / 0.52917721090380
D_TO_eA = 0.2081943


def assign_job_details(job_details, dataset_flag=True):
    job_details = Munch(job_details)

    if job_details.dropout_rate != 0:
        job_details.date = (
            job_details.date + "_d" + str(job_details.dropout_rate).split(".")[-1]
        )
    if "set_charge" in job_details.job_name or job_details.generation == "gen0":
        job_details.learn_charge_flag = False
    else:
        job_details.learn_charge_flag = True
    print("learn charge flag:", job_details.learn_charge_flag)
    print("moving to this directory:", job_details.train_autopath)
    os.chdir(job_details.train_autopath)
    if job_details.creg_hyp:
        condition = "{}_a{}_m{}_creg{}_i{}_LR_{}".format(
            job_details.date,
            job_details.atom_hyp,
            job_details.mol_hyp,
            job_details.creg_hyp,
            str(job_details.ion_sc_hyp)[-1],
            str(job_details.learning_rate).split(".")[-1],
        )
    else:
        condition = "{}_a{}_m{}_i{}_LR_{}".format(
            job_details.date,
            job_details.atom_hyp,
            job_details.mol_hyp,
            str(job_details.ion_sc_hyp)[-1],
            str(job_details.learning_rate),
        )
    if job_details.generation:
        condition = "_".join([job_details.generation, condition])
    job_details.TEMPLATEDIR = os.path.join(
        os.path.abspath("."), job_details.job_name, "template"
    )
    job_details.WORKDIR = os.path.join(os.path.abspath("."), job_details.job_name)
    path = os.path.join(job_details.WORKDIR, condition)

    if dataset_flag:
        if not os.path.exists(path):
            os.makedirs(path)
        if job_details.description:
            with open(os.path.join(path, "README"), "w") as READMEFile:
                READMEFile.write(job_details.description)
            del job_details.description
        with open(os.path.join(path, "job_details.json"), "w") as jsonFile:
            json.dump(job_details, jsonFile, indent=1)
    else:
        with open(os.path.join(path, "job_details.json"), "r") as jsonFile:
            job_details = json.load(jsonFile)
            job_details = Munch(job_details)
    return job_details, condition


def create_forcefield(
    job_details, condition, dataset, template_dataset=None, single_dataset=None
):
    # Create forcefield object that will contain the learned parameters
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            "forcefield_{}_rep{}.py".format(condition, job_details.rep),
        )
    else:
        path = os.path.join(
            job_details.WORKDIR, condition, "forcefield_{}.py".format(condition)
        )
    phi0_flag = False
    forcefield = Forcefield.ForceField(dataset, job_details, phi0_flag=phi0_flag)
    print("inside the train create_forcefield.py")
    path = os.path.join(
        job_details.WORKDIR, condition, "forcefield_{}.py".format(condition)
    )
    torch.save(forcefield, path)
    forcefield.train(job_details.device)
    return forcefield


def calc_multipoles(batch, job_details, forcefield):
    # get cur multipoles
    types = batch.get_data("node", "type")
    PARAM = get_param_from_ff(forcefield, batch)
    qs = PARAM["charge"][types].squeeze(1)
    multipole = qs * batch.get_data("node", "xyz")
    c = batch.get_data("node", "component").to(torch.long)
    c = list(c.split(batch.get_data("num", "node").view(-1).tolist()))
    for i in range(len(c)):
        c[i] = c[i].unique(return_counts=True)[1]
    c = torch.cat(c).tolist()

    multipole = list(multipole.split(c))
    for i in range(len(multipole)):  # This adds to get the total charge per molecule
        multipole[i] = multipole[i].sum(0)
    multipole = torch.stack(multipole).detach().cpu()
    multipole_target = (
        torch.cat(
            (
                batch.get_data("prop", "multipole_x"),
                batch.get_data("prop", "multipole_y"),
                batch.get_data("prop", "multipole_z"),
            ),
            dim=1,
        )
        .detach()
        .cpu()
    )
    return multipole, multipole_target


def calc_forces(batch, job_details, forcefield, train_flag=False):
    F = 0.0
    forcefield.optimizer.zero_grad()
    F1 = forcefield.bond(batch, job_details, train_flag)
    F2 = forcefield.angle(batch, job_details, train_flag)
    F3 = forcefield.dihedral(batch, job_details, train_flag)
    # F4 = forcefield.improper(batch, job_details, train_flag) * 0
    F4 = 0
    F5 = forcefield.pair(batch, job_details)
    F = F1 + F2 + F3 + F5
    F_target = batch.get_data("node", "F")
    return F, F_target, forcefield


def calc_energies(batch, job_details, forcefield, train_flag=False):
    E = 0.0
    E1 = forcefield.bond(batch, job_details, train_flag, energy_flag=True)
    E2 = forcefield.angle(batch, job_details, train_flag, energy_flag=True)
    E3 = forcefield.dihedral(batch, job_details, train_flag, energy_flag=True)
    # E4 = forcefield.improper(batch, job_details, train_flag, energy_flag=True) * 0
    E4 = 0
    E5 = forcefield.pair(batch, job_details, energy_flag=True)
    E = E1 + E2 + E3 + E4 + E5
    E_target = batch.get_data("prop", "energy")
    ensemble = batch.get_data("prop", "ensemble")
    return E, E_target, ensemble


def calc_loss(forcefield, F, F_target, batch, job_details):
    MSE = (F - F_target).pow(2).mean()
    loss = MSE
    if job_details.learn_charge_flag:
        loss = loss + forcefield.get_charge_loss(batch, job_details)
    if job_details.reg_hyp:
        reg_MSE = calc_dihedral_regularization(forcefield, job_details, batch)
        loss = loss + reg_MSE
    if job_details.top_reg_hyp:
        reg_MSE = calc_topology_regularization(forcefield, job_details)
        loss = loss + reg_MSE
    if job_details.energy_hyp:
        loss = loss + calc_energy_loss(forcefield, job_details, batch)
    MAE = (F - F_target).abs().mean()
    return loss, MSE, MAE, forcefield


def calc_dihedral_regularization(forcefield, job_details, batch):
    ks = forcefield.get_dihedral_params(job_details)
    if "pcff" in job_details.job_name:
        # ks = K1, K2, K3, K_MT1, K_MT2, K_MT3, K_BCT10, K_BCT20, K_BCT30, K_BCT11, K_BCT21, K_BCT31, K_AT10, K_AT20, K_AT30, K_AT11, K_AT21, K_AT31, K_AAT, K_B1B3
        # let's only normalize the K1, K2, K3 so 0:3
        norm = torch.linalg.norm(torch.cat(ks[:3]), 1)
        crossnorm = torch.cat(ks[6:18]).pow(2).sum()

        reg_MSE = job_details.reg_hyp * (norm + crossnorm)
    else:
        norm = torch.linalg.norm(torch.cat(ks[:4]), 1)
        reg_MSE = job_details.reg_hyp * norm
    return reg_MSE


def calc_energy_loss(forcefield, job_details, batch):
    # E_target in kcal/mol, E is...
    E, E_target, ensembles = calc_energies(batch, job_details, forcefield)
    MSE = 0.0
    # multiply by len(E_cur) to get the mean weighted by the number of species of cur ensemble
    for ensemble in ensembles.unique().tolist():
        E_cur, E_tar_cur = E[ensembles == ensemble], E_target[ensembles == ensemble]
        cur_loss = ((E_cur - E_cur.mean()) - (E_tar_cur - E_tar_cur.mean())).pow(
            2
        ).mean() * len(E_cur)
        MSE = MSE + cur_loss
    # divide MSE by the total number of training data in batch to get the actual MSE (len(E) = sum(len(E_cur)))
    MSE = MSE / len(E)
    loss = job_details.energy_hyp * MSE
    return loss


def calc_topology_regularization(forcefield, job_details):
    if "b0" in forcefield.__dict__.keys():
        norm = (forcefield.b0_tar - forcefield.b0.pow(2)).pow(2).mean()
        norm = norm + (forcefield.theta0_tar - forcefield.theta0.pow(2)).pow(2).mean()
    elif "k_b21" in forcefield.__dict__.keys():
        # if the parameter is not used, it is set to nan
        B0 = (forcefield.k_b21 * MIN_BOND_DIS + forcefield.k_b22 * MAX_BOND_DIS) / (
            forcefield.k_b21 + forcefield.k_b22
        )
        THETA0 = (
            forcefield.k_a21 * MIN_ANGLE_DIS + forcefield.k_a22 * MAX_ANGLE_DIS
        ) / (forcefield.k_a21 + forcefield.k_a22)
        norm = (forcefield.b0_tar - B0).pow(2).mean()
        norm = norm + (forcefield.theta0_tar - THETA0).pow(2).mean()
    reg_MSE = job_details.top_reg_hyp * norm
    return reg_MSE


def scale_down_solvent_charges(
    train_df, val_df, test_df, single_solv_dataset, solv_scale, other_datasets=None
):
    # get the unique atom types that correspond to the solvent
    solv_types = []
    for cur_solv_dataset in single_solv_dataset:
        solv_types.append(cur_solv_dataset._data["node"]["type"].unique().to("cpu"))
    if other_datasets:
        for cur_dataset in other_datasets:
            solv_types.append(cur_dataset._data["node"]["type"].unique().to("cpu"))
    solv_types = torch.concat(solv_types)
    # scale down the charges of all the atoms in the training,val,and test data corresponding to the solvent
    print("distinct charges in df:", train_df.batches[0].get_data("node", "q").unique())
    for cur_df in [train_df, val_df, test_df]:
        cur_df_types = cur_df._data["node"]["type"]
        solv_mask = torch.isin(cur_df_types, solv_types)
        cur_df._data["node"]["q"][solv_mask] = (
            solv_scale * cur_df._data["node"]["q"][solv_mask]
        )
        # also scale down the charges that are in the batches:
        for curbatch in cur_df.batches:
            cur_df_types = curbatch._data["node"]["type"]
            solv_mask = torch.isin(cur_df_types, solv_types)
            curbatch._data["node"]["q"][solv_mask] = (
                solv_scale * curbatch._data["node"]["q"][solv_mask]
            )
    print(
        "distinct charges in df after scaling:",
        train_df.batches[0].get_data("node", "q").unique(),
    )
    return train_df, val_df, test_df


def reorder_cluster_atoms(rdkit_mol, z):
    # Retreive atomic order from a random parsed geometry
    atom_numbers_true = z
    split_smiles = Chem.MolToSmiles(rdkit_mol).split(".")
    mols = [Chem.MolFromSmiles(i) for i in split_smiles]
    mols = [Chem.AddHs(i) for i in mols]
    atomic_nums_separate = [[x.GetAtomicNum() for x in i.GetAtoms()] for i in mols]
    hydrogen_dict = {}
    for ind, i in enumerate(atomic_nums_separate):
        if 1 in i:
            hydrogen_dict[ind] = (i.index(1), len(i) - i.index(1))
        else:
            hydrogen_dict[ind] = (len(i), 0)

    # Compare and reorder the smiles so that both match
    order_dict = {}
    k = 0
    idx = 0
    while idx < len(atom_numbers_true):
        for ind, sub_list in enumerate(atomic_nums_separate):
            if atom_numbers_true[idx : idx + len(sub_list)] == sub_list:
                order_dict[k] = ind
                k += 1
                idx += len(sub_list)
    smiles_reordered = []
    for _, val in order_dict.items():
        smiles_reordered.append(split_smiles[val])

    # Create atomic numbers, atomic indicies, neighbors and bonds in the new, correct order
    mols = [Chem.MolFromSmiles(i) for i in smiles_reordered]
    mols = [Chem.AddHs(i) for i in mols]
    atomic_nums = [x.GetAtomicNum() for i in mols for x in i.GetAtoms()]
    atom_idxs = [x for x, _ in enumerate(atomic_nums)]

    # Really niche lines of code to get the indeces for each separate molecule in a cluster as a list of lists
    x = 0
    atom_idxs_separate = []
    for i in mols:
        idxs = []
        for j in i.GetAtoms():
            idxs.append(x)
            x += 1
        atom_idxs_separate.append(idxs)
    # Fixing the indicies of the separate mols
    neighbors_separate = [
        [[x.GetIdx() for x in y.GetNeighbors()] for y in mol.GetAtoms()] for mol in mols
    ]
    curr_len_neigh = len(neighbors_separate[0])
    for ind_1, neigh in enumerate(neighbors_separate[1:]):
        for ind_2, atom in enumerate(neigh):
            if atom:
                for ind_3, each in enumerate(atom):
                    neighbors_separate[ind_1 + 1][ind_2][ind_3] += curr_len_neigh
        curr_len_neigh += len(neighbors_separate[ind_1 + 1])

    neighbors = []
    for i in neighbors_separate:
        neighbors.extend(i)
    adjmat = np.zeros((len(neighbors), len(neighbors)))
    for i, cur_atom in enumerate(neighbors):
        for cur_neigh in cur_atom:
            adjmat[i][cur_neigh] = 1
    N = [len(Chem.GetAdjacencyMatrix(cur).tolist()) for cur in mols]
    return list(adjmat), N, atomic_nums


def create_train_and_val_df_file(
    path, job_details, condition, template_dataset, charge=False
):
    (
        reference_adjmat,
        reference_z,
        N,
        Qs,
        poles,
        energies,
        xyzs,
        forces,
        permutation,
        geom_ids,
    ) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    Qs_val, poles_val, energies_val, xyzs_val, forces_val, geom_ids_val = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    Qs_test, poles_test, energies_test, xyzs_test, forces_test, geom_ids_test = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    df = pd.read_csv(f"{path}/file_contents.csv")
    species_ids = df.species_id.unique()
    for species_id in species_ids:
        print("newsmiles:", df[df.species_id == species_id].smiles.to_list()[0])
        smiles = df[df.species_id == species_id].smiles.to_list()[0]
        # use the rdkit default adjmat and atom order'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]'
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        adjmat = Chem.GetAdjacencyMatrix(rdkit_mol).tolist()
        z_rdkit = [cur.GetAtomicNum() for cur in rdkit_mol.GetAtoms()]
        curN = [
            len(Chem.GetAdjacencyMatrix(Chem.AddHs(Chem.MolFromSmiles(cur))).tolist())
            for cur in smiles.split(".")
        ]
        with open(
            f"{path}/{species_id}/{df[df.species_id == species_id].geom_id.unique()[0]}.xyz",
            "r",
        ) as f:
            lines = f.readlines()
            z = []
            for curline in lines[2:]:
                z.append(ELEMENT_TO_NUM[curline[:2].strip()])

        if z_rdkit != z:
            print("uh oh!")
            adjmat, curN, z_rdkit = reorder_cluster_atoms(rdkit_mol, z)
            print(
                "REORDERING: The training data files must be in the same order as RDKIT, or the adjacency matrix must be provided"
            )

        reference_adjmat[species_id] = adjmat
        reference_z[species_id] = z_rdkit
        N[species_id] = curN
        all_geomids = df[df.species_id == species_id].geom_id.unique()
        all_xyzs, all_forces, all_Qs, all_energies = [], [], [], []
        for ind_b, _id in enumerate(all_geomids):
            # xyz
            with open(f"{path}/{species_id}/{_id}.xyz", "r") as f:
                lines = f.readlines()
                geom = []
                for ind, i in enumerate(lines[2:]):
                    geom.append(
                        [z_rdkit[ind]]
                        + list(np.fromstring(i[2:-1], dtype=np.float64, sep=" "))
                    )
            all_xyzs.append(geom)
            # forces
            with open(f"{path}/{species_id}/{_id}.force", "r") as f:
                lines = f.readlines()
                geom = []
                for ind, i in enumerate(lines[2:]):
                    geom.append(list(np.fromstring(i[2:-1], dtype=np.float64, sep=" ")))
            all_forces.append(geom)

            # charges
            if charge:
                with open(f"{path}/{species_id}/{_id}.charge", "r") as f:
                    lines = f.readlines()
                    geom = []
                    for ind, i in enumerate(lines[2:]):
                        geom.append(
                            list(np.fromstring(i[2:-1], dtype=np.float64, sep=" "))
                        )
                all_Qs.append(geom)

        df_smol = df[df["smiles"] == smiles]
        all_xyzs = torch.tensor(all_xyzs).to(job_details.device)
        all_forces = torch.tensor(all_forces).to(job_details.device)
        all_Qs = torch.tensor(all_Qs).to(job_details.device)
        all_energies = torch.tensor(df_smol.energy.to_numpy(), dtype=torch.float32).to(
            job_details.device
        )  # make sure kcal/mol
        # split into train-test-val
        (
            all_forces,
            all_forces_test,
            all_xyzs,
            all_xyzs_test,
            all_Qs,
            all_Qs_test,
            all_energies,
            all_energies_test,
            all_geomids,
            all_geomids_test,
        ) = train_test_split(
            all_forces,
            all_xyzs,
            all_Qs,
            all_energies,
            all_geomids,
            test_size=job_details.train_val_test_split[2],
        )
        (
            all_forces,
            all_forces_val,
            all_xyzs,
            all_xyzs_val,
            all_Qs,
            all_Qs_val,
            all_energies,
            all_energies_val,
            all_geomids,
            all_geomids_val,
        ) = train_test_split(
            all_forces,
            all_xyzs,
            all_Qs,
            all_energies,
            all_geomids,
            test_size=job_details.train_val_test_split[1]
            / (
                job_details.train_val_test_split[0]
                + job_details.train_val_test_split[1]
            ),
        )
        (
            xyzs[species_id],
            forces[species_id],
            Qs[species_id],
            energies[species_id],
            geom_ids[species_id],
        ) = (all_xyzs, all_forces, all_Qs, all_energies, all_geomids)
        (
            xyzs_test[species_id],
            forces_test[species_id],
            Qs_test[species_id],
            energies_test[species_id],
            geom_ids_test[species_id],
        ) = (
            all_xyzs_test,
            all_forces_test,
            all_Qs_test,
            all_energies_test,
            all_geomids_test,
        )
        (
            xyzs_val[species_id],
            forces_val[species_id],
            Qs_val[species_id],
            energies_val[species_id],
            geom_ids_val[species_id],
        ) = (
            all_xyzs_val,
            all_forces_val,
            all_Qs_val,
            all_energies_val,
            all_geomids_val,
        )

    print("beginning to create the datasets")
    dataset = training_utils.create_dataset(
        template_dataset,
        job_details,
        xyzs,
        reference_adjmat,
        N,
        forces,
        Qs,
        poles,
        energies,
        geom_ids,
    )
    dataset_val = training_utils.create_dataset(
        template_dataset,
        job_details,
        xyzs_val,
        reference_adjmat,
        N,
        forces_val,
        Qs_val,
        poles_val,
        energies_val,
        geom_ids_val,
    )
    dataset_test = training_utils.create_dataset(
        template_dataset,
        job_details,
        xyzs_test,
        reference_adjmat,
        N,
        forces_test,
        Qs_test,
        poles_test,
        energies_test,
        geom_ids_test,
    )
    print("done creating create the datasets")
    path = os.path.join(job_details.WORKDIR, condition)
    dump_path = os.path.join(
        path, "{}_{}_dataset.pkl".format(condition, job_details.mode)
    )
    pickle.dump(dataset, open(dump_path, "wb"))
    dump_path = os.path.join(
        path, "{}_{}_dataset_val.pkl".format(condition, job_details.mode)
    )
    pickle.dump(dataset_val, open(dump_path, "wb"))
    dump_path = os.path.join(
        path, "{}_{}_dataset_test.pkl".format(condition, job_details.mode)
    )
    pickle.dump(dataset_test, open(dump_path, "wb"))
    print("dumping to:", dump_path)

    return dataset, dataset_val, dataset_test, template_dataset, job_details


def get_bond_def(batch, condition, logtext):
    bond_types = np.array(
        batch.GetTopology("bond", create_graph=True)["type"].squeeze(1).tolist()
    )
    bond_inx = np.array(
        batch.GetTopology("bond", create_graph=True)["index"].squeeze(1).tolist()
    )
    node_z = batch.get_data("node")["atomic_num"]
    bonds = []
    for btyp in torch.tensor(bond_types).unique():
        inx = np.where(bond_types == btyp.item())[0][0]
        logtext += (
            " ".join(
                [
                    "bond type",
                    str(btyp.item()),
                    "is between these elements:",
                    str(node_z[bond_inx[inx]].squeeze(1).tolist()),
                ]
            )
            + "\n"
        )
    logtext += "\n"
    return logtext


def loop(forcefield, dataset, dataset_val, job_details, condition, logtext):
    L, L_val, L_MSE, L_val_MSE, L_MAE, L_val_MAE, q_evo = [], [], [], [], [], [], []
    Fs, F_targets, geom_ids, atomic_num, node_types, nums = [], [], [], [], [], []
    ensembles, Es, E_targets, ps, p_targets = [], [], [], [], []
    batch = dataset.batches[0]
    logtext = ""
    import time

    start = time.time()
    for epoch in range(job_details.num_epoch):
        Fs, F_targets, geom_ids, atomic_num, node_types, nums = [], [], [], [], [], []
        ensembles, Es, E_targets, ps, p_targets = [], [], [], [], []
        # ---------------------------train loop -------------------------------
        forcefield.optimizer.zero_grad()
        num_atoms = 0
        # import ipdb
        # ipdb.set_trace()
        batch_loss, batch_MSE, batch_loss_MAE, num_atoms, forcefield = loop_train(
            forcefield, dataset, job_details, num_atoms
        )
        # Calculate the average loss per epoch
        avg_train_loss = (torch.stack(batch_loss).sum() / num_atoms).item()
        L.append(avg_train_loss)
        L_MSE.append((torch.stack(batch_MSE).sum() / num_atoms).item())
        L_MAE.append((torch.stack(batch_loss_MAE).sum() / num_atoms).item())

        # ---------------------------val loop -------------------------------
        forcefield.optimizer.zero_grad()
        num_atoms = 0
        (
            batch_loss,
            batch_MSE,
            batch_loss_MAE,
            num_atoms,
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
            batch,
            forcefield,
        ) = loop_val(
            forcefield,
            dataset_val,
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
            job_details,
            num_atoms,
            epoch,
        )
        # Calculate the average loss per epoch and update the learning rate every epoch with the validation loss
        avg_val_loss = (torch.stack(batch_loss).sum() / num_atoms).item()
        forcefield.scheduler.step(avg_val_loss)
        L_val.append(avg_val_loss)
        L_val_MSE.append((torch.stack(batch_MSE).sum() / num_atoms).item())
        L_val_MAE.append((torch.stack(batch_loss_MAE).sum() / num_atoms).item())

        # print progress
        if job_details.mode == "train" and epoch % 10 == 0:
            logtext += str(epoch) + "\n"
            print("epoch:", epoch)
            print("time since last print [min]:", (time.time() - start) / 60)
            print(
                "cur dih1 [kcal/molA]:", np.array(forcefield.learned_params["k1"][-1])
            )
            print(
                "cur dih2 [kcal/molA]:", np.array(forcefield.learned_params["k2"][-1])
            )
            print(
                "train MAE",
                L_MAE[-1],
                "val MAE",
                L_val_MAE[-1],
                "train RMSE",
                L_MSE[-1] ** 0.5,
                "val RMSE",
                L_val_MSE[-1] ** 0.5,
            )
            print("train loss", L[-1], "val loss", L_val[-1])
            param_compare_utils.plot_param_evolution(job_details, condition, forcefield)
            if "pcff" in job_details.job_name:
                param_compare_utils.plot_param_evolution_cross(
                    job_details, condition, forcefield
                )
            start = time.time()

    forcefield.set_params()
    path = os.path.join(
        job_details.WORKDIR, condition, "forcefield_{}.py".format(condition)
    )
    torch.save(forcefield, path)
    logtext += "\n"
    return (
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
    )


def loop_train(forcefield, dataset, job_details, num_atoms):
    batch_loss, batch_MSE, batch_loss_MAE = [], [], []
    for n, batch in enumerate(random.sample(dataset.batches, len(dataset.batches))):
        F, F_target, forcefield = calc_forces(
            batch, job_details, forcefield, train_flag=True
        )
        if "normFloss" in job_details.date:
            MSE, MAE, forcefield = calc_loss_norm(
                forcefield, F, F_target, batch, job_details
            )
        else:
            loss, MSE, MAE, forcefield = calc_loss(
                forcefield, F, F_target, batch, job_details
            )
        loss.backward()
        # to get correct average train loss per epoch, need to take into account the different number of atoms each batch
        num_atoms_batch = batch.get_data("num", "node").sum()
        batch_loss.append(loss * num_atoms_batch)
        batch_MSE.append(MSE * num_atoms_batch)
        batch_loss_MAE.append(MAE * num_atoms_batch)
        num_atoms += num_atoms_batch

        # Update the parameters every batch, not every epoch
        forcefield.optimizer.step()
        forcefield.update_learned_params()
        forcefield.optimizer.zero_grad()
    return batch_loss, batch_MSE, batch_loss_MAE, num_atoms, forcefield


def loop_test(forcefield, dataset_test, job_details, condition, logtext):
    L, L_MSE, L_MAE, q_evo, Fs, F_targets = [], [], [], [], [], []
    ensembles, Es, E_targets, ps, p_targets = [], [], [], [], []
    geom_ids, atomic_num, node_types, nums = [], [], [], []
    forcefield.optimizer.zero_grad()
    batch_loss, batch_MSE, batch_loss_MAE = [], [], []
    num_atoms = 0

    for n, batch in enumerate(
        random.sample(dataset_test.batches, len(dataset_test.batches))
    ):  # 30 x12
        F, F_target, forcefield = calc_forces(
            batch, job_details, forcefield, train_flag=False
        )
        E, E_target, ensemble = calc_energies(
            batch, job_details, forcefield, train_flag=False
        )

        if "normFloss" in job_details.date:
            MSE, MAE, forcefield = calc_loss_norm(
                forcefield, F, F_target, batch, job_details
            )
        else:
            loss, MSE, MAE, forcefield = calc_loss(
                forcefield, F, F_target, batch, job_details
            )
        # save these in order to analyze the final test force parity plots
        Fs.append(F)
        F_targets.append(F_target)
        Es.append(E)
        E_targets.append(E_target)
        ensembles.append(ensemble)
        ps, p_targets = None, None
        geom_ids.append(batch.get_data("prop", "geom_id"))
        atomic_num.append(batch.get_data("node", "atomic_num"))
        node_types.append(batch.get_data("node", "type"))
        nums.append(batch.get_data("num", "node"))
        # to get correct average val loss per epoch, need to take into account the different number of atoms each batch
        num_atoms_batch = batch.get_data("num", "node").sum()
        batch_loss.append(loss * num_atoms_batch)
        batch_MSE.append(MSE * num_atoms_batch)
        batch_loss_MAE.append(MAE * num_atoms_batch)
        num_atoms += num_atoms_batch
        forcefield.optimizer.zero_grad()

    # Calculate the average loss per epoch
    avg_train_loss = (torch.stack(batch_loss).sum() / num_atoms).item()
    L.append(avg_train_loss)
    L_MSE.append((torch.stack(batch_MSE).sum() / num_atoms).item())
    L_MAE.append((torch.stack(batch_loss_MAE).sum() / num_atoms).item())
    logtext += "Test MAE loss: " + str(L_MAE[-1]) + "\n"
    print("Test MAE loss:", L_MAE[-1])
    print("Test MSE loss:", L_MSE[-1])
    return (
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
    )


def loop_val(
    forcefield,
    dataset_val,
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
    job_details,
    num_atoms,
    epoch,
):
    batch_loss, batch_MSE, batch_loss_MAE = [], [], []
    for n, batch in enumerate(
        random.sample(dataset_val.batches, len(dataset_val.batches))
    ):  # 30 x12
        F, F_target, forcefield = calc_forces(
            batch, job_details, forcefield, train_flag=False
        )
        if "normFloss" in job_details.date:
            MSE, MAE, forcefield = calc_loss_norm(
                forcefield, F, F_target, batch, job_details
            )
        else:
            loss, MSE, MAE, forcefield = calc_loss(
                forcefield, F, F_target, batch, job_details
            )
        # save these for the final epoch in order to analyze the final parameter force parity plots
        Fs.append(F)
        F_targets.append(F_target)
        E, E_target, ensemble = calc_energies(
            batch, job_details, forcefield, train_flag=False
        )
        Es.append(E)
        E_targets.append(E_target)
        ensembles.append(ensemble)
        ps, p_targets = None, None
        geom_ids.append(batch.get_data("prop", "geom_id"))
        atomic_num.append(batch.get_data("node", "atomic_num"))
        node_types.append(batch.get_data("node", "type"))
        nums.append(batch.get_data("num", "node"))
        # to get correct average val loss per epoch, need to take into account the different number of atoms each batch
        num_atoms_batch = batch.get_data("num", "node").sum()
        batch_loss.append(loss * num_atoms_batch)
        batch_MSE.append(MSE * num_atoms_batch)
        batch_loss_MAE.append(MAE * num_atoms_batch)
        num_atoms += num_atoms_batch
        forcefield.optimizer.zero_grad()
    return (
        batch_loss,
        batch_MSE,
        batch_loss_MAE,
        num_atoms,
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
        batch,
        forcefield,
    )


def get_param_from_ff(forcefield, batch):
    # replaces the default (0.1) charge put in place of the forcefield when setting the charge to the actual set charge
    _CHARGE = forcefield._charge
    CHARGE = forcefield.charge
    param = {}
    param["charge"] = CHARGE
    _param = {}
    _param["charge"] = _CHARGE
    topology_dict = batch.GetTopology("pair", create_graph=True)
    _types = (
        torch.arange(forcefield.num_types).to(torch.long).view(-1, 1).to(CHARGE.device)
    )
    PARAM = {}
    for name in param.keys():
        custom_types = forcefield.type_map[name][_types.view(-1)].to(_types.device)
        custom_type_mask = (custom_types < 0).to(torch.float)
        PARAM[name] = 0.0
        PARAM[name] += _param[name][custom_types] * (1 - custom_type_mask)
        PARAM[name] += param[name][_types] * custom_type_mask
    return PARAM


def plot_charge_evolution(
    forcefield,
    job_details,
    condition,
    dataset,
    single_anion_dataset,
    single_cation_dataset,
):
    PARAMcharge = (
        get_param_from_ff(forcefield, dataset.batches[0])["charge"]
        .clone()
        .detach()
        .to("cpu")
    )
    all_z = dataset._data["node"]["atomic_num"].view(-1).to("cpu")
    all_types = dataset._data["node"]["type"].view(-1).to("cpu")
    all_ion_types = []
    all_ion_z = []
    learned_charges = torch.tensor(forcefield.learned_params["charge"])
    if single_anion_dataset:
        for cur_ion_dataset in single_anion_dataset + single_cation_dataset:
            figsize = (10, 10)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            # ani_types has right number of each atom type, so for example pf6- would be 7 long, with 6F and 1P.
            ion_types = cur_ion_dataset._data["node"]["type"].view(-1).to("cpu")
            ion_z = cur_ion_dataset._data["node"]["atomic_num"].view(-1).to("cpu")
            all_ion_types = all_ion_types + ion_types.tolist()
            all_ion_z = all_ion_z + ion_z.tolist()
            smiles = cur_ion_dataset.details["smiles"]
            if job_details.learn_charge_flag:
                learned_charges = torch.tensor(
                    forcefield.learned_params["charge"]
                ).cpu()
                num_updates = len(learned_charges[:, 0])
                num_batches = len(dataset.batches)
                ax.plot(
                    np.arange(num_updates) / num_batches,
                    learned_charges[:, ion_types].sum(dim=1),
                    label=f"total {smiles}",
                )
                # also plot the charge of each unique atom type corresponding to an atom in the anion
                for cur_type in ion_types.unique().tolist():
                    ax.plot(
                        np.arange(num_updates) / num_batches,
                        learned_charges[:, cur_type],
                        label=f"{Z_TO_SYMBOLS[ion_z[ion_types==cur_type][0].item()]}",
                    )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Charge")
                ax.set_yticks(np.arange(-1.5, 1.01, 0.25))
                ax.set_ylim(-1.5, 1.0)
                plt.legend()
                plt.grid()
                plt.tight_layout()
                if job_details.rep != None:
                    path = os.path.join(
                        job_details.WORKDIR,
                        condition,
                        str(job_details.rep),
                        job_details.mode,
                    )
                else:
                    path = os.path.join(
                        job_details.WORKDIR, condition, job_details.mode
                    )

                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(
                    os.path.join(
                        path,
                        "{}_{}_train_IONchargevEpoch_EV_BATCH.png".format(
                            condition, smiles
                        ),
                    )
                )
                plt.close()
            else:
                # use the PARAM dictionary which contains the charge for each atom type because there is 'learned charge' saved
                # at each step because the charge was not updated
                # start with anion
                update_ones = torch.ones(len(forcefield.learned_params["charge"])).cpu()
                total_ani = PARAMcharge[ani_types].sum()
                ax.plot(total_ani * update_ones, label="total anion")
                # also plot the charge of each unique atom type corresponding to an atom in the anion
                for cur_type in ani_types.unique().tolist():
                    ax.plot(
                        PARAMcharge[cur_type].cpu() * update_ones,
                        # label=f'anion {Element(ani_z[ani_types==cur_type][0].item()).symbol}')
                        label=f"anion {Z_TO_SYMBOLS[ion_z[ion_types==cur_type][0].item()]}",
                    )
                # repeat with cation
                total_cat = PARAMcharge[cat_types].sum()
                ax.plot(total_cat * update_ones, label="total cation")
                if len(cat_types) > 1:
                    # also plot the charge of each unique atom type corresponding to an atom in the anion
                    for cur_type in ani_types.unique().tolist():
                        ax.plot(
                            PARAMcharge[cur_type].cpu() * update_ones,
                            label=f"anion {Z_TO_SYMBOLS[ion_z[ion_types==cur_type][0].item()]}",
                        )
                ax.set_xlabel(
                    "Parameter update (1 epoch={} updates)".format(num_updates)
                )
                ax.set_ylabel("Charge")
                plt.legend()
                plt.tight_layout()
                plt.grid()
                if job_details.rep != None:
                    path = os.path.join(
                        job_details.WORKDIR,
                        condition,
                        str(job_details.rep),
                        job_details.mode,
                    )
                else:
                    path = os.path.join(
                        job_details.WORKDIR, condition, job_details.mode
                    )
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(
                    os.path.join(
                        path,
                        "{}_{}_train_IONchargevEpoch_EV_BATCH.png".format(
                            condition, smiles
                        ),
                    )
                )
                plt.close()
    # create a plot that has all the solvent charges:
    if (
        job_details.learn_charge_flag
        and "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]"
        in job_details.ensemble_to_smiles
    ):
        figsize = (10, 10)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        # ion_types = torch.cat([ani_types.unique(), cat_types.unique()])
        pre_mask, solv_types = (
            torch.arange(len(PARAMcharge)),
            torch.arange(len(PARAMcharge)),
        )
        mask = ~pre_mask.apply_(lambda x: x in all_ion_types).bool()
        solv_types = solv_types[mask]
        num_updates = len(learned_charges[:, 0])
        num_batches = len(dataset.batches)
        for cur_type in solv_types.unique().tolist():
            if len(all_z[all_types == cur_type]) > 0:
                ax.plot(
                    np.arange(num_updates) / num_batches,
                    learned_charges[:, cur_type].cpu(),
                    label=f"{Z_TO_SYMBOLS[all_z[all_types==cur_type][0].item()]}",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Charge")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if job_details.rep != None:
            path = os.path.join(
                job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
            )
        else:
            path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(
            os.path.join(path, "{}_SOLVchargevEpoch_EV_BATCH.png".format(condition))
        )
        plt.close()
    else:
        # create a plot that has all the solvent charges:
        update_ones = torch.ones(len(forcefield.learned_params["charge"])).cpu()
        all_types = dataset._data["node"]["type"].view(-1).to("cpu")
        all_z = dataset._data["node"]["atomic_num"].view(-1).to("cpu")
        figsize = (10, 10)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        # ion_types = torch.cat([ani_types.unique(), cat_types.unique()])
        pre_mask, solv_types = (
            torch.arange(len(PARAMcharge)).cpu(),
            torch.arange(len(PARAMcharge)).cpu(),
        )
        mask = ~pre_mask.apply_(lambda x: x in all_ion_types).bool()
        solv_types = solv_types[mask]
        for cur_type in solv_types.unique().tolist():
            ax.plot(
                PARAMcharge[cur_type].cpu() * update_ones,
                label=f"{Z_TO_SYMBOLS[all_z[all_types==cur_type][0].item()]}{cur_type}",
            )
        ax.set_xlabel("Parameter update (1 epoch={} updates)".format(num_updates))
        ax.set_ylabel("Charge")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if job_details.rep != None:
            path = os.path.join(
                job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
            )
        else:
            path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(
            os.path.join(path, "{}_SOLVchargevEpoch_EV_BATCH.png".format(condition))
        )
        plt.close()


def plot_multipole_parity(
    job_details, condition, multipoles, multipole_targets, logtext
):
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if job_details.mode == "train":
        x = torch.cat(multipoles).tolist()
        y = torch.cat(multipole_targets).tolist()
        multipoles_torch = torch.cat(multipoles).view(-1, 3).cpu().detach()
        multipole_targets_torch = (
            torch.cat(multipole_targets).view(-1, 3).cpu().detach()
        )
    elif job_details.mode == "test":
        #         x = F.view(-1).tolist()
        #         y = F_target.view(-1).tolist()
        x = torch.cat(multipoles).view(-1).tolist()
        y = torch.cat(multipole_targets).view(-1).tolist()
        multipoles_torch = torch.cat(multipoles).view(-1, 3).cpu().detach()
        multipole_targets_torch = (
            torch.cat(multipole_targets).view(-1, 3).cpu().detach()
        )
    ax.scatter(x=x, y=y, s=20, alpha=0.55)
    L_MAE = (multipoles_torch - multipole_targets_torch).abs().mean().item()
    if job_details.mode == "train":
        ax.set_title("Dipole Moment (e-Å), MAE={:0.5f}".format(L_MAE), fontsize=40)
    elif job_details.mode == "test":
        ax.set_title("Dipole Moment(e-Å), MAE={:0.5f}".format(L_MAE), fontsize=40)
    ax.set_xlabel("AuTopology", fontsize=40)
    ax.set_ylabel("DFT", fontsize=40)
    lim = torch.tensor(x + y).abs().max().item()
    ax.plot((-lim, lim), (-lim, lim), ls="--", c=".3", lw=3)
    ax.tick_params(labelsize=32)
    outliers = []

    path = os.path.join(job_details.WORKDIR, condition)
    path = os.path.join(
        path, "{}_{}_multipoles_outliers.pkl".format(condition, job_details.mode)
    )
    pickle.dump(outliers, open(path, "wb"))
    path = os.path.join(
        job_details.WORKDIR,
        condition,
        "{}_{}_multipole.pkl".format(condition, job_details.mode),
    )
    pickle.dump(x, open(path, "wb"))
    path = os.path.join(
        job_details.WORKDIR,
        condition,
        "{}_{}_multipole_target.pkl".format(condition, job_details.mode),
    )
    pickle.dump(y, open(path, "wb"))
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            job_details.WORKDIR,
            condition,
            "{}_{}_multipole_diff.png".format(condition, job_details.mode),
        )
    )
    plt.close()
    logtext += "\n"
    return logtext


def plot_force_parity(
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
    df_data,
    logtext,
    outlier_cut=25,
    scaled_flag=False,
):
    print("plotting force")
    cumnums = torch.cat(nums).cumsum(dim=0).detach().to("cpu")
    figsize = (12, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if job_details.mode == "train":
        x = torch.cat(Fs).view(-1).tolist()
        y = torch.cat(F_targets).view(-1).tolist()
        Fs_torch = torch.cat(Fs).view(-1, 3).cpu().detach()
        F_targets_torch = torch.cat(F_targets).view(-1, 3).cpu().detach()
        geom_ids = torch.cat(geom_ids).view(-1).tolist()
        atomic_num = torch.cat(atomic_num).view(-1).tolist()
        node_types = torch.cat(node_types).view(-1).tolist()
        ensembles = torch.cat(ensembles).cpu().detach()
    elif job_details.mode == "test":
        x = torch.cat(Fs).view(-1).tolist()
        y = torch.cat(F_targets).view(-1).tolist()
        Fs_torch = torch.cat(Fs).view(-1, 3).cpu().detach()
        F_targets_torch = torch.cat(F_targets).view(-1, 3).cpu().detach()
        geom_ids = torch.cat(geom_ids).view(-1).tolist()
        atomic_num = torch.cat(atomic_num).view(-1).tolist()
        node_types = torch.cat(node_types).view(-1).tolist()
        ensembles = torch.cat(ensembles).cpu().detach()
    nums = torch.cat(nums).view(-1).cpu().detach().tolist()
    # need to split the forces by molecule to plot by species
    Fs_torch = torch.split(Fs_torch, nums)
    F_targets_torch = torch.split(F_targets_torch, nums)
    for ensemble in ensembles.unique().tolist():
        x_start, y_start = [], []
        for i, curflag in enumerate(ensembles == ensemble):
            if curflag:
                x_start.append(Fs_torch[i])
                y_start.append(F_targets_torch[i])
        x = torch.cat(x_start).view(-1).tolist()
        y = torch.cat(y_start).view(-1).tolist()
        ax.scatter(
            x=x, y=y, s=20, alpha=0.55, label=job_details.ensemble_to_smiles[ensemble]
        )
    if job_details.mode == "train":
        ax.set_title(
            "Force (kcal/mol-Å), MAE={:0.5f}".format(L_val_MAE[-1]), fontsize=40
        )
    elif job_details.mode == "test":
        ax.set_title("Force (kcal/mol-Å), MAE={:0.5f}".format(L_MAE[-1]), fontsize=40)
    ax.set_xlabel("AuTopology", fontsize=40)
    ax.set_ylabel("DFT", fontsize=40)
    lim = torch.tensor(x + y).abs().max().item()
    # lim = 200
    ax.plot((-lim, lim), (-lim, lim), ls="--", c=".3", lw=3)
    ax.tick_params(labelsize=32)
    outliers = []
    outlier_data = {
        "geom_id": [],
        "n": [],
        "smiles": [],
        "F_auto": [],
        "F_DFT": [],
        "atomic_num": [],
    }
    Fs_torch = torch.cat(Fs_torch)
    F_targets_torch = torch.cat(F_targets_torch)
    for n, cur_x in enumerate(x):
        if abs(cur_x - y[n]) > outlier_cut:
            #         #plot as red dot labelled by index
            geomid_inx = np.where(n // 3 < cumnums)[0][0].item()
            outlier_id = geom_ids[geomid_inx]
            outliers.append(outlier_id)
            outlier_data["geom_id"].append(outlier_id)
            outlier_data["n"].append(n)
            outlier_data["smiles"].append(
                df_data[df_data.geom_id == outlier_id].smiles.item()
            )
            outlier_data["atomic_num"].append(atomic_num[n // 3])
            outlier_data["F_auto"].append(Fs_torch[n // 3].tolist())
            outlier_data["F_DFT"].append(F_targets_torch[n // 3].tolist())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    df_outlier = pd.DataFrame(outlier_data).sort_values("geom_id").set_index("geom_id")
    if job_details.rep != None:
        base_path = os.path.join(
            job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
        )
    else:
        base_path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = os.path.join(base_path, f"{condition}_{job_details.mode}_outlier_geoms.csv")
    df_outlier.to_csv(open(path, "w"))
    path = os.path.join(base_path, "{}_{}_F.pkl".format(condition, job_details.mode))
    pickle.dump(x, open(path, "wb"))
    path = os.path.join(
        base_path, "{}_{}_F_target.pkl".format(condition, job_details.mode)
    )
    pickle.dump(y, open(path, "wb"))
    plt.legend(fontsize=15)
    plt.tight_layout()
    if scaled_flag:
        plt.savefig(
            os.path.join(
                base_path,
                "SCALED_{}_{}_force_diff.png".format(condition, job_details.mode),
            )
        )
    else:
        plt.savefig(
            os.path.join(
                base_path, "{}_{}_force_diff.png".format(condition, job_details.mode)
            )
        )
    plt.close()
    logtext += "\n"
    return logtext


def plot_energy_parity(
    job_details,
    condition,
    Es,
    E_targets,
    ensembles,
    geom_ids,
    df_data,
    logtext,
    scaled_flag=False,
):
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    Es_torch = torch.cat(Es).cpu().detach()
    E_targets_torch = torch.cat(E_targets).cpu().detach()
    ensembles = torch.cat(ensembles).cpu().detach()
    geom_ids = torch.cat(geom_ids).view(-1).tolist()
    print("new plot energy")
    for ensemble in ensembles.unique().tolist():
        Es_torch[ensembles == ensemble] = (
            Es_torch[ensembles == ensemble] - Es_torch[ensembles == ensemble].mean()
        )
        E_targets_torch[ensembles == ensemble] = (
            E_targets_torch[ensembles == ensemble]
            - E_targets_torch[ensembles == ensemble].mean()
        )
        x = Es_torch[ensembles == ensemble].squeeze().tolist()
        y = E_targets_torch[ensembles == ensemble].squeeze().tolist()
        ax.scatter(
            x=x, y=y, s=20, alpha=0.55, label=job_details.ensemble_to_smiles[ensemble]
        )
    L_MAE = (Es_torch - E_targets_torch).abs().mean().item()
    if job_details.mode == "train":
        ax.set_title("Energy (kcal/mol), MAE={:0.5f}".format(L_MAE), fontsize=40)
    elif job_details.mode == "test":
        ax.set_title("Energy (kcal/mol-atom), MAE={:0.5f}".format(L_MAE), fontsize=40)
    ax.set_xlabel("AuTopology", fontsize=40)
    ax.set_ylabel("DFT", fontsize=40)
    lim = torch.tensor(x + y).abs().max().item()
    ax.plot((-lim, lim), (-lim, lim), ls="--", c=".3", lw=3)
    ax.tick_params(labelsize=32)
    outliers = []
    outlier_data = {"geom_id": [], "smiles": [], "E_auto": [], "E_DFT": []}
    for n, cur_x in enumerate(x):
        outlier_id = geom_ids[n]
        outlier_data["geom_id"].append(outlier_id)
        outlier_data["smiles"].append(
            df_data[df_data.geom_id == outlier_id].smiles.item()
        )
        outlier_data["E_auto"].append(Es_torch[n].tolist())
        outlier_data["E_DFT"].append(E_targets_torch[n].tolist())
    df_outlier = pd.DataFrame(outlier_data).sort_values("geom_id").set_index("geom_id")
    if job_details.rep != None:
        base_path = os.path.join(
            job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
        )
    else:
        base_path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = os.path.join(
        base_path, f"{condition}_{job_details.mode}_energy_outlier_geoms.csv"
    )
    df_outlier.to_csv(open(path, "w"))

    if job_details.rep != None:
        base_path = os.path.join(
            job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
        )
    else:
        base_path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = os.path.join(
        base_path, "{}_{}_E_outliers.pkl".format(condition, job_details.mode)
    )
    pickle.dump(outliers, open(path, "wb"))
    path = os.path.join(base_path, "{}_{}_E.pkl".format(condition, job_details.mode))
    pickle.dump(x, open(path, "wb"))
    path = os.path.join(
        base_path, "{}_{}_E_target.pkl".format(condition, job_details.mode)
    )
    pickle.dump(y, open(path, "wb"))
    plt.legend(fontsize=15)
    plt.tight_layout()
    if scaled_flag:
        plt.savefig(
            os.path.join(
                base_path,
                "SCALED_{}_{}_energy_diff.png".format(condition, job_details.mode),
            )
        )
    else:
        plt.savefig(
            os.path.join(
                base_path, "{}_{}_energy_diff.png".format(condition, job_details.mode)
            )
        )
    plt.close()
    logtext += "\n"
    return logtext


def plot_loss_evolution(forcefield, job_details, condition, L_MAE, L_val_MAE):
    forcefield.set_params()
    path = os.path.join(
        job_details.WORKDIR, condition, "forcefield_{}.py".format(condition)
    )
    torch.save(forcefield, path)
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MAE, kcal/molA)")
    ax.plot(L_MAE, label="train: {:0.2f}".format(L_MAE[-1]))
    ax.plot(L_val_MAE, label="val: {:0.2f}".format(L_val_MAE[-1]))
    plt.legend()
    plt.tight_layout()
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR, condition, str(job_details.rep), job_details.mode
        )
    else:
        path = os.path.join(job_details.WORKDIR, condition, job_details.mode)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, "{}_train_LossvEpoch.png".format(condition)))
    plt.close()
