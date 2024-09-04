import sys, subprocess, os, importlib, shutil, socket, copy

# import random, itertools, logging, argparse
# import materialbuilder
from materialbuilder import (
    # graphbuilder, matbuilder, potentials, topologies,
    # terms, typers, output, transformations,
    # utils, units, plotting, inout, dbinterface, dbsettings)
    graphbuilder,
    matbuilder,
    output,
    utils,
)

# from materialbuilder.dbsettings import Geom, Group, Species, MolSet, Calc, Jacobian, Job, Count
from materialbuilder.dbsettings import Geom
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from munch import Munch
import _pickle as pickle
import pandas as pd
from rdkit.Chem import AllChem as Chem
import networkx as nx
import networkx.algorithms.isomorphism as iso
from sklearn.model_selection import train_test_split
from scipy.linalg import block_diag
from materialbuilder.plotting import *
import self_contained.utils.train as train


def get_dataset_from_xyz(
    param_name_xyz, job_details, condition, template_dataset, device, return_rdkit=False
):
    # Uses rdkit molecules to get a representation of the desired xyz file into an AutoBADDIE dataset.
    # input - param_name = name of xyz file containing the coordinates of atoms in the molecule to be parameterized
    # job_details = Munch collection of run options
    # output - dataset - materialbuilder dataset object that contains only one molecule in it - from the xyz file
    # rdkit_mol = rdkit Molecule object representation of molecule to be parameterized.
    # NOTE- this dataset is not yet batched and also does not contain the topology information for the particular conformation,
    # only the coordinates
    dataset = graphbuilder.GraphDataset()
    ensemble = graphbuilder.Ensemble()
    # param_name_xyz could be used as the entire xyz string for example from Geom.as_xyz() from htvs
    # usually, though, it is the filename to an xyz file
    if param_name_xyz.endswith(".xyz"):
        path = os.path.join(job_details.WORKDIR, param_name_xyz)
    else:
        path = param_name_xyz
    geometry, rdkit_mol = utils.geometry_from_xyz(filename=path, return_mol=True)
    ensemble.Add(geometry)
    dataset.AddEnsemble(ensemble)
    dataset.Close()
    dataset = get_atom_types_for_single_mol_from_template(
        job_details,
        condition,
        param_name_xyz,
        rdkit_mol,
        template_dataset,
        dataset,
        device,
    )
    if return_rdkit:
        return dataset, rdkit_mol
    return dataset


def get_dataset_from_smiles(
    smiles,
    job_details,
    condition,
    template_dataset,
    device,
    return_rdkit=False,
    save_flag=True,
):
    # Queries the database to find the conformer with the lowest energy in the database for the smiles to be parameterized.
    # input - param_name = string of SMILES represetation of the molecule to be learned.
    # job_details = Munch collection of run options
    # condition - string containing the label of the particular forcefield that is being used to parameterize the molecule
    # output - dataset - materialbuilder dataset object that contains a single batch with only one molecule in it - from the xyz file
    # rdkit_mol = rdkit Molecule object representation of molecule to be parameterized.
    # NOTE- this dataset is not yet batched and also does not contain the topology information for the particular conformation,
    # only the coordinates
    dataset = graphbuilder.GraphDataset()
    ensemble = graphbuilder.Ensemble()
    geometry, rdkit_mol = matbuilder.geometry_from_smiles(
        smiles, dim=3, return_adjmat=False, return_rdkit_mol=True
    )

    ensemble.Add(geometry)
    dataset.AddEnsemble(ensemble)
    dataset.Close()
    dataset = get_atom_types_for_single_mol_from_template(
        job_details, condition, smiles, rdkit_mol, template_dataset, dataset, device
    )
    dataset.details["smiles"] = smiles
    if return_rdkit:
        return dataset, rdkit_mol
    return dataset


def write_mol_and_pdb_files_for_single_dataset(
    job_details, condition, mol_name, rdkit_mol
):
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}.mol".format(mol_name),
        )
    else:
        path = os.path.join(job_details.WORKDIR, condition, "{}.mol".format(mol_name))

    file = open(path, "w")
    file.write(Chem.MolToMolBlock(rdkit_mol))
    file.close()
    if job_details.rep != None:
        mol_path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}.mol".format(mol_name),
        )
        pdb_path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}.pdb".format(mol_name),
        )
    else:
        mol_path = os.path.join(
            job_details.WORKDIR, condition, "{}.mol".format(mol_name)
        )
        pdb_path = os.path.join(
            job_details.WORKDIR, condition, "{}.pdb".format(mol_name)
        )
    subprocess.call(["obabel", mol_path, "-O", pdb_path])


def get_atom_types_for_single_mol_from_template(
    job_details, condition, param_name, rdkit_mol, template_dataset, dataset, device
):
    # Creates a batch in the parameterize dataset to add the topology calculations (bond distances, angle/dihedral/improper angles, etc)
    # input - job_details, condtion, param_name, rdkit_mol same as get_param_dataset_from_smiles() help
    # template_dataset = materialbuilder object that contains the topology types (terms) and (discrete) atom types that were used to train the FF
    # dataset = parameterize dataset containing molecule to be parameterized.
    # output - dataset object that contains a single batch with a single molecule but now with all topologies and types calculated
    for _transformation in template_dataset.transformations.keys():
        for transformation in template_dataset.transformations[_transformation]:
            dataset.AddTransformation(transformation, job_details.device)
    base_node_type = template_dataset.container_batch.base_node_type_label
    if base_node_type is not None:
        dataset.DefineBaseNodeTypes(base_node_type)
    node_type = template_dataset.container_batch.node_type_label
    if node_type is not None:
        dataset.DefineNodeTypes(node_type)
    terms = list(template_dataset.container_batch.topologies.keys())
    if "bond" in terms:
        dataset.AddTopology(
            template_dataset.container_batch.topologies["bond"], job_details.device
        )
    if "angle" in terms:
        dataset.AddTopology(
            template_dataset.container_batch.topologies["angle"], job_details.device
        )
    if "dihedral" in terms:
        dataset.AddTopology(
            template_dataset.container_batch.topologies["dihedral"], job_details.device
        )
    if "improper" in terms:
        dataset.AddTopology(
            template_dataset.container_batch.topologies["improper"], job_details.device
        )
    if "pair" in terms:
        dataset.AddTopology(
            template_dataset.container_batch.topologies["pair"], job_details.device
        )
    dataset.UnzipEnsembles()
    dataset.CreateBatches(batch_size=1, shuffle=False, device=job_details.device)
    if job_details and job_details.mode == "parameterize":
        path = os.path.join(job_details.WORKDIR, condition)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(
            path, "{}_{}_{}_dataset.pkl".format(condition, job_details.mode, param_name)
        )
        pickle.dump(dataset, open(path, "wb"))
    dataset = dataset.to(device)
    return dataset


def scale_charges_v2(
    job_details,
    smiles,
    dataset,
    forcefield,
    condition,
    single_anion_dataset,
    single_cation_dataset,
    single_solv_dataset,
    ani_smiles,
    cat_smiles,
    logtext,
):
    # scales the charges of each molecule type to have an exact net charge (0 for polymer or job_details.param_sc_hyp for anion and cation)
    # to ensure there is no net charge in the final MD simulations
    PARAMcharge = (
        train.get_param_from_ff(forcefield, dataset.batches[0])["charge"]
        .clone()
        .detach()
        .to(job_details.device)
    )
    scaled_ff = copy.deepcopy(forcefield).to(job_details.device)
    new_charge = torch.zeros(forcefield.charge.size(), device=job_details.device)
    if single_anion_dataset:
        # if job_details.learn_charge_flag:
        # raise NameError('Need to add the functionality of scaling charge with multiple anions and cations')
        #         else:
        if not job_details.param_sc_hyp:
            param_sc_hyp = 1
        else:
            param_sc_hyp = job_details.param_sc_hyp
        for cur_dataset in single_cation_dataset + single_anion_dataset:
            # find the total charge for a ions if it is being used
            cur_types = cur_dataset._data["node"]["type"].view(-1)
            total_ion_charge = PARAMcharge[cur_types].sum()
            unique_types = cur_types.unique()
            new_charge = torch.scatter_add(
                new_charge,
                0,
                unique_types,
                PARAMcharge[unique_types].view(-1)
                / abs(total_ion_charge)
                * param_sc_hyp,
            )
        for cur_dataset in single_solv_dataset:
            # find the total charge for a solvent molecule by averaging the pos and neg charges
            cur_types = cur_dataset._data["node"]["type"].view(-1)
            cur_charges = PARAMcharge[cur_types].view(-1)
            pol_pos = cur_charges[cur_charges > 0].sum().item()
            pol_neg = cur_charges[cur_charges < 0].sum().item()
            if (
                len(cur_charges[cur_charges > 0]) == 0
                or len(cur_charges[cur_charges < 0]) == 0
            ):
                continue
            pol_sc = np.average([pol_pos, np.abs(pol_neg)])
            scaled_charges = []
            for cur_type in cur_types.unique():
                cur_charge = PARAMcharge[cur_type].item()
                if cur_charge > 0:
                    scaled_charges.append(cur_charge / abs(pol_pos) * pol_sc)
                else:
                    scaled_charges.append(cur_charge / abs(pol_neg) * pol_sc)
            new_charge = torch.scatter_add(
                new_charge,
                0,
                cur_types.unique(),
                torch.tensor(
                    scaled_charges, device=job_details.device, dtype=new_charge.dtype
                ),
            )
        scaled_ff.charge = torch.tensor(
            new_charge.tolist(),
            requires_grad=True,
            device=job_details.device,
            dtype=torch.float,
        )
    else:
        for cur_dataset in single_solv_dataset:
            # find the total charge for a solvent molecule by averaging the pos and neg charges
            cur_types = cur_dataset._data["node"]["type"].view(-1)
            cur_charges = PARAMcharge[cur_types].view(-1)
            pol_pos = cur_charges[cur_charges > 0].sum().item()
            pol_neg = cur_charges[cur_charges < 0].sum().item()
            pol_sc = np.average([pol_pos, np.abs(pol_neg)])
            scaled_charges = []
            # import ipdb
            # ipdb.set_trace()
            for cur_type in cur_types.unique():
                cur_charge = PARAMcharge[cur_type].item()
                # import ipdb
                # ipdb.set_trace()
                if cur_charge > 0:
                    scaled_charges.append(cur_charge / abs(pol_pos) * pol_sc)
                else:
                    scaled_charges.append(cur_charge / abs(pol_neg) * pol_sc)
            new_charge = torch.scatter_add(
                new_charge,
                0,
                cur_types.unique(),
                torch.tensor(
                    scaled_charges, device=job_details.device, dtype=new_charge.dtype
                ),
            )

    print("\n\nthese are the scaled ff charges:\n", scaled_ff.charge, "\n\n")
    logtext += (
        "\n".join(["these are the scaled ff charges:", str(scaled_ff.charge), "\n"])
        + "\n"
    )

    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            "scaled_{}_forcefield_{}_rep{}.py".format(
                str(job_details.param_sc_hyp)[-1], condition, str(job_details.rep)
            ),
        )
    else:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            "scaled_{}_forcefield_{}.py".format(
                str(job_details.param_sc_hyp)[-1], condition
            ),
        )
    torch.save(scaled_ff, path)
    return scaled_ff, logtext


def write_lammps_data_files(
    param_name, job_details, condition, batch, forcefield, scaled_ff, logtext
):
    print("beginning to write lammps data file for:", param_name)
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}_{}.data".format(condition, param_name),
        )
    else:
        path = os.path.join(
            job_details.WORKDIR, condition, "{}_{}.data".format(condition, param_name)
        )
    print(path)
    if "pcff" in job_details.job_name:
        output.write_lammps_data_file(
            batch=batch,
            filename=path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="compass",
        )
    else:
        output.write_lammps_data_file(
            batch=batch,
            filename=path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="OPLS",
        )
    atomic_nums = batch.get_data("node", "atomic_num").view(-1).tolist()
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}.z".format(param_name),
        )
    else:
        path = os.path.join(job_details.WORKDIR, condition, "{}.z".format(param_name))
    f = open(path, "w+")
    for z in atomic_nums:
        f.write(str(z) + " ")
    f.close()
    # ----------------------------scaled data file-----------------------------------------------------
    batch._data["node"]["charge"] = scaled_ff.charge[batch.get_data("node", "type")]
    if job_details.rep != None:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            str(job_details.rep),
            "{}_{}_SCALED{}.data".format(
                condition, param_name, str(job_details.param_sc_hyp).split(".")[-1]
            ),
        )
    else:
        path = os.path.join(
            job_details.WORKDIR,
            condition,
            "{}_{}_SCALED{}.data".format(
                condition, param_name, str(job_details.param_sc_hyp).split(".")[-1]
            ),
        )

    print(path)
    if "pcff" in job_details.job_name:
        output.write_lammps_data_file(
            batch=batch,
            filename=path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="compass",
        )
    else:
        output.write_lammps_data_file(
            batch=batch,
            filename=path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="OPLS",
        )
    # -------------------------------------write a version of the scaled in single_mol --------------------------------
    print("writing single_mol datafile")
    logtext += "writing single_mol datafile\n"
    if "carbonate" in job_details.job_name:
        engaging_path = os.path.join(
            os.path.abspath("../../.."),
            "engaging_cluster/proj/carbonate/single_mol",
            job_details.job_name,
            condition,
        )
    else:
        engaging_path = os.path.join(
            os.path.abspath("../../.."),
            "engaging_cluster/proj/bor/single_mol",
            job_details.job_name,
            condition,
        )
    if not os.path.exists(engaging_path):
        os.makedirs(engaging_path)
        print("Directory ", engaging_path, " Created ")
        logtext += " ".join(["\nDirectory", engaging_path, "Created\n"])
    else:
        print("Directory ", engaging_path, " already exists")
        logtext += " ".join(["\nDirectory", engaging_path, "already exists\n"])
    if job_details.rep != None:
        engaging_path = os.path.join(
            engaging_path,
            "{}_{}_SCALED{}_rep{}.data".format(
                condition,
                param_name,
                str(job_details.param_sc_hyp).split(".")[-1],
                str(job_details.rep),
            ),
        )
    else:
        engaging_path = os.path.join(
            engaging_path,
            "{}_{}_SCALED{}.data".format(
                condition, param_name, str(job_details.param_sc_hyp).split(".")[-1]
            ),
        )

    print(engaging_path)
    if "pcff" in job_details.job_name:
        output.write_lammps_data_file(
            batch=batch,
            filename=engaging_path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="compass",
            single_mol=True,
        )
    else:
        output.write_lammps_data_file(
            batch=batch,
            filename=engaging_path,
            job_details=job_details,
            phi0_flag=forcefield.phi0_flag,
            style="OPLS",
            single_mol=True,
        )
    return logtext
