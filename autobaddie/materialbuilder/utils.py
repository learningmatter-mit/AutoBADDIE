import torch, random
from autobaddie.materialbuilder import (
    topologies,
    transformations,
    matbuilder,
    graphbuilder,
)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem as Chem

from autobaddie.materialbuilder.xyz2mol import xyz2mol
from collections import OrderedDict

from autobaddie.materialbuilder.matbuilder import Z_TO_SYMBOLS, SYMBOLS_TO_Z


def fix_A(A, z, dists):
    H_indices = torch.nonzero((z == 1).to(torch.long), as_tuple=False).view(-1).tolist()
    for i in H_indices:
        neighbors = torch.nonzero(A[i], as_tuple=False).view(-1).tolist()
        for j in neighbors:
            if j in H_indices:
                A[i][j] = 0
                A[j][i] = 0
        neighbors = torch.nonzero(A[i], as_tuple=False).view(-1).tolist()
        if len(neighbors) > 1:
            neighbor_dists = [dists[i][n] for n in neighbors]
            j_to_keep = neighbors[neighbor_dists.index(min(neighbor_dists))]
            for j in neighbors:
                if j != j_to_keep:
                    A[i][j] = 0
                    A[j][i] = 0
        neighbors = A[i].nonzero(as_tuple=False).view(-1).tolist()
        if len(neighbors) != 1:
            raise Exception("Too few or too many bonds to H. Could not be fixed.")
    return A


def geometry_to_dataset(geometry):
    dataset = graphbuilder.GraphDataset()
    dataset.AddGraph(geometry)
    dataset.Close()
    dataset.AddTransformation(
        transformations.GraphDistinctNodes(
            in_label="atomic_num", out_label="rad_2_one_hot_atom_types", radius=2
        )
    )
    dataset.AddTransformation(
        transformations.OneHotEncoding(
            level="node", in_label="atomic_num", out_label="one_hot_atomic_num"
        )
    )
    dataset.DefineNodeTypes("rad_2_one_hot_atom_types")
    for top in [
        topologies.BondTopology(),
        topologies.AngleTopology(),
        topologies.DihedralTopology(),
    ]:
        dataset.AddTopology(top)
    dataset.CreateBatches(
        batch_size=dataset.dataset_size, shuffle=False, device=job_details.device
    )
    return dataset


def get_dihedral_profiles(params):
    OPLS_constants = params["OPLS_constants"].unique(dim=0)
    phi_list = torch.linspace(0, np.pi, 1000)
    E_list = []
    for phi in phi_list:
        E = 0.0
        for m in range(1):
            V = OPLS_constants[:, [m]]
            E += (V / 2) * torch.cos((m + 1) * phi)
        E_list.append(E)
    E = torch.cat(E_list, dim=1)
    num_types = len(OPLS_constants)
    types = torch.arange(num_types).view(1, num_types).expand(len(phi_list), num_types)
    phi_list = phi_list.view(-1, 1).expand(len(phi_list), num_types)
    E = E.t()
    df = pd.DataFrame()
    df["phi"] = phi_list.reshape(-1).tolist()
    df["type"] = types.reshape(-1).tolist()
    df["E"] = E.reshape(-1).tolist()
    return df


def get_substructures(rdkit_mol, pattern, is_hydrogen=False, show_indices=False):
    pattern_mol = Chem.MolFromSmarts(pattern)
    indices = torch.tensor(
        [list(x) for x in rdkit_mol.GetSubstructMatches(pattern_mol)]
    )
    X = torch.zeros(rdkit_mol.GetNumAtoms()).to(torch.long)
    if indices.tolist():
        if is_hydrogen:
            z = torch.tensor(
                [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
            )
            A_hydrogens = torch.tensor(Chem.GetAdjacencyMatrix(rdkit_mol).tolist())
            A_hydrogens[:, z.view(-1) != 1] = 0
            indices = A_hydrogens[indices.view(-1)].nonzero()[:, [1]]
    if indices.tolist():
        X[indices] = 1
    if show_indices:
        print(X.nonzero().view(-1).tolist())
    return X


def read_ff_file(path):
    file = open(path, "r")
    ff_file = file.read()
    file.close()
    rows = ff_file.split("\n")
    while True:
        if rows[-1] == "":
            rows = rows[:-1]
        else:
            break
    for i in range(len(rows)):
        rows[i] = " ".join(rows[i].split()).split(" ")
    patterns = OrderedDict({})
    for row in rows:
        if row[0] == "def":
            patterns[row[3]] = row[4]
    variables = {}
    for row in rows:
        if row[0] == "var":
            top = row[1]
            variables[top] = row[2:]
    param_dicts = {}
    for row in rows:
        if row[0] == "set":
            top = row[1]
            num_params = len(variables[top])
            if top not in param_dicts.keys():
                param_dicts[top] = {}
            indices = tuple([int(i) for i in row[2:-num_params]])
            param_dicts[top][indices] = [float(x) for x in row[-num_params:]]
    for top in param_dicts.keys():
        df = pd.DataFrame(param_dicts[top]).transpose()
        num_params = df.shape[1]
        df = df.rename(columns={i: variables[top][i] for i in range(num_params)})
        df = df.reset_index(drop=False)
        df = df.rename(
            columns={"level_{}".format(i): i for i in range(df.shape[1] - num_params)}
        )
        param_dicts[top] = df
    return (patterns, variables, param_dicts)


def get_train_test_split(job_details, geoms_dataframe):
    geom_ids = {"train": [], "test": []}
    spec_ids = {"train": [], "test": []}
    if job_details.species_train_test_split_method == "distinct":
        species_ids = list(geoms_dataframe["species_id"])
        random.shuffle(species_ids)
        for species_id in species_ids:
            if len(geom_ids["train"]) < job_details.num_train:
                split = "train"
            else:
                split = "test"
            spec_ids[split].append(species_id)
            for geom_id in geoms_dataframe[geoms_dataframe["species_id"] == species_id][
                "geom_ids"
            ].item():
                geom_ids[split].append(geom_id)
        if job_details.shuffle_geoms:
            for split in ["train", "test"]:
                random.shuffle(geom_ids[split])
            geom_ids["train"] = geom_ids["train"][: job_details.num_train]
            geom_ids["test"] = geom_ids["test"][: job_details.num_test]
    elif job_details.species_train_test_split_method == "overlap":
        species_ids = list(geoms_dataframe["species_id"])
        random.shuffle(species_ids)
        for species_id in species_ids:
            spec_ids["train"].append(species_id)
            spec_ids["test"].append(species_id)
            geom_id_list = geoms_dataframe[geoms_dataframe["species_id"] == species_id][
                "geom_ids"
            ].item()
            if job_details.shuffle_geoms:
                random.shuffle(geom_id_list)
            train_fraction = float(job_details.num_train) / (
                job_details.num_train + job_details.num_test
            )
            num_train_geoms = train_fraction * len(geom_id_list)
            if num_train_geoms % int(num_train_geoms) != 0:
                if random.random() < num_train_geoms % int(num_train_geoms):
                    num_train_geoms = int(num_train_geoms) + 1
                else:
                    num_train_geoms = int(num_train_geoms)
            else:
                num_train_geoms = int(num_train_geoms)
            for geom_id in geom_id_list[:num_train_geoms]:
                geom_ids["train"].append(geom_id)
            for geom_id in geom_id_list[num_train_geoms:]:
                geom_ids["test"].append(geom_id)
    else:
        geom_and_species = []
        for spec_id, row in geoms_dataframe.iterrows():
            for geom_id in row.geom_ids:
                geom_and_species.append([geom_id, spec_id])
        if job_details.shuffle_geoms:
            random.shuffle(geom_and_species)
        geom_and_species = np.array(geom_and_species)
        A = job_details.num_train
        B = job_details.num_test
        geom_ids["train"] = geom_and_species[:A, 0].tolist()
        spec_ids["train"] = list(set(geom_and_species[:A, 1].tolist()))
        geom_ids["test"] = geom_and_species[A : A + B, 0].tolist()
        spec_ids["test"] = list(set(geom_and_species[A : A + B, 1].tolist()))
    return (geom_ids, spec_ids)


def geometry_from_xyz(filename, return_mol=False):
    if filename.endswith(".xyz"):
        file = open(filename, "r")
        xyz = file.read()
        file.close()
    else:
        # can directly input the xyz, for example from Geom.as_xyz()
        # print('xyz string directly input')
        xyz = filename
    xyz = xyz.split("\n")[2:-1]
    for i in range(len(xyz)):
        xyz[i] = " ".join(xyz[i].split()).split(" ")
        if type(xyz[i][0]) is str:
            try:
                # xyz[i][0] = Element(int(xyz[i][0])).atomic_number
                xyz[i][0] = Z_TO_SYMBOLS[int(xyz[i][0])]
            except:
                # xyz[i][0] = Element(xyz[i][0]).atomic_number
                xyz[i][0] = SYMBOLS_TO_Z[xyz[i][0]]
        else:
            # xyz[i][0] = Element(xyz[i][0]).atomic_number
            xyz[i][0] = SYMBOLS_TO_Z[xyz[i][0]]
        for j in [1, 2, 3]:
            xyz[i][j] = float(xyz[i][j])
    xyz = torch.tensor(xyz)
    rdkit_mol = xyz2mol.xyz2mol(
        xyz[:, 0].to(torch.long).tolist(), 0, xyz[:, -3:].tolist(), True, True, False
    )
    conformer = list(rdkit_mol.GetConformers())[0]
    xyz = conformer.GetPositions()
    xyz = torch.tensor(xyz.tolist())
    A_ref = torch.tensor(Chem.GetAdjacencyMatrix(rdkit_mol).tolist())
    atomic_nums = torch.tensor(
        [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
    )
    geometry = matbuilder.Geometry(A=A_ref, atomic_nums=atomic_nums, xyz=xyz)
    if return_mol:
        return (geometry, rdkit_mol)
    else:
        return geometry


def get_parameterization_dataset(job_details, datasets, geometry):
    dataset = graphbuilder.GraphDataset()
    ensemble = graphbuilder.Ensemble()
    ensemble.Add(geometry)
    dataset.AddEnsemble(ensemble)
    dataset.Close()
    """
    for trans in job_details.transformations:
        dataset.AddTransformation(
            eval('transformations.'+trans[0])(*tuple(trans[1:])))
    """
    for _transformation in datasets["template"].transformations.keys():
        for transformation in datasets["template"].transformations[_transformation]:
            dataset.AddTransformation(transformation)
    dataset.DefineBaseNodeTypes(job_details.base_node_type)
    dataset.DefineNodeTypes(job_details.node_type)
    if "bond" in job_details.terms:
        dataset.AddTopology(topologies.BondTopology())
    if "angle" in job_details.terms:
        dataset.AddTopology(topologies.AngleTopology())
    if "dihedral" in job_details.terms:
        dataset.AddTopology(topologies.DihedralTopology())
    if "improper" in job_details.terms:
        dataset.AddTopology(topologies.ImproperTopology())
    if "pair" in job_details.terms:
        dataset.AddTopology(
            topologies.PairTopology(job_details.use_1_4_pairs, job_details.pair_cutoff)
        )
    dataset.UnzipEnsembles()
    dataset.CreateBatches(
        batch_size=job_details.batch_size, shuffle=False, device=job_details.device
    )
    datasets = {**datasets, **{"parameterize": dataset}}
    return datasets
