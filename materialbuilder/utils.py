import torch, random, copy, itertools, ast
import materialbuilder
from materialbuilder import (
    potentials,
    topologies,
    terms,
    transformations,
    units,
    htvs,
    matbuilder,
    graphbuilder,
    typers,
    dbsettings,
    inout,
    dbinterface,
)
from materialbuilder.dbsettings import Geom, Group, Species
from htvs.djangochem.analysis.metalation_energy import stoich_energy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import AllChem as Chem

try:
    from materialbuilder.xyz2mol import xyz2mol
except:
    pass
from collections import OrderedDict
from munch import Munch

# from mendeleev import element as Element
from django import db
from scipy.linalg import block_diag
from materialbuilder.matbuilder import Z_TO_SYMBOLS, SYMBOLS_TO_Z


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


def geometry_from_geom(job_details, geom, return_adjmat=False):
    xyz = torch.tensor(geom.xyz)
    if xyz.shape[1] == 3:
        atomic_nums = geom.species.connectivitymatrix.atomicnums
        atomic_nums = torch.tensor(atomic_nums).view(-1, 1).to(torch.float)
        xyz = torch.cat([atomic_nums, xyz], dim=1)
        geom.xyz = xyz.tolist()
        geom.save()
    components = list(geom.species.components.all())
    if components:
        component_dict = {component.smiles: component for component in components}
        components = [component_dict[s] for s in geom.species.smiles.split(".")]
        target_atomic_nums = torch.tensor(geom.xyz)[:, 0].to(torch.long)
        for permutation in itertools.permutations(list(range(len(components)))):
            atomic_nums = []
            for i in permutation:
                atomic_nums.append(
                    torch.tensor(components[i].geom_set.first().xyz)[:, 0].to(
                        torch.long
                    )
                )
            atomic_nums = torch.cat(atomic_nums)
            if (atomic_nums == target_atomic_nums).all():
                break
        components = [components[i] for i in permutation]
        A, z = [], []
        for component in components:
            A_reference_component = get_reference_geometry(
                job_details, smiles=component.smiles
            )[1]
            A.append(A_reference_component)
            z.append(torch.tensor(np.array(component.geom_set.first().xyz)[:, 0]))
        A = [a.tolist() for a in A]
        A = torch.tensor(block_diag(*A)).to(torch.long)
        z = torch.cat(z)
    else:
        atomic_numbers, adjmat, dists = geom.adjdistmat()
        A = torch.tensor(adjmat)
        z = torch.tensor(np.array(geom.xyz)[:, 0])
        A = fix_A(A, z, dists)
    geometry = matbuilder.Geometry(
        A=A,
        atomic_nums=torch.tensor(geom.xyz)[:, 0].tolist(),
        xyz=torch.tensor(geom.xyz)[:, -3:].tolist(),
    )
    if return_adjmat:
        return (geometry, A)
    return geometry


def get_reference_geometry(job_details, smiles):
    species = Group.objects.get(name=job_details.group).species_set.get(smiles=smiles)
    geoms = species.geom_set.all()
    reference_geom = (
        geoms.filter(calcs__props__isnull=False)
        .order_by("calcs__props__" + job_details.energy_key)
        .first()
    )
    reference_geometry, A_reference = geometry_from_geom(
        job_details, reference_geom, return_adjmat=True
    )
    return (reference_geometry, A_reference)


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
    """
    for split in ['train', 'test']:
        inout.update_job_output(job_details.output_path, 
            keys='num_geoms_{}'.format(split), value=len(geom_ids[split]))
        inout.update_job_output(job_details.output_path, 
            keys='num_species_{}'.format(split), value=len(spec_ids[split]))
    """
    return (geom_ids, spec_ids)


def get_frames(
    job_details,
    geom_ids,
    show_output=False,
    skip_reorder=False,
    return_reference_geoms=False,
    source="database",
):
    geom_ids = list(itertools.chain(*[geom_ids[key] for key in geom_ids.keys()]))
    smileslist, speciesidlist = dbinterface.get_smiles_list(
        geom_ids, get_species_ids=True
    )
    reference_geoms = {}
    for counter, smiles in enumerate(smileslist):
        species_id = speciesidlist[counter]
        if source == "rdkit":
            reference_geometry, adjmat = matbuilder.geometry_from_smiles(
                smiles, return_adjmat=True
            )
        elif source == "xyz2mol":
            species = Group.objects.get(name=job_details.group).species_set.get(
                smiles=smiles
            )
            geoms = species.geom_set.all()
            reference_geom = (
                geoms.filter(calcs__props__isnull=False)
                .order_by("calcs__props__" + job_details.energy_key)
                .first()
            )
            xyz = torch.tensor(reference_geom.xyz)
            rdkit_mol = xyz2mol.xyz2mol(
                xyz[:, 0].to(torch.long).tolist(),
                0,
                xyz[:, -3:].tolist(),
                True,
                True,
                False,
            )
            atomic_nums = [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
            A = Chem.GetAdjacencyMatrix(rdkit_mol)
            reference_geometry = matbuilder.Geometry(atomic_nums, xyz, A=A)
            inring = [int(bool(atom.IsInRing())) for atom in rdkit_mol.GetAtoms()]
            aromatic = [
                int(bool(atom.GetIsAromatic())) for atom in rdkit_mol.GetAtoms()
            ]
            reference_geometry.AddNodeLabel("inring", inring, torch.long)
            reference_geometry.AddNodeLabel("aromatic", aromatic, torch.long)
            adjmat = torch.tensor(A.tolist())
        elif source == "database":
            reference_geometry, adjmat = get_reference_geometry(
                job_details, smiles=smiles
            )
        atomic_numbers = reference_geometry.atomic_nums.view(-1).tolist()
        reference_geoms[species_id] = OrderedDict(
            {
                "geometry": reference_geometry,
                "atomic_nums": atomic_numbers,
                "A": adjmat.tolist(),
            }
        )
        species = Species.objects.get(id=species_id)
        reference_geoms[species_id]["E_ref"] = stoich_energy(
            formula=species.stoichiometry.formula
        )  # Ha
    mol_ref = None
    frames = Munch(
        {
            "geom_id": [],
            "graph": [],
            "atomic_nums": [],
            "xyz": [],
            "charge": [],
            "dipole": [],
            "E": [],
            "E_ref": [],
            "F": [],
            "singlets": [],
        }
    )
    if source == "xyz2mol":
        frames.inring = []
        frames.aromatic = []
    for counter, geom_id in enumerate(geom_ids):
        if source == "rdkit":
            geom_data = Munch(
                {
                    "A": None,
                    "xyz": None,
                    "charge": None,
                    "dipole": None,
                    "E": None,
                    "E_ref": None,
                    "F": None,
                    "singlets": None,
                }
            )
            species_id = Geom.objects.get(id=geom_id).species.id
            geom_data.A = reference_geoms[species_id]["A"]
            geom_data.atomic_nums = reference_geoms[species_id]["atomic_nums"]
            geom_data.xyz = reference_geoms[species_id]["geometry"].xyz.tolist()
            geom_data.A = torch.tensor(geom_data.A).to(torch.long)
            geom_data.atomic_nums = torch.tensor(geom_data.atomic_nums).to(torch.long)
            geom_data.xyz = torch.tensor(geom_data.xyz)
        elif source in ["xyz2mol", "database"]:
            geom_data = dbinterface.get_reordered_geom_data(
                job_details, geom_id, mol_ref, reference_geoms, skip_reorder=True
            )
            geom_data.A = torch.tensor(geom_data.A).to(torch.long)
            geom_data.atomic_nums = torch.tensor(geom_data.xyz)[:, 0].to(torch.long)
            geom_data.xyz = torch.tensor(geom_data.xyz)[:, -3:]
            if source == "xyz2mol":
                species_id = Geom.objects.get(id=geom_id).species.id
                geom_data.inring = (
                    reference_geoms[species_id]["geometry"]
                    .get_data("node", "inring")
                    .view(-1)
                )
                geom_data.aromatic = (
                    reference_geoms[species_id]["geometry"]
                    .get_data("node", "aromatic")
                    .view(-1)
                )
        geom_data.geom_id = geom_id
        if "charge" in job_details.properties_to_calculate:
            geom_data.charge = torch.tensor(geom_data.charge)[:, [0]]
        if "dipole" in job_details.properties_to_calculate:
            geom_data.dipole = torch.tensor(geom_data.dipole)
        if "E" in job_details.properties_to_calculate:
            geom_data.E = torch.tensor(geom_data.E).view(1, 1)  # Ha
            geom_data.E = geom_data.E * units.HARTREE / units.KCAL_MOL  # kcal/mol
            if job_details.subtract_reference_energy:
                geom_data.E_ref = torch.tensor(geom_data.E_ref).view(1, 1)  # Ha
                geom_data.E_ref = (
                    geom_data.E_ref * units.HARTREE / units.KCAL_MOL
                )  # kcal/mol
        if "F" in job_details.properties_to_calculate:
            geom_data.F = torch.tensor(geom_data.F)  # Ha/(Bohr radius)
            geom_data.F = (
                geom_data.F * (units.HARTREE / units.BOHR) / units.KCAL_MOL
            )  # kcal/mol-Angstrom
        if "singlets" in job_details.properties_to_calculate:
            geom_data.singlets = torch.tensor(geom_data.singlets)
        geom_graph = matbuilder.Geometry(
            A=geom_data.A, atomic_nums=geom_data.atomic_nums, xyz=geom_data.xyz
        )
        keys = ["geom_id", "atomic_nums", "xyz"]
        if "charge" in job_details.properties_to_calculate:
            keys.append("charge")
        if "dipole" in job_details.properties_to_calculate:
            keys.append("dipole")
        if "E" in job_details.properties_to_calculate:
            keys.append("E")
            if job_details.subtract_reference_energy:
                keys.append("E_ref")
        if "F" in job_details.properties_to_calculate:
            keys.append("F")
        if "singlets" in job_details.properties_to_calculate:
            keys.append("singlets")
        if source == "xyz2mol":
            keys.append("inring")
            inring = (
                reference_geoms[species_id]["geometry"]
                .get_data("node", "inring")
                .view(-1)
            )
            geom_graph.AddNodeLabel("inring", inring.tolist(), torch.long)
            geom_data.inring = inring
            keys.append("aromatic")
            aromatic = (
                reference_geoms[species_id]["geometry"]
                .get_data("node", "aromatic")
                .view(-1)
            )
            geom_graph.AddNodeLabel("aromatic", aromatic.tolist(), torch.long)
            geom_data.aromatic = aromatic
        for key in keys:
            frames[key].append(geom_data[key])
        frames["graph"].append(geom_graph)
        if show_output:
            if (counter + 1) % 10 == 0:
                print(counter + 1)
        for key in copy.deepcopy(list(frames.keys())):
            if frames[key] == []:
                frames.pop(key)
    if return_reference_geoms:
        return (frames, reference_geoms)
    return frames


def get_template_dataset(job_details, template_geoms_dataframe):
    dataset = graphbuilder.GraphDataset()
    for species_id, row in template_geoms_dataframe.iterrows():
        geoms = Geom.objects.filter(id__in=row.geom_ids)
        smiles = Species.objects.get(id=species_id).smiles
        geom_ref, A_ref = get_reference_geometry(job_details, smiles=smiles)
        xyz = torch.tensor(geom_ref.xyz.tolist())[:, -3:].tolist()
        atomic_nums = geom_ref.atomic_nums.view(-1).tolist()
        geometry = matbuilder.Geometry(A=A_ref, atomic_nums=atomic_nums, xyz=xyz)
        dataset.AddGraph(geometry)
        dataset.Close()
    for trans in job_details.transformations:
        dataset.AddTransformation(
            eval("transformations." + trans[0])(*tuple(trans[1:]))
        )
    db.connections.close_all()
    return dataset


def datasets_from_frames(
    job_details, geoms_dataframe, frames, reference_geoms=None, datasets=None
):
    if reference_geoms is not None:
        # species_ids = torch.tensor(
        #    geoms_dataframe['species_id'].tolist()).sort()[0].tolist()
        species_ids = torch.tensor(list(reference_geoms.keys())).sort()[0].tolist()
        # print(200); import IPython; IPython.embed()
        reference_geoms = OrderedDict(
            {species_id: reference_geoms[species_id] for species_id in species_ids}
        )
    ensembles = {"train": {}, "test": {}}
    for counter, geom_id in enumerate(frames.geom_id):
        geom = Geom.objects.get(id=geom_id)
        species_id = geom.species.id
        geometry = frames.graph[counter]
        if "E" in frames.keys():
            if job_details.subtract_reference_energy:
                geometry.AddProperty("E", frames.E[counter] - frames.E_ref[counter])
            else:
                geometry.AddProperty("E", frames.E[counter])
        if "F" in frames.keys():
            geometry.AddNodeLabel("F", frames.F[counter])
        if "charge" in frames.keys():
            geometry.AddNodeLabel("chelpg_charge", frames.charge[counter])
        if "singlets" in frames.keys():
            geometry.AddProperty("singlets", frames.singlets[counter])
        if counter < job_details.num_train:
            split = "train"
        elif counter < job_details.num_train + job_details.num_test:
            split = "test"
        else:
            break
        if species_id not in ensembles[split].keys():
            ensembles[split][species_id] = graphbuilder.Ensemble()
        geometry.AddProperty("species_id", species_id, torch.long)
        ensembles[split][species_id].Add(geometry)
    reference_geoms = OrderedDict(
        {
            ensemble_counter: reference_geoms[species_id]
            for ensemble_counter, species_id in enumerate(species_ids)
        }
    )
    if datasets is None:
        datasets = {}
    datasets = {
        **datasets,
        **{"train": graphbuilder.GraphDataset(), "test": graphbuilder.GraphDataset()},
    }
    for split in ["train", "test"]:
        for species_id in ensembles[split].keys():
            datasets[split].AddEnsemble(ensembles[split][species_id])
        datasets[split].Close()
        if "template" in datasets.keys():
            for _transformation in datasets["template"].transformations.keys():
                for transformation in datasets["template"].transformations[
                    _transformation
                ]:
                    datasets[split].AddTransformation(transformation)
        else:
            for trans in job_details.transformations:
                datasets[split].AddTransformation(
                    eval("transformations." + trans[0])(*tuple(trans[1:]))
                )
        if "base_node_type" in job_details.keys():
            datasets[split].DefineBaseNodeTypes(job_details.base_node_type)
        if "node_type" in job_details.keys():
            datasets[split].DefineNodeTypes(job_details.node_type)
        if "compass" in job_details.custom_types:
            datasets[split].AddTyper(typers.CompassTyper(reference_geoms))
        if "bond" in job_details.terms:
            datasets[split].AddTopology(topologies.BondTopology())
        if "angle" in job_details.terms:
            datasets[split].AddTopology(topologies.AngleTopology())
        if "dihedral" in job_details.terms:
            datasets[split].AddTopology(topologies.DihedralTopology())
        if "improper" in job_details.terms:
            datasets[split].AddTopology(topologies.ImproperTopology())
        if "pair" in job_details.terms:
            datasets[split].AddTopology(
                topologies.PairTopology(
                    job_details.use_1_4_pairs, job_details.pair_cutoff
                )
            )
        datasets[split].UnzipEnsembles()
        datasets[split].CreateBatches(
            batch_size=job_details.batch_size, shuffle=False, device=job_details.device
        )  # job_details.shuffle)
    # print('check ensembles after batching'); import IPython; IPython.embed()
    db.connections.close_all()
    return datasets


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
    if "compass" in job_details.custom_types:
        dataset.AddTyper(typers.CompassTyper())
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
