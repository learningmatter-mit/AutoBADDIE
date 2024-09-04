import os
import random, itertools, ast
from materialbuilder import (
    graphbuilder,
    matbuilder,
)
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
from rdkit.Chem import AllChem as Chem
import networkx as nx
from scipy.linalg import block_diag
import sklearn.cluster


def get_df(
    species,
    job_details,
    condition,
):
    """
    Returns a DataFrame of energies and forces for the Geoms of the given Species.
    The Geoms are filtered by geom_method and the Calcs are filtered by calc_method or config.
    The returned DataFrame is sorted by geom_id.
    """
    print("species is:", species.id, species.smiles)
    if job_details.calc_method == None:
        job_details.calc_method = "dft_hyb_wb97xd3"
    if job_details.geom_method == None:
        job_details.geom_method = "nn_dynamics_dft_d3_gga_bp86"

    print("starting get_df")
    df = pd.DataFrame(
        species.geom_set.values_list(
            "id",
            "method__name",
            "reactivecluster",
            "details__temperature",
            "details__generation",
            "details__ion_scale",
            "details__lj_dist_cluster",
            "calcs__parentjob__config__name",
            "calcs__method__name",
            "calcs__props__totalenergy",
            "calcs__jacobian__id",
            "parentjob__details__file",
            "parentjob__details__temperature",
            "calcs__props__atomiccharges__chelpg",
            "parentjob__parentgeom__id",
        ).distinct(),
        columns=[
            "geom_id",
            "geom_method",
            "geom_reactivecluster",
            "geom_temp",
            "geom_gen",
            "geom_ion_scale",
            "geom_lj_cluster",
            "config",
            "calc_method",
            "energy",
            "jacobian_id",
            "training_ff",
            "training_temp",
            "chelpg",
            "starting_id",
        ],
    )
    if job_details.config is not None:
        df = df[df["config"].isin(job_details.config)]
    # else:
    # df = df[df['calc_method'].isin(job_details.calc_method)]
    df = df[df["calc_method"].isin(job_details.calc_method)]
    print(job_details.geom_method)
    df = df[df["geom_method"].isin(job_details.geom_method)]
    counts = df.groupby("geom_id").count()
    keep_geomids = counts[counts["config"] <= 2].index
    df = df[df["geom_id"].isin(keep_geomids)]
    print(df["geom_method"].unique())
    df = df.drop("calc_method", axis=1)
    df = df.drop("config", axis=1)
    df = df.drop("geom_method", axis=1)
    df = df.drop("geom_temp", axis=1)
    df = df.drop("geom_lj_cluster", axis=1)
    df = df.drop("geom_reactivecluster", axis=1)
    df = df.drop("training_ff", axis=1)
    df = df.drop("training_temp", axis=1)
    df = df.drop("starting_id", axis=1)

    A = (
        df[~df["energy"].isnull()]
        .set_index("geom_id")
        .drop("jacobian_id", axis=1)
        .sort_values(by=["geom_id"])
    )
    B = (
        df[~df["jacobian_id"].isnull()]
        .set_index("geom_id")
        .drop("energy", axis=1)
        .sort_values(by=["geom_id"])
    )
    df = pd.concat([A, B], join="inner", axis=1)

    # only keep ones with atomic charges
    df = df[~df["chelpg"].isnull().all(axis=1)].drop("chelpg", axis=1)

    # get the max magnitude of force component for each geometry, and only keep geometries with max force below 200 kcal/molA
    jacs = Jacobian.objects.filter(id__in=df.jacobian_id).order_by("id")
    allforces = (
        torch.tensor(jacs.values_list("forces", flat=True))
        * 627.50947415
        / 0.52917721090380
    )
    try:
        maxes = allforces.abs().max(dim=1)[0].max(dim=1)[0]
    except:
        print(species.smiles)
        print(df)
        import ipdb

        ipdb.set_trace()

    df = df.sort_values("jacobian_id")
    df["maxF"] = maxes
    df_with_max = df[df.maxF >= 200]

    if species.smiles == "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]":
        # remove the random geoms where the Li is not either first or last (so cannot be permuted)
        jacs = Jacobian.objects.filter(id__in=df["jacobian_id"])
        allzs = np.array(jacs.order_by("id").values_list("geoms__xyz", flat=True))[
            :, :, 0
        ]
        mask = (allzs[:, -1] == 3) + (allzs[:, 0] == 3)
        df = df[mask]
    xyzs = []
    print("starting")
    for cur in df_with_max.index:
        try:
            curxyz = Geom.objects.get(id=int(cur)).as_xyz().split("\n")
            curxyz[1] = str(cur)
            curxyz = "\n".join(curxyz)
            xyzs.append(curxyz)
        except:
            continue
    with open(
        "/".join([job_details.WORKDIR, condition, "geoms_with_maxF.xyz"]), "w"
    ) as outfile:
        outfile.write("".join(xyzs))
    print(df_with_max.maxF)
    df = df[df.maxF < 200]
    df = df.drop("maxF", axis=1)

    df = df.reset_index(
        drop=False
    )  # makes the geom_id a column instead of the row indexes
    df = df[~df["geom_id"].duplicated(keep=False)]

    # remove broken geoms
    geomids = np.array(list(df.geom_id))
    print(
        "# of nnmd before:",
        Geom.objects.filter(
            id__in=geomids, method__name="nn_dynamics_dft_d3_gga_bp86"
        ).count(),
    )
    # if species.id in [5386711, 5387528]:
    #     print("number samples before excluding broken:", df.shape[0])
    #     geoms = Geom.objects.filter(id__in=df["geom_id"])
    #     nonbroken = list(get_nonbroken_ids(geoms).values_list("id", flat=True))
    #     df = df[df["geom_id"].isin(nonbroken)]
    geomids = np.array(list(df.geom_id))
    # print('# of nnmd after:', Geom.objects.filter(id__in=geomids, method__name='nn_dynamics_dft_d3_gga_bp86').count())
    print("available number of samples:", df.shape[0])

    if (
        species.smiles == "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]"
        and "pcff" in job_details.job_name
        and "only" in job_details.date
    ):
        newgeomids = get_only_nitrogen_clusters(geomids, job_details, condition, df)
        df = df[df["geom_id"].isin(newgeomids)]
        print("available number of samples in Li-N cluster:", df.shape[0])
    if df.shape[0] > job_details.geoms_per_species:
        df = df.sample(job_details.geoms_per_species)
    df = df.set_index("geom_id")
    df = df.sort_values(by=["geom_id"])
    df["jacobian_id"] = df["jacobian_id"].astype(int)
    print("number of samples:", df.shape[0])
    print(
        Geom.objects.filter(id__in=df.index)
        .values_list("method__name")
        .annotate(count=Count("method"))
    )
    xyzs = []
    for cur in df.index:
        try:
            curxyz = Geom.objects.get(id=int(cur)).as_xyz().split("\n")
            curxyz[1] = str(cur)
            curxyz = "\n".join(curxyz)
            xyzs.append(curxyz)
        except:
            continue
    with open(
        "/".join([job_details.WORKDIR, condition, "all_geoms_used.xyz"]), "w"
    ) as outfile:
        outfile.write("".join(xyzs))
    # if species.smiles == 'COCCOCCOCCOCCOC':
    #     job_details.geom_method = old_geom_method
    #     job_details.geoms_per_species = old_geoms_per_species
    return df


def get_only_nitrogen_clusters(geomids, job_details, condition, df):
    geoms, example_zs, zs, xyzs, qs, ref_anion_z, neg_mask = get_props_of_geoms(
        geomids, job_details, df
    )
    li_neg_d, labels = get_li_neg_d(xyzs, example_zs, ref_anion_z, neg_mask)
    newgeomids = filter_geoms(
        job_details, condition, geoms, li_neg_d, labels, qs, ref_anion_z
    )
    return newgeomids


def get_props_of_geoms(geomids, job_details, df):
    geoms = Geom.objects.filter(
        id__in=geomids,
        childjobs__childcalcs__props__atomiccharges__isnull=False,
        childjobs__config__name__in=job_details.config,
        childjobs__childcalcs__method__name__in=job_details.calc_method,
    )
    print("number of geometries:", geoms.count())
    example_zs = np.array(geoms.first().xyz, dtype=int)[:, 0]
    zs = np.array(list(geoms.values_list("xyz", flat=True)))[:, :, 0]
    mask = [not np.array_equal(example_zs, cur) for cur in zs]
    xyzs = np.array(list(geoms.values_list("xyz", flat=True)))[:, :, 1:]
    xyzs[mask] = np.concatenate([xyzs[mask][:, 15:], xyzs[mask][:, :15]], axis=1)
    # qs = np.array(list(geoms.filter(childjobs__childcalcs__props__atomiccharges__isnull=False,
    #                     childjobs__config__name__in=job_details.config,
    #                     childjobs__childcalcs__method__name__in=job_details.calc_method).values_list('childjobs__childcalcs__props__atomiccharges__chelpg',flat=True)))
    qs = np.array(
        list(
            geoms.values_list(
                "childjobs__childcalcs__props__atomiccharges__chelpg", flat=True
            )
        )
    )
    qs[mask] = np.concatenate([qs[mask][:, 15:], qs[mask][:, :15]], axis=1)
    neg_mask = qs[0] < 0
    ref_anion_z = example_zs[neg_mask]
    return geoms, example_zs, zs, xyzs, qs, ref_anion_z, neg_mask


def get_li_neg_d(xyzs, example_zs, ref_anion_z, neg_mask):
    li_neg_d = []
    li_inx = example_zs == 3
    for curxyz in xyzs:
        curlixyz = curxyz[li_inx]
        cur_neg_xyzs = curxyz[neg_mask]
        cur_d = np.sqrt(np.power(curlixyz - cur_neg_xyzs, 2).sum(axis=1))
        df = pd.DataFrame({"z": ref_anion_z, "d": cur_d}).sort_values(["z", "d"])
        li_neg_d.append(df["d"].tolist())
    li_neg_d = np.array(li_neg_d)
    dbscan = sklearn.cluster.DBSCAN(
        eps=1.125,
        min_samples=li_neg_d[:, :-5].shape[1] * 2,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    )
    labels = dbscan.fit_predict(li_neg_d[:, :-5])
    return li_neg_d, labels


def are_ismomorphic(A1, A2):
    """
    Uses networkx to determine if the provided adjacency matrices
    describe two isomorphic graphs.
    """
    G1 = nx.DiGraph()
    G1.add_edges_from(A1.nonzero().tolist())
    G2 = nx.DiGraph()
    G2.add_edges_from(A2.nonzero().tolist())
    if nx.is_isomorphic(G1, G2):
        return True
    return False


CPK_COLORS = {
    0: (np.array([0, 0, 0]) / 255).tolist(),
    1: (np.array([255, 255, 255]) / 255).tolist(),
    2: (np.array([217, 255, 255]) / 255).tolist(),
    3: (np.array([204, 128, 255]) / 255).tolist(),
    4: (np.array([194, 255, 0]) / 255).tolist(),
    5: (np.array([255, 181, 181]) / 255).tolist(),
    6: (np.array([144, 144, 144]) / 255).tolist(),
    7: (np.array([48, 80, 248]) / 255).tolist(),
    8: (np.array([255, 13, 13]) / 255).tolist(),
    9: (np.array([144, 224, 80]) / 255).tolist(),
    10: (np.array([179, 227, 245]) / 255).tolist(),
    11: (np.array([171, 92, 242]) / 255).tolist(),
    12: (np.array([138, 255, 0]) / 255).tolist(),
    13: (np.array([191, 166, 166]) / 255).tolist(),
    14: (np.array([240, 200, 160]) / 255).tolist(),
    15: (np.array([255, 128, 0]) / 255).tolist(),
    16: (np.array([255, 255, 48]) / 255).tolist(),
    17: (np.array([31, 240, 31]) / 255).tolist(),
    18: (np.array([128, 209, 227]) / 255).tolist(),
    19: (np.array([143, 64, 212]) / 255).tolist(),
    20: (np.array([61, 255, 0]) / 255).tolist(),
    25: (np.array([156, 122, 199]) / 255).tolist(),
    26: (np.array([224, 102, 51]) / 255).tolist(),
    27: (np.array([240, 144, 160]) / 255).tolist(),
    28: (np.array([80, 208, 80]) / 255).tolist(),
    29: (np.array([200, 128, 51]) / 255).tolist(),
    30: (np.array([125, 128, 176]) / 255).tolist(),
    34: (np.array([255, 161, 0]) / 255).tolist(),
    35: (np.array([166, 41, 41]) / 255).tolist(),
    47: (np.array([192, 192, 192]) / 255).tolist(),
    53: (np.array([148, 0, 148]) / 255).tolist(),
    55: (np.array([87, 23, 143]) / 255).tolist(),
    79: (np.array([255, 209, 35]) / 255).tolist(),
}

BOND_TYPE_TO_COLOR = {
    0: "red",
    1: "brown",
    2: "pink",
    3: "green",
    4: "purple",
    5: "orange",
    6: "blue",
    7: "black",
    8: "yellow",
    9: "gray",
}


def create_dataset(
    template_dataset,
    job_details,
    trajectory,
    reference_adjmat,
    N,
    forces,
    Qs,
    poles,
    energies,
    geom_ids,
):
    dataset = graphbuilder.GraphDataset()
    for species_id in trajectory.keys():
        ensemble = graphbuilder.Ensemble()
        for n in range(
            len(trajectory[species_id])
        ):  # want as many geometries as possible
            xyz = trajectory[species_id][n][:, -3:]
            z = trajectory[species_id][n][:, 0]
            geometry = matbuilder.Geometry(
                atomic_nums=z, xyz=xyz, A=reference_adjmat[species_id]
            )
            geometry.AddNodeLabel("F", forces[species_id][n].tolist())
            geometry.AddProperty(
                "geom_id", geom_ids[species_id][n].tolist(), dtype=torch.int
            )
            geometry.AddProperty("energy", energies[species_id][n].tolist())
            num_nodes = N[species_id]
            if job_details.learn_charge_flag:
                q = Qs[species_id][n]  # sets DFT charges as target
                geometry.AddNodeLabel("q", q.tolist())
            c = []
            for counter, _n in enumerate(num_nodes):
                c.append([counter for x in range(_n)])
            c = list(itertools.chain(*c))
            c = torch.tensor(c).to(torch.long)
            geometry.AddNodeLabel("component", c.tolist())
            ensemble.Add(geometry)
        dataset.AddEnsemble(ensemble)
    dataset.Close()
    # at this point, dataset does not have any topologies or typse
    for _transformation in (
        template_dataset.transformations.keys()
    ):  # ['OneHotEncoding', 'GraphDistinctNodes']
        for transformation in template_dataset.transformations[
            _transformation
        ]:  # eachGraphDistinctNodes has inlabel,outlabel,radius,unique_values
            dataset.AddTransformation(transformation, job_details.device)
    # now the dataset._data['node'] has the onehotencoding and graphdistinctnodes, but only in one-hot mode, not as type

    base_node_type = template_dataset.container_batch.base_node_type_label
    if base_node_type is not None:
        dataset.DefineBaseNodeTypes(base_node_type)
    node_type = template_dataset.container_batch.node_type_label
    if node_type is not None:
        dataset.DefineNodeTypes(
            node_type
        )  # adds dataset._data['node']['type'] based on dataset._data['node']['one_hot_graph_distinct_r#']
    for topology in job_details.terms:  #'bond', 'angle', 'dihedral', 'improper', 'pair'
        dataset.container_batch.topologies.update(
            {topology: template_dataset.container_batch.topologies[topology]}
        )
    #
    dataset.UnzipEnsembles(Qs)
    dataset.CreateBatches(
        batch_size=job_details.batch_size, shuffle=True, device=job_details.device
    )
    return dataset


top_to_param = {
    "bond": ["k_b21", "k_b3", "k_b4", "k_b22", "b0_tar"],
    "angle": ["k_a21", "k_a3", "k_a4", "k_a22", "k_bb", "k_ba0", "k_ba1", "theta0_tar"],
    "dihedral": [
        "k1",
        "k2",
        "k3",
        "k_mt1",
        "k_mt2",
        "k_mt3",
        "k_bct10",
        "k_bct20",
        "k_bct30",
        "k_bct11",
        "k_bct21",
        "k_bct31",
        "k_at10",
        "k_at20",
        "k_at30",
        "k_at11",
        "k_at21",
        "k_at31",
        "k_b1b3",
        "k_aat",
    ],
    "improper": ["k_improper", "k_aa1", "k_aa2", "k_aa3"],
    "charge": ["charge"],
}
top_default = {
    "bond": -10,
    "angle": -10,
    "dihedral": 0,
    "improper": 0,
    "charge": np.nan,
}
top_reference_param = {
    "bond": "b0_tar",
    "angle": "theta0_tar",
    "dihedral": "k_aat",
    "improper": "k_aa3",
    "charge": "charge",
}


def combine_pretrained_ffs(glymeff, tfsiff, liff, job_details=None):
    param_keys = [
        "k_b21",
        "k_b3",
        "k_b4",
        "k_b22",
        "b0_tar",
        "k_a21",
        "k_a3",
        "k_a4",
        "k_a22",
        "k_bb",
        "k_ba0",
        "k_ba1",
        "theta0_tar",
        "k1",
        "k2",
        "k3",
        # 'k1', 'k2', 'k3', 'phi1', 'phi2', 'phi3',
        "k_mt1",
        "k_mt2",
        "k_mt3",
        "k_bct10",
        "k_bct20",
        "k_bct30",
        "k_bct11",
        "k_bct21",
        "k_bct31",
        "k_at10",
        "k_at20",
        "k_at30",
        "k_at11",
        "k_at21",
        "k_at31",
        "k_b1b3",
        "k_aat",
        "k_improper",
        "k_aa1",
        "k_aa2",
        "k_aa3",
        "charge",
    ]

    # the "pretrained_ff" should begin with only glyme parameters
    # cur_ff will be the li.tfsi parameters in this case
    pretrained_ff = glymeff
    # for cur_ff in [liff, tfsiff]:
    for cur_ff in [tfsiff]:
        # for cur_ff in [tfsiff, liff]:
        for curkey in param_keys:
            print(curkey)
            if curkey in top_to_param["bond"]:
                cur_topology_name = "bond"
            elif curkey in top_to_param["angle"]:
                cur_topology_name = "angle"
            elif curkey in top_to_param["dihedral"]:
                cur_topology_name = "dihedral"
            elif curkey in top_to_param["improper"]:
                cur_topology_name = "improper"
            elif curkey in top_to_param["charge"]:
                cur_topology_name = "charge"
            default_param = top_default[cur_topology_name]
            # get the indices of parameters that have already been learned for both the current ff
            # and the combined pretrained ff
            pretrained_np = (
                pretrained_ff.__dict__[top_reference_param[cur_topology_name]]
                .detach()
                .cpu()
                .numpy()
            )
            cur_np = (
                cur_ff.__dict__[top_reference_param[cur_topology_name]]
                .detach()
                .cpu()
                .numpy()
            )
            if curkey == "charge":
                print("new charge stuff")
                # the charge has been learned in any index where the learned charge is NOT none, isnan is False
                # nonlearned charges can be 0 or null
                pretrained_np[pretrained_np == 0] = np.nan
                pretrainedff_inx = set(np.where(np.isnan(pretrained_np) == False)[0])
                curff_inx = set(np.where(np.isnan(cur_np) == False)[0])
            else:
                pretrainedff_inx = set(np.where(pretrained_np != default_param)[0])
                curff_inx = set(np.where(cur_np != default_param)[0])

            # set.intersection, set.difference
            totalset = set(range(len(pretrained_np)))
            learnedset = pretrainedff_inx.union(curff_inx)
            combined_params = []
            print("totalset, learnedset", totalset, learnedset)
            for cur_inx in totalset:
                # if neither of them have been learned, set to default which will be overwritten during ff initialization:
                if cur_inx in totalset.difference(learnedset):
                    print("it was neither! saving:", default_param)
                    combined_params.append(default_param)
                # if the param was learned by both forcefields, take the tfsi param because we only want to use the glyme benchmark params:
                elif cur_inx in curff_inx.intersection(pretrainedff_inx):
                    print(
                        "it was in both! saving only glyme tho:",
                        pretrained_ff.__dict__[curkey][cur_inx].item(),
                    )
                    combined_params.append(
                        pretrained_ff.__dict__[curkey][cur_inx].item()
                    )
                # if param was learned by only one, save that learned parameter and ignore the default of the other:
                else:
                    if cur_inx in curff_inx:
                        print(
                            cur_inx,
                            "it was in curff (ion) saving:",
                            cur_ff.__dict__[curkey][cur_inx].item(),
                        )
                        combined_params.append(cur_ff.__dict__[curkey][cur_inx].item())
                    else:
                        print(
                            cur_inx,
                            "it was in pretrained (glyme) saving:",
                            pretrained_ff.__dict__[curkey][cur_inx].item(),
                        )
                        combined_params.append(
                            pretrained_ff.__dict__[curkey][cur_inx].item()
                        )
            # save the combined paramters to the pretrainedff
            pretrained_ff.__dict__[curkey] = torch.tensor(
                combined_params, requires_grad=True, dtype=torch.float
            )
            if curkey == "charge":
                print(curkey, pretrained_ff.__dict__[curkey])
    return pretrained_ff
