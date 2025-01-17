import os, subprocess, ast
from munch import Munch

from autobaddie.materialbuilder.plotting import CPK_COLORS
from autobaddie.materialbuilder import (
    graphbuilder,
    matbuilder,
    topologies,
    transformations,
)
from rdkit.Chem import AllChem as Chem
import torch
import pandas as pd
from autobaddie.utils.constants import Z_TO_ELEMENT

import numpy as np
import matplotlib.pyplot as plt

"""

"""

# -------------------------these need to be set
WORKDIR_BASE = # path to AutoBADDIE directory (ending in [...]/AutoBADDIE)
job_name = "autoBADDIE_example" # desired name for chemical system, ex. "autoBADDIE_example"
node_type = "one_hot_graph_distinct_r3"  # the nubmer after the r refers to neighbor depth, ex "one_hot_graph_distinct_r3" or "one_hot_graph_distinct_r2"

template_smiles = "COCCOCCOCCOCCOC.O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F.[Li+]"  # atomic environments to be learned in smiles format


# ------------begin populating data
job_details = Munch()
job_details.job_name = job_name
job_details.device = "cpu"

job_details.template_smiles = template_smiles
job_details.transformations = [
    ["OneHotEncoding", "node", "atomic_num", "one_hot_atomic_num"],
    ["GraphDistinctNodes", "one_hot_atomic_num", "one_hot_graph_distinct_r1", 1],
    ["GraphDistinctNodes", "one_hot_atomic_num", "one_hot_graph_distinct_r2", 2],
    ["GraphDistinctNodes", "one_hot_atomic_num", "one_hot_graph_distinct_r3", 3],
    ["GraphDistinctNodes", "one_hot_atomic_num", "one_hot_graph_distinct_r4", 4],
]

job_details.base_node_type = "one_hot_atomic_num"
job_details.node_type = node_type

if "pcff" in job_details.job_name:
    job_details.forcefield_class2 = True
else:
    job_details.forcefield_class2 = False

job_details.terms = ["bond", "angle", "dihedral", "improper", "pair"]
job_details.use_1_4_pairs = True
job_details.pair_cutoff = None


# ----------start code
job_details.WORKDIR = os.path.join(
    WORKDIR_BASE, "training_results", job_details.job_name, "template"
)
print(job_details.WORKDIR)
if not os.path.exists(job_details.WORKDIR):
    os.makedirs(job_details.WORKDIR)

# -------------create template dataset
template_dataset = graphbuilder.GraphDataset()
ensemble = graphbuilder.Ensemble()
geometry, rdkit_mol = matbuilder.geometry_from_smiles(
    smiles=job_details.template_smiles, dim=2, return_rdkit_mol=True
)
ensemble.Add(geometry)
template_dataset.AddEnsemble(ensemble)
template_dataset.Close()

# write the template molecule as a mol file
path = os.path.join(job_details.WORKDIR, "template.mol")
file = open(path, "w")
file.write(Chem.MolToMolBlock(rdkit_mol))
file.close()
mol_path = os.path.join(job_details.WORKDIR, "template.mol")

# ----------------find atomic environments and multibody topology types
for trans in job_details.transformations:  # "one_hot_graph_distinct_r2"
    # this will create the different node types based on graph neighborhood. this runs transformation.ApplyTo(template_dataset.container_batch)
    template_dataset.AddTransformation(
        eval("transformations." + trans[0])(*tuple(trans[1:])), job_details.device
    )
if "base_node_type" in job_details.keys():
    template_dataset.DefineBaseNodeTypes(job_details.base_node_type)
if "node_type" in job_details.keys():
    template_dataset.DefineNodeTypes(job_details.node_type)
# these dataset.AddTopology() calls create the topology types based on node type using Graph.AddTopology()
# this uses each Topology.apply_to()
if "bond" in job_details.terms:
    template_dataset.AddTopology(topologies.BondTopology(), job_details.device)
if "angle" in job_details.terms:
    template_dataset.AddTopology(topologies.AngleTopology(), job_details.device)
if "dihedral" in job_details.terms:
    template_dataset.AddTopology(topologies.DihedralTopology(), job_details.device)
if "improper" in job_details.terms:
    template_dataset.AddTopology(topologies.ImproperTopology(), job_details.device)
if "pair" in job_details.terms:
    template_dataset.AddTopology(
        topologies.PairTopology(
            job_details.use_1_4_pairs,
            job_details.forcefield_class2,
            job_details.pair_cutoff,
        ),
        job_details.device,
    )

# save the template dataset
path = os.path.join(job_details.WORKDIR, "template.py")
torch.save(template_dataset, path)

# Create Custom Parameter File
xyz = template_dataset.container_batch._data["node"]["xyz"]
bonds = template_dataset.container_batch._data["bond"]["index"]
types = template_dataset.container_batch._data["node"]["type"].view(-1)
z = template_dataset.container_batch._data["node"]["atomic_num"].view(-1)
z_t = torch.stack([z, types], axis=1).unique(dim=0).tolist()
types = types.tolist()
z = z.tolist()


fig = plt.figure(figsize=(5, 10))
ax = fig.add_subplot(111)
ax.set_title("Atom Types")
ax.set_xlim([-5, 5])
ax.set_ylim([0, 100])

y = np.linspace(0, 100, len(z_t) + 2).tolist()[::-1]
for n in range(len(z_t)):
    element = Z_TO_ELEMENT[z_t[n][0]]
    ax.scatter(
        0,
        y[n + 1],
        s=500,
        color=CPK_COLORS[z_t[n][0]],
        alpha=1.0,
        edgecolors="black",
        lw=1,
    )
    ax.annotate(element + str(z_t[n][1]), (1, y[n + 1] - 1), color="black", size=15)

plt.grid(False)
plt.axis("off")
plt.savefig(os.path.join(job_details.WORKDIR, "atom_types_list.png"))
plt.close()

fig = plt.figure(figsize=(50, 30))
ax = fig.add_subplot(111)
ax.set_title(job_details.template_smiles)

x = xyz[:, 0].view(-1).tolist()
y = xyz[:, 1].view(-1).tolist()

z_t = torch.stack(
    [
        template_dataset.container_batch._data["node"]["atomic_num"].view(-1),
        template_dataset.container_batch._data["node"]["type"].view(-1),
    ],
    axis=1,
)
z_t = z_t.tolist()
for bond in bonds:
    bond = bond.view(-1).tolist()
    ax.plot(
        [x[bond[0]], x[bond[1]]], [y[bond[0]], y[bond[1]]], c="black", zorder=0, lw=10
    )

for n in range(len(x)):
    ax.scatter(
        x[n], y[n], s=12500, color=CPK_COLORS[z[n]], alpha=1.0, edgecolors="black", lw=5
    )
    ax.annotate(str(z_t[n][1]), (x[n] - 0.45, y[n] - 0.1), color="black", size=75)

plt.grid(False)
plt.axis("off")
print(job_details.WORKDIR)
plt.savefig(os.path.join(job_details.WORKDIR, "atom_types.png"))
plt.close()
