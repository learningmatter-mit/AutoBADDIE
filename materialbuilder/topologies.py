import torch, itertools, copy
from scipy.linalg import block_diag
import numpy as np
from collections import OrderedDict
import materialbuilder
from materialbuilder import topcalc
import ipdb


def index_of(inp, source, max_index):
    X = torch.randint(0, 9999999999, (max_index,))
    inp = X[inp].sum(1)
    source = X[source].sum(1)
    source, sorted_index, inverse = np.unique(
        source.tolist(), return_index=True, return_inverse=True
    )
    index = torch.cat([torch.tensor(source), inp]).unique(
        sorted=True, return_inverse=True
    )[1][-len(inp) :]
    index = torch.tensor(sorted_index)[index]
    return index


class Topology:
    def __init__(self):
        self.name = None
        self.unique_values = {"base_type": None, "type": None}
        self.num_node_types = {"base_type": None, "type": None}

    def _get_indices(self, graph):
        return (indices, num)

    def _get_values(self, graph, create_graph, device):
        return values

    def index_of(self, vectors, source):
        source, sorted_index, inverse = np.unique(
            source.tolist(), return_index=True, return_inverse=True, axis=0
        )
        index = torch.cat(
            [torch.tensor(source, device=vectors.device), vectors]
        ).unique(sorted=True, return_inverse=True, dim=0)[1][-len(vectors) :]
        index = torch.tensor(sorted_index, device=index.device)[index]
        return index

    def _get_types(self, graph, use_base_type=False, use_typer=False):
        return self.types

    def _get_num(self, graph, indices):
        if len(indices) == 0:
            num = torch.zeros_like(graph._data["num"]["node"]).view(-1, 1)
            return num
        num_nodes = graph._data["num"]["node"].view(-1).to(indices.device)
        num = indices[:, [0]] - (num_nodes.cumsum(0) - num_nodes)
        num = (num >= 0).to(torch.long)
        num = num.sum(1) - 1
        num_graphs = len(graph._data["num"]["node"])
        graph_indices = torch.arange(num_graphs).view(-1, 1).to(indices.device)
        num = (num == graph_indices).to(torch.long).sum(1)
        num = num.view(-1, 1)
        return num

    def _get_unshifted_indices(self, graph):
        num_nodes = graph._data["num"]["node"].view(-1)
        node_index_shift = num_nodes.cumsum(0) - num_nodes
        node_index_shift = node_index_shift.tolist()
        indices = self._get_indices(graph)
        num = self._get_num(graph, indices)
        indices = indices.split(num.view(-1).tolist())
        indices = list(indices)
        for b in range(len(num)):
            indices[b] = indices[b] - node_index_shift[b]
        indices = torch.cat(indices)
        return (indices, num)

    def apply_to(self, graph, device, type_keys=[]):
        if self.name not in graph._data.keys():
            graph._data[self.name] = {}
            indices, num = self._get_unshifted_indices(graph)
            graph._data[self.name]["index"] = indices
            graph._data["num"][self.name] = num
        graph._data[self.name] = {
            **graph._data[self.name],
            **self._get_values(graph, create_graph=False, device=device),
        }
        if "type" in graph._data["node"].keys():
            types = self._get_types(graph, type_key="type")
            graph._data[self.name]["type"] = types
        if "base_type" in graph._data["node"].keys():
            base_types = self._get_types(graph, type_key="base_type")
            graph._data[self.name]["base_type"] = base_types
        if graph.typer is not None:
            custom_types = self._get_types(graph, type_key=graph.typer.name)
            graph._data[self.name][graph.typer.name] = custom_types

    def apply_values_to(self, graph):
        graph._data[self.name] = {
            **graph._data[self.name],
            **self._get_values(graph, create_graph=False),
        }

    def __call__(self, graph, create_graph=False):
        data = graph.get_data(self.name)
        if create_graph:
            data = {**data, **self._get_values(graph, create_graph=True)}
        data["num"] = graph._data["num"][self.name]
        return data


class BondTopology(Topology):
    def __init__(self):
        super(BondTopology, self).__init__()
        self.name = "bond"

    def _get_indices(
        self, graph
    ):  # indices of a bond are the indices of the two nodes that create the bond
        bonds = graph.get_data("edge", "index")
        bonds = bonds[bonds[:, 0] < bonds[:, 1]].tolist()
        bonds = torch.LongTensor(bonds)
        bonds = bonds.view(-1, 2)
        return bonds

    def _get_values(self, graph, create_graph=False, device="cpu"):
        values = {}
        bonds = graph.get_data(self.name, "index")
        if len(bonds) != 0:
            if create_graph:
                xyz = graph._data["node"]["xyz"]
                graph._data["node"]["xyz"] = torch.tensor(
                    xyz.tolist(),
                    requires_grad=True,
                    dtype=torch.float,
                    device=bonds.device,
                )
            xyz = graph._data["node"]["xyz"]
            b = topcalc.get_bond_geometry(xyz, bonds)
            db = topcalc.get_bond_derivatives(xyz, bonds)
            values["b"] = b.view(-1, 1)
            values["db"] = db.view(-1, 2 * 3)
        else:
            values["b"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["db"] = torch.tensor([]).view(-1, 2 * 3).to(torch.float)
        return values

    def _get_types(self, graph, type_key):
        if graph.typer is not None and type_key == graph.typer.name:
            custom_types = graph.typer.get_types(graph, self.name)
            return custom_types
        # node_types is a list of the type of each node
        node_types = graph.get_data("node", type_key).view(-1)
        # top_types contains the node types of each node inside this topology
        top_types = node_types[graph.get_data(self.name, "index")]
        if len(top_types) == 0:
            return torch.tensor([]).view(-1, 1)
        if self.num_node_types[type_key] is None:
            self.num_node_types[type_key] = (
                graph.get_data("node", type_key).max().item() + 1
            )
        # create a diagonal square array whose dimension is the number of node types (one hot encoding of the possible node types)
        # then changes top_types to be these one-hot-encoding representations fo the different possible node types by indexing [top_types]
        # import ipdb
        # ipdb.set_trace()
        try:
            top_types = torch.eye(
                self.num_node_types[type_key], device=top_types.device
            )[top_types].to(torch.long)
        except:
            import pdb

            pdb.set_trace()

        # sums the two one-hot encodings to allow for permutation invariance in bond-types
        top_types = top_types.sum(1)
        if self.unique_values[type_key] is None:
            # top_types is now the integer index of the distinct two-hot encodings (sum of 2 one-hot encodings),
            # while self.unique_values[type_key] now holds the distinct two-hot encodings...
            # so like dataset.topologies['bond'].unique_values['type'][0] will have the 9th and 12th index be 1 and
            # the rest 0 if the bond contains node type 9 and 12!
            self.unique_values[type_key], top_types = top_types.unique(
                dim=0, return_inverse=True
            )
        else:
            # this index_of is what actually assigns the topology type to a particular topology,
            # because it will give the index of the one-hot encoding from self.unique_values[type_key]
            top_types = self.index_of(
                vectors=top_types, source=self.unique_values[type_key]
            )
        types = top_types.view(-1, 1)
        return types


class AngleTopology(Topology):
    def __init__(self):
        super(AngleTopology, self).__init__()
        self.name = "angle"

    def _get_indices(
        self, graph
    ):  # indices of an angle are the indices of the three nodes that create the angle
        angles = [list(itertools.combinations(x, 2)) for x in graph.neighbors]
        angles = [
            [[pair[0]] + [i] + [pair[1]] for pair in pairs]
            for i, pairs in enumerate(angles)
        ]
        angles = list(itertools.chain(*angles))
        angles = torch.tensor(
            angles, dtype=torch.int64, device=graph._data["node"]["atomic_num"].device
        )
        angles = angles.view(-1, 3)
        all_node_types = graph._data["node"]["atomic_num"][angles].squeeze()
        mask = all_node_types[:, 0] > all_node_types[:, 2]
        try:
            angles[mask] = angles[mask].flip(dims=(1,))
        except:
            import pdb

            pdb.set_trace()

        return angles

    def _get_values(self, graph, create_graph=False, device="cpu"):
        values = {}
        angles = graph.get_data(self.name, "index")
        if len(angles) != 0:
            if create_graph:
                xyz = graph._data["node"]["xyz"]
                graph._data["node"]["xyz"] = torch.tensor(
                    xyz.tolist(),
                    requires_grad=True,
                    dtype=torch.float,
                    device=angles.device,
                )
            xyz = graph._data["node"]["xyz"]
            bonds = topcalc.get_bonds_in_angles(graph, angles, device)
            # bond_types = graph.GetTopology('bond')['type'][bonds].view(-1,2)
            # mask = bond_types[:,0] > bond_types[:,1]
            # angles[mask] = angles[mask].flip(dims=(1,))

            cos_theta, theta, b = topcalc.get_angle_geometry(xyz, angles)
            bonds = topcalc.get_bonds_in_angles(graph, angles, device)
            dtheta, db = topcalc.get_angle_derivatives(xyz, angles)
            values["cos_theta"] = cos_theta.view(-1, 1)
            values["theta"] = theta.view(-1, 1)
            values["b"] = b.view(-1, 2)
            values["bond"] = bonds
            values["dtheta"] = dtheta.view(-1, 3 * 3)
            values["db"] = db.view(-1, 2 * 3 * 3)
        else:
            values["cos_theta"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["theta"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["b"] = torch.tensor([]).view(-1, 2).to(torch.float)
            values["bond"] = torch.tensor([]).view(-1, 2).to(torch.long)
            values["dtheta"] = torch.tensor([]).view(-1, 3 * 3).to(torch.float)
            values["db"] = torch.tensor([]).view(-1, 2 * 3 * 3).to(torch.float)
        return values

    def _get_types(self, graph, type_key):
        if graph.typer is not None and type_key == graph.typer.name:
            custom_types = graph.typer.get_types(graph, self.name)
            return custom_types
        node_types = graph.get_data("node", type_key).view(-1)
        top_types = node_types[graph.get_data(self.name, "index")]
        try:
            node_degrees = graph.get_data("node", "degree").view(-1)[
                graph.get_data(self.name, "index")
            ]
        except:
            import pdb

            pdb.set_trace()
        cos_theta = graph.get_data(self.name, "cos_theta")
        if len(top_types) == 0:
            return torch.tensor([]).view(-1, 1)
        if self.num_node_types[type_key] is None:
            self.num_node_types[type_key] = (
                graph.get_data("node", type_key).max().item() + 1
            )
        top_types = torch.eye(self.num_node_types[type_key], device=top_types.device)[
            top_types
        ].to(torch.long)
        # only do the permutation invariance on the outer nodes; the inner node is unique to each angle type
        top_types = torch.cat([top_types[:, 1], top_types[:, [0, 2]].sum(1)], dim=1)
        # need to separate the atom types of linear PF6- angles and perpendicular PF6- angles, also for 5-coordinated species
        if 6 in node_degrees or 5 in node_degrees:
            if 6 in node_degrees:
                # find all the angles with the potential linear/nonliear separation (only occurs in 6-coordiniated shapes
                hybridization_mask = node_degrees[:, 1] == 6
            else:
                # find all the angles with the potential linear/nonliear separation (only occurs in 5-coordiniated shapes
                hybridization_mask = node_degrees[:, 1] == 5
            # find all the angles that are close to linear
            angle_mask = torch.isclose(
                cos_theta.abs(), torch.ones_like(cos_theta), rtol=0.3
            ).view(-1)
            # create a mask that contains indexes when both above conditions are true
            mask = (
                torch.logical_and(hybridization_mask, angle_mask)
                .view(-1, 1)
                .expand(top_types.shape)
            )
            # add "1" to the one-hot-encodings generated in top_types to ENSURE that the new types (for linear angles
            # in 6-coordinated molecules) are distinct from any possible outcome of one-hot-encoding from using "torch.eye" above
            top_types = torch.where(mask, top_types + 1, top_types)
        if self.unique_values[type_key] is None:
            self.unique_values[type_key], top_types = top_types.unique(
                dim=0, return_inverse=True
            )
        else:
            # this index_of is what actually assigns the topology type to a particular topology,
            # because it will give the index of the one-hot encoding from self.unique_values[type_key]
            top_types = self.index_of(
                vectors=top_types, source=self.unique_values[type_key]
            )
        types = top_types.view(-1, 1)
        return types


class DihedralTopology(Topology):
    def __init__(self):
        super(DihedralTopology, self).__init__()
        self.name = "dihedral"
        self.unique_pair_values = {"base_type": None, "type": None}
        self.num_pair_types = {"base_type": None, "type": None}

    def _get_indices(self, graph):
        dihedrals = copy.deepcopy(graph.neighbors)
        neigh = copy.deepcopy(graph.neighbors)
        for i in range(len(neigh)):
            for counter, j in enumerate(neigh[i]):
                k = set(neigh[i]) - set([j])
                l = set(neigh[j]) - set([i])
                pairs = list(
                    filter(lambda pair: pair[0] < pair[1], itertools.product(k, l))
                )
                dihedrals[i][counter] = [
                    [pair[0]] + [i] + [j] + [pair[1]] for pair in pairs
                ]
        dihedrals = list(itertools.chain(*list(itertools.chain(*dihedrals))))
        dihedrals = torch.LongTensor(dihedrals)
        dihedrals = dihedrals.view(-1, 4)
        return dihedrals

    def _get_values(self, graph, create_graph=False, device="cpu"):
        # import time
        # start = time.time()
        values = {}
        dihedrals = graph.get_data(self.name, "index").to(device)
        # print('\t\t\ttime to get diherdals:', time.time()-start)
        # start = time.time()
        if len(dihedrals) != 0:
            if create_graph:
                xyz = graph._data["node"]["xyz"]
                graph._data["node"]["xyz"] = torch.tensor(
                    xyz.tolist(),
                    requires_grad=True,
                    dtype=torch.float,
                    device=dihedrals.device,
                )
            xyz = graph._data["node"]["xyz"]
            displacements = xyz.unsqueeze(1).expand(len(xyz), len(xyz), 3)
            displacements = displacements.transpose(0, 1) - displacements
            D2 = displacements.pow(2).sum(2)
            # print('\t\t\ttime to get displacements:', time.time()-start)
            # start = time.time()
            # this works for opls permutation invariance, but not for class2 forcefields cross-term invariance
            #             pairs = dihedrals[:,[0,3]]
            #             inv_D = D2[pairs[:,0], pairs[:,1]].pow(-0.5)
            #             cos_phi, phi, theta, b = topcalc.get_dihedral_geometry(xyz, dihedrals)
            #             angles = topcalc.get_angles_in_dihedrals(graph, dihedrals)
            #             dphi, dtheta, db = topcalc.get_dihedral_derivatives(xyz, dihedrals)

            # change the order of the dihedral nodes so the lowest integer bond type are always first
            # this instance of bonds will be overwritten by the new indexing of dihedral nodes, but need an original to get same permuation of bond types for all dihedrals
            bonds = topcalc.get_bonds_in_dihedrals(graph, dihedrals, device)
            bond_types = graph.GetTopology("bond")["type"][bonds].view(-1, 3).to(device)
            mask = bond_types[:, 0] > bond_types[:, 2]
            dihedrals = dihedrals.to(device)
            dihedrals[mask] = dihedrals[mask].flip(dims=(1,))
            # print('\t\t\ttime to get reorder:', time.time()-start)
            # start = time.time()
            pairs = dihedrals[:, [0, 3]]
            inv_D = D2[pairs[:, 0], pairs[:, 1]].pow(-0.5)
            # print('\t\t\ttime to get invD:', time.time()-start)
            # start = time.time()
            cos_phi, phi, theta, b = topcalc.get_dihedral_geometry(xyz, dihedrals)
            # print('\t\t\ttime to get phis:', time.time()-start)
            # start = time.time()
            bonds = topcalc.get_bonds_in_dihedrals(graph, dihedrals, device)
            # print('\t\t\ttime to get bonds:', time.time()-start)
            # start = time.time()
            angles = topcalc.get_angles_in_dihedrals(graph, dihedrals, device)
            # print('\t\t\ttime to get angles:', time.time()-start)
            # start = time.time()
            # dphi, dtheta, db = topcalc.get_dihedral_derivatives(xyz, dihedrals)
            # dphi = topcalc.get_dihedral_derivatives(xyz, dihedrals)
            # import ipdb
            # ipdb.set_trace()
            # print('\t\t\ttime to get calculate dihedral derivatives:', time.time()-start)
            # start = time.time()

            # unshift dihedrals so that it repeats indices for different molecules and can be placed into graph._data['dihedral']['index']
            index_shift = graph._data["num"]["node"].view(-1)
            index_shift = (index_shift.cumsum(0) - index_shift).tolist()
            num = graph._data["num"][self.name].view(-1).tolist()
            dihedrals = list(dihedrals.split(num))
            for cur in range(len(num)):
                dihedrals[cur] = dihedrals[cur] - index_shift[cur]
            dihedrals = torch.cat(dihedrals)
            graph._data[self.name]["index"] = dihedrals
            # print('\t\t\t\ttime to shift indices:', time.time()-start)
            # start = time.time()

            values["phi"] = phi.view(-1, 1)
            values["inv_D"] = inv_D.view(-1, 1)
            values["cos_phi"] = torch.cos(phi).view(-1, 1)
            values["cos_2phi"] = torch.cos(2 * phi).view(-1, 1)
            values["cos_3phi"] = torch.cos(3 * phi).view(-1, 1)
            values["cos_4phi"] = torch.cos(4 * phi).view(-1, 1)
            values["sin_phi"] = torch.sin(phi).view(-1, 1)
            values["sin_2phi"] = torch.sin(2 * phi).view(-1, 1)
            values["sin_3phi"] = torch.sin(3 * phi).view(-1, 1)
            values["sin_4phi"] = torch.sin(4 * phi).view(-1, 1)
            values["b"] = b.view(-1, 3)
            values["theta"] = theta.view(-1, 2)
            values["bond"] = bonds
            values["angle"] = angles
            # values['dphi'] = dphi.view(-1,4*3)
            # values['dtheta'] = dtheta.view(-1,2*4*3)
            # values['db'] = db.view(-1,3*4*3)
            # print('\t\t\ttime to save values [s]', time.time()-start)
        else:
            values["inv_D"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["cos_phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["cos_2phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["cos_3phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["cos_4phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["sin_phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["sin_2phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["sin_3phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["sin_4phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["phi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["b"] = torch.tensor([]).view(-1, 3).to(torch.float)
            values["theta"] = torch.tensor([]).view(-1, 2).to(torch.float)
            values["bond"] = torch.tensor([]).view(-1, 3).to(torch.long)
            values["angle"] = torch.tensor([]).view(-1, 2).to(torch.long)
            values["dphi"] = torch.tensor([]).view(-1, 4 * 3).to(torch.float)
            values["dtheta"] = torch.tensor([]).view(-1, 2 * 4 * 3).to(torch.float)
            values["db"] = torch.tensor([]).view(-1, 3 * 4 * 3).to(torch.float)
        return values

    def _get_types(self, graph, type_key):
        if graph.typer is not None and type_key == graph.typer.name:
            custom_types = graph.typer.get_types(graph, self.name)
            return custom_types
        node_types = graph.get_data("node", type_key).view(-1)
        top_types = node_types[graph.get_data(self.name, "index")]
        if len(top_types) == 0:
            return torch.tensor([]).view(-1, 1)
        if self.num_node_types[type_key] is None:
            self.num_node_types[type_key] = (
                graph.get_data("node", type_key).max().item() + 1
            )
        top_types = torch.eye(self.num_node_types[type_key], device=top_types.device)[
            top_types
        ].to(torch.long)
        # top_types after the stack is of dimension (# topologies, 2, 2*num_node_types)
        top_types = torch.stack(
            [
                torch.cat([top_types[:, 1], top_types[:, 0]], dim=1),
                torch.cat([top_types[:, 2], top_types[:, 3]], dim=1),
            ],
            dim=1,
        )
        # import ipdb
        # ipdb.set_trace()
        # pair_types is this stack but now the 0-1 and 3-2 bonds are all in sequence instead of being stacked together in the 1st dimension
        # pair_types shape is now (2*(# topologies), 2*num_node_types))
        pair_types = top_types.view(-1, top_types.shape[2])
        if self.unique_pair_values[type_key] is None:
            self.unique_pair_values[type_key], pair_types = pair_types.unique(
                dim=0, return_inverse=True
            )
        else:
            pair_types = self.index_of(
                vectors=pair_types, source=self.unique_pair_values[type_key]
            )
        pair_types = pair_types.view(-1, 2)
        if self.num_pair_types[type_key] is None:
            self.num_pair_types[type_key] = pair_types.max().item() + 1
        top_types = torch.eye(self.num_pair_types[type_key], device=pair_types.device)[
            pair_types
        ].to(torch.long)
        top_types = top_types.sum(1)
        if self.unique_values[type_key] is None:
            self.unique_values[type_key], top_types = top_types.unique(
                dim=0, return_inverse=True
            )
        else:
            top_types = self.index_of(
                vectors=top_types, source=self.unique_values[type_key]
            )
        types = top_types.view(-1, 1)
        return types


class ImproperTopology(Topology):
    def __init__(self):
        super(ImproperTopology, self).__init__()
        self.name = "improper"

    def _get_indices(self, graph):
        _impropers = copy.deepcopy(graph.neighbors)
        impropers = []
        edges = graph.get_data("edge", "index").tolist()
        for i in range(len(_impropers)):
            if len(_impropers[i]) == 3:
                neighbors = _impropers[i]
                pairs = list(itertools.combinations(neighbors, 2))
                is_improper = True
                for pair in pairs:
                    if list(pair) in edges:
                        is_improper = False
                        break
                if is_improper:
                    impropers.append([i] + _impropers[i])
            elif len(_impropers[i]) == 4:
                neighbors = _impropers[i]
                pairs = list(
                    itertools.combinations(neighbors, 3)
                )  # get the 4 different impropers from 4-coordinated atoms
                for pair in pairs:
                    impropers.append([i] + list(pair))
        impropers = torch.LongTensor(impropers)
        impropers = impropers.view(-1, 4)
        impropers = impropers[
            :, [1, 0, 2, 3]
        ]  # switch so that the center atom is the one-th index ("j" index of lammps improper documentation: https://docs.lammps.org/improper_class2.html)
        return impropers

    def _get_values(self, graph, create_graph=False, device="cpu"):
        values = {}
        impropers = graph.get_data(self.name, "index").to(device)
        if len(impropers) != 0:
            if create_graph:
                xyz = graph._data["node"]["xyz"]
                graph._data["node"]["xyz"] = torch.tensor(
                    xyz.tolist(),
                    requires_grad=True,
                    dtype=torch.float,
                    device=impropers.device,
                )
            xyz = graph._data["node"]["xyz"]
            angles = topcalc.get_angles_in_impropers(graph, impropers, device).to(
                device
            )
            angle_types = (
                graph.GetTopology("angle")["type"][angles].squeeze().to(device)
            )
            ref_bonds = graph.get_data("bond", "index").to(device)
            bond_nodes = impropers[:, [1, 0, 1, 2, 1, 3]].view(-1, 3, 2).to(device)
            bonds = topcalc.get_bonds_in_impropers(graph, impropers, device).to(device)
            bond_types = graph.GetTopology("bond")["type"][bonds].squeeze().to(device)
            min_inx = torch.argmin(bond_types, dim=1).to(device)
            # (cur-1).abs() makes it so that if all three bond types are the same, the middle index will always be chosen
            # regardless of if torch.argmax(torch.tensor([1,1,1])) returns 0 or 2.
            max_inx = torch.tensor(
                [
                    (cur - 1).abs() if cur == min_inx[i] else cur
                    for (i, cur) in enumerate(torch.argmax(bond_types, dim=1))
                ],
                device=device,
            )
            mask = (
                torch.ones_like(angle_types)
                .scatter_(1, torch.stack((min_inx, max_inx)).transpose(1, 0), 0.0)
                .to(device)
            )
            mid_inx = (
                torch.ones([angle_types.shape[0], 1], dtype=torch.int64)
                * torch.arange(3)
            )[mask.bool()].to(device)

            impropers_i = bond_nodes[(range(bond_nodes.shape[0]), min_inx, 1)]
            impropers_k = bond_nodes[(range(bond_nodes.shape[0]), mid_inx, 1)]
            impropers_l = bond_nodes[(range(bond_nodes.shape[0]), max_inx, 1)]
            impropers = torch.stack(
                (impropers_i, impropers[:, 1], impropers_k, impropers_l), dim=1
            )

            chi, cos_theta, theta, d = topcalc.get_improper_geometry(xyz, impropers)
            dchi, dtheta = topcalc.get_improper_derivatives(xyz, impropers)
            angles = topcalc.get_angles_in_impropers(graph, impropers, device)

            # unshift impropers so that it repeats indices for different molecules and can be placed into graph._data['improper']['index']
            index_shift = graph._data["num"]["node"].view(-1)
            index_shift = (index_shift.cumsum(0) - index_shift).tolist()
            num = graph._data["num"][self.name].view(-1).tolist()
            impropers = list(impropers.split(num))
            for cur in range(len(num)):
                impropers[cur] = impropers[cur] - index_shift[cur]
            impropers = torch.cat(impropers)

            graph._data[self.name]["index"] = impropers

            values["chi"] = chi.view(-1, 1)
            values["d"] = d.view(-1, 3)
            values["dchi"] = dchi.view(-1, 4 * 3)
            values["dtheta"] = dtheta.view(-1, 3 * 4 * 3)
            values["theta"] = theta.view(-1, 3)
            values["angle"] = angles
        else:
            values["chi"] = torch.tensor([]).view(-1, 1).to(torch.float)
            values["d"] = torch.tensor([]).view(-1, 3).to(torch.float)
            values["dchi"] = torch.tensor([]).view(-1, 4 * 3).to(torch.float)
            values["dtheta"] = torch.tensor([]).view(-1, 4 * 3).to(torch.float)
            values["theta"] = torch.tensor([]).view(-1, 3).to(torch.float)
            values["angle"] = torch.tensor([]).view(-1, 3).to(torch.long)
        return values

    def _get_types(self, graph, type_key):
        if graph.typer is not None and type_key == graph.typer.name:
            custom_types = graph.typer.get_types(graph, self.name)
            return custom_types
        node_types = graph.get_data("node", type_key).view(-1)
        # column 1 of improper_types is the center node of the improper
        improper_types = node_types[graph.get_data(self.name, "index")]
        if len(improper_types) == 0:
            return torch.tensor([]).view(-1, 1)
        num_node_types = graph.get_data("node", type_key).max().item() + 1
        improper_types = torch.eye(num_node_types, device=improper_types.device)[
            improper_types
        ].to(torch.long)
        improper_types = improper_types.sum(1)
        if self.unique_values[type_key] is None:
            self.unique_values[type_key], improper_types = improper_types.unique(
                dim=0, return_inverse=True
            )
        else:
            unique_values, improper_types = improper_types.unique(
                dim=0, return_inverse=True
            )
        types = improper_types.view(-1, 1)
        return types


class PairTopology(Topology):
    def __init__(self, use_1_4_pairs=False, forcefield_class2=False, cutoff=None):
        super(PairTopology, self).__init__()
        self.name = "pair"
        self.use_1_4_pairs = use_1_4_pairs
        self.forcefield_class2 = forcefield_class2
        self.cutoff = cutoff

    def _get_indices(self, graph):
        pairs = torch.eye(graph.N, graph.N).to(torch.long)
        topologies = [graph.topologies[name] for name in ["bond", "angle"]]
        if self.use_1_4_pairs is False:
            topologies.append(graph.topologies["dihedral"])
        for topology in topologies:
            for interaction_list in graph.get_data(topology.name, "index"):
                for pair in itertools.combinations(interaction_list, 2):
                    pairs[pair[0], pair[1]] = 1
                    pairs[pair[1], pair[0]] = 1
        num_nodes = graph._data["num"]["node"].view(-1).tolist()
        intergraph_A = block_diag(*[np.ones((n, n)) for n in num_nodes])
        intergraph_A = torch.tensor(intergraph_A.tolist()).to(torch.long)
        pairs = pairs + (1 - intergraph_A)
        pairs = torch.nonzero((pairs == 0), as_tuple=False)
        pairs = pairs.sort(dim=1)[0].unique(dim=0).tolist()
        pairs = torch.LongTensor(pairs)
        pairs = pairs.view(-1, 2)
        if self.cutoff is not None:
            xyz = graph._data["node"]["xyz"]
            D = (xyz[pairs[:, 1]] - xyz[pairs[:, 0]]).pow(2).sum(-1).pow(0.5)
            pairs = pairs[D < self.cutoff]
        return pairs

    def _get_values(self, graph, create_graph=False, device="cpu"):
        values = {}
        pairs = graph.get_data(self.name, "index").to(device)
        if len(pairs) != 0:
            if create_graph:
                xyz = graph._data["node"]["xyz"]
                graph._data["node"]["xyz"] = torch.tensor(
                    xyz.tolist(),
                    requires_grad=True,
                    dtype=torch.float,
                    device=pairs.device,
                )
            xyz = graph._data["node"]["xyz"]
            XYZ = xyz[pairs]
            inv_D = (XYZ[:, 0] - XYZ[:, 1]).pow(2).sum(1).pow(-0.5)
            values["inv_D"] = inv_D.view(-1, 1)
            values["inv_D_2"] = inv_D.pow(2).view(-1, 1)  ###############
            values["D"] = inv_D.view(-1, 1).pow(-1)  ###############
            #################################################################
            r = graph._data["node"]["xyz"][pairs]
            D = inv_D.pow(-1)
            R = D.view(-1, 1).unsqueeze(1)
            try:
                I = torch.eye(2).to(pairs.device)
                C = (
                    (r[:, [1]] - r[:, [0]])
                    * (I[1] - I[0]).view(1, 2, 1).to(pairs.device)
                    / R
                )
            except:
                import pdb

                pdb.set_trace()
            values["C"] = C.view(-1, 2 * 3)
            if "f_1_4" not in graph._data["pair"].keys():
                f_1_4 = torch.ones_like(inv_D)
                # only do this if there are dihedrals present
                if (
                    self.use_1_4_pairs
                    and graph.get_data("dihedral", "index")[:, [0, 3]].shape[0]
                ):
                    pairs_1_4 = graph.get_data("dihedral", "index")[:, [0, 3]].to(
                        device
                    )
                    a = torch.cat([pairs, pairs_1_4.unique(dim=0)])
                    v, c = a.unique(dim=0, return_counts=True)
                    b = index_of(
                        v[c == 2],
                        pairs,
                        max_index=max([pairs.max().item(), pairs_1_4.max().item()]) + 1,
                    )
                    if self.forcefield_class2:
                        f_1_4[b] = f_1_4[b] * 1.0
                    else:
                        f_1_4[b] = f_1_4[b] * 0.5
                    values["f_1_4"] = f_1_4.view(-1, 1)
                else:
                    # pairs_1_4 = torch.tensor([]).view(0,2).to(torch.long)
                    values["f_1_4"] = f_1_4.view(-1, 1)
        else:
            values["inv_D"] = torch.tensor([]).view(-1, 1)
            values["inv_D_2"] = torch.tensor([]).view(-1, 1)
            values["D"] = torch.tensor([]).view(-1, 1)
            values["C"] = torch.tensor([]).view(-1, 2 * 3).to(torch.float)
            values["f_1_4"] = torch.tensor([]).view(-1, 1)
        return values

    def _get_types(self, graph, type_key):
        if graph.typer is not None:
            node_types = graph._data["node"][graph.typer.name].view(-1)
        else:
            node_types = graph._data["node"][type_key].view(-1)
        pair_types = node_types[graph.get_data(self.name, "index")]
        if len(pair_types) == 0:
            return torch.tensor([]).view(-1, 2)
        types = pair_types.view(-1, 2)
        return types
