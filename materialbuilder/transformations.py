import torch
import materialbuilder
from materialbuilder import topcalc
import numpy as np


class Transformation:
    def __init__(self):
        pass

    def ApplyTo(self, graph):
        pass


class Normalize(Transformation):
    def __init__(self, level, in_label, out_label=None):
        super(Normalize, self).__init__()
        self.level = level
        self.in_label = in_label
        if out_label is not None:
            self.out_label = out_label
        else:
            self.out_label = in_label
        self.unique_values = None
        self.avg = None
        self.std = None

    def ApplyTo(self, graph):
        r = graph._data[self.level][self.in_label].clone()
        self.avg = r.mean(0)
        self.std = r.std(0)
        if torch.isnan(self.std).any():
            self.std = torch.ones_like(self.avg)
        r_normalized = (r - self.avg) / self.std
        if self.level == "node":
            graph.AddNodeLabel(self.out_label, r_normalized)
        elif self.level == "edge":
            graph.AddEdgeLabel(self.out_label, r_normalized)
        elif self.level == "prop":
            graph.AddProperty(self.out_label, r_normalized)


class OneHotEncoding(Transformation):
    def __init__(self, level, in_label, out_label=None):
        super(OneHotEncoding, self).__init__()
        self.level = level
        self.in_label = in_label
        if out_label is not None:
            self.out_label = out_label
        else:
            self.out_label = in_label
        self.unique_values = None

    def ApplyTo(self, graph, device):
        if self.unique_values is None:
            r = graph._data[self.level][self.in_label]
            self.unique_values, one_hot_mapping = r.unique(dim=0, return_inverse=True)
        r = graph._data[self.level][self.in_label].clone()
        r = topcalc.index_of(
            r.to(device), self.unique_values.to(device), self.unique_values.max() + 1
        )
        r = torch.eye(len(self.unique_values)).to(torch.long).to(r.device)[r]
        graph.AddNodeLabel(self.out_label, r, torch.long)


def index_of(input, source):
    source, sorted_index, inverse = np.unique(
        source.tolist(), return_index=True, return_inverse=True, axis=0
    )
    index = torch.cat([torch.tensor(source, device=input.device), input]).unique(
        sorted=True, return_inverse=True, dim=0
    )[1][-len(input) :]
    try:
        index = torch.tensor(sorted_index, device=index.device)[index]
    except:
        print("error in one-hot encoding")
        import IPython

        IPython.embed()
    return index


class GraphDistinctNodes(Transformation):
    def __init__(self, in_label, out_label, radius):
        super(GraphDistinctNodes, self).__init__()
        self.in_label = in_label
        if out_label is not None:
            self.out_label = out_label
        else:
            self.out_label = in_label
        self.radius = radius
        self.unique_values = {}

    def ApplyTo(self, graph, device):
        # r = one hot encoding of atomic number
        # a = indices of atoms in each edge
        # d = number of edges connected to each node (the sum of all of these should equal the total number of edges in the system)
        # print(self.in_label)
        # import ipdb
        # ipdb.set_trace()
        r = graph._data["node"][self.in_label].clone().to(device)
        a = graph.get_data("edge", "index").to(device)
        d = graph._data["node"]["degree"].view(-1).to(device)  # len = num atoms
        for rad in range(self.radius + 1):
            if rad != 0:
                # the message is in the direction (atom 1 -> atom 0) for each edge, so the message is the current atom label of atom1
                # the messages from each incoming atom are then split by receiving atom,
                # so the messages that are going into a particular atom are all grouped together
                messages = list(torch.split(r[a[:, 1]], d.tolist()))
                # the messages incoming to each atom are then added together to enforce permutation invariance
                messages = [messages[n].sum(0) for n in range(len(d))]
                messages = torch.stack(messages)
                # the message is then appended to the current state to remember order of messages
                r = torch.cat([r, messages], dim=1)
            if rad not in self.unique_values.keys():
                self.unique_values[rad], one_hot_mapping = r.unique(
                    dim=0, return_inverse=True
                )
            index = index_of(r, self.unique_values[rad])
            r = (
                torch.eye(len(self.unique_values[rad]))
                .to(torch.long)
                .to(r.device)[index]
            )
            if len(torch.nonzero((r.sum(1) == 0).to(torch.long), as_tuple=False)) != 0:
                raise Exception("Unrecognized graph neighborhood.")
        graph.AddNodeLabel(self.out_label, r, torch.long)


class Concatenate(Transformation):
    def __init__(self, level, in_label1, in_label2, out_label=None):
        super(Concatenate, self).__init__()
        self.level = level
        self.in_label1 = in_label1
        self.in_label2 = in_label2
        if out_label is not None:
            self.out_label = out_label
        else:
            self.out_label = in_label

    def ApplyTo(self, graph):
        r1 = graph._data[self.level][self.in_label1].clone()
        r2 = graph._data[self.level][self.in_label2].clone()
        r = torch.cat([r1, r2], dim=1)
        graph.AddNodeLabel(self.out_label, r, torch.long)
