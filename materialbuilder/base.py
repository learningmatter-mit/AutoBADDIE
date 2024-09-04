import torch, itertools
import materialbuilder
from materialbuilder import units
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GraphBase:
    def __init__(self):
        self._data = {"node": {}, "edge": {}, "prop": {}, "num": {}}
        self.topologies = {}

    @property
    def N(self):
        return self._data["num"]["node"].sum().item()

    @property
    def num_edges(self):
        return self._data["num"]["edge"].sum().item()

    @property
    def d(self):
        return self._data["node"]["degree"]

    @property
    def neighbor_list(self):
        return self.get_data("edge", "index")

    @property
    def neighbors(self):
        neighbors = list(torch.split(self.neighbor_list, self.d.view(-1).tolist()))
        neighbors = [a[:, 1].tolist() for a in neighbors]
        return neighbors

    @property
    def template(self):
        template = {}
        for level in self._data.keys():
            template[level] = {}
            for label in self._data[level].keys():
                shape = self._data[level][label].shape
                template[level][label] = shape[1:]
        return template

    def to(self, device):
        for level in self._data.keys():
            for label in self._data[level].keys():
                x = self._data[level][label]
                self._data[level][label] = x.to(device)
        return self

    def get_data(self, level, _label=None):
        data = {}
        labels = self._data[level].keys() if (_label is None) else [_label]
        for label in labels:
            if label == "index":
                indices = self._data[level][label].clone()
                index_shift = self._data["num"]["node"].view(-1)
                index_shift = (index_shift.cumsum(0) - index_shift).tolist()
                num = self._data["num"][level].view(-1).tolist()
                indices = list(indices.split(num))
                for b in range(len(num)):
                    indices[b] += index_shift[b]
                indices = torch.cat(indices)
                data[label] = indices
            else:
                data[label] = self._data[level][label]
        if _label is None:
            return data
        else:
            return data[_label]

    def get_dataframe(self, level):
        def fix_precision(x):
            if type(x) is not list:
                return x
            for counter, y in enumerate(x):
                x[counter] = round(y, 3)
            return x

        data = self.get_data(level, _label=None)
        if "index" in data.keys():
            if len(data["index"]) == 0:
                return pd.DataFrame(columns=data.keys())
        for key in data.keys():
            if data[key].shape[1] == 1:
                data[key] = data[key].view(-1).tolist()
            else:
                data[key] = data[key].tolist()
        df = pd.DataFrame(data)
        keys = list(df.columns)
        for key in keys:
            try:
                df[key] = df[key].apply(lambda x: fix_precision(x))
            except:
                pass
        if "base_type" in keys:
            keys.pop(keys.index("base_type"))
            keys.insert(1, "base_type")
        if "type" in keys:
            keys.pop(keys.index("type"))
            keys.insert(2, "type")
        if "compass_type" in keys:
            keys.pop(keys.index("compass_type"))
            keys.insert(3, "compass_type")
        if "xyz" in keys:
            keys.pop(keys.index("xyz"))
            keys.append("xyz")
        df = df[keys]
        return df

    def _populate_from_adjmat(self, A):
        A = self._clean_tensor(A, dtype=torch.long)
        self.AddNum("node", A.shape[0], torch.long)
        self.AddNodeLabel("index", torch.arange(self.N), torch.long)
        self.AddNodeLabel("degree", A.sum(1), torch.long)
        self.AddNum("edge", A.sum().item(), torch.long)
        self.AddEdgeLabel("index", torch.nonzero(A, as_tuple=False), torch.long)

    def _clean_tensor(self, T, dtype):
        if type(T) != torch.Tensor:
            T = torch.tensor(np.array(T))
        T = T.to(dtype)
        if T.dim() == 0:
            T = T.view(1)
        try:
            T = T.view(len(T), -1)
        except:
            T = T.view(0, T.shape[1])
        return T

    def _check_template(self, graph):
        if len(self.graphs) > 0:
            if graph.template != self.graphs[0].template:
                import ipdb

                ipdb.set_trace()
                raise TemplateError

    def _initialize_data(self, template):
        for level in template.keys():
            self._data[level] = {label: [] for label in template[level].keys()}

    def _add_data_from_graph(self, graph):
        for level in self._data.keys():
            for label in self._data[level].keys():
                self._data[level][label].append(graph._data[level][label])

    def _transform_data(self, split=False, order=None, join=False):
        for level in self._data.keys():
            if split:
                if level in ["num", "prop"]:
                    num = [1 for b in self._data["num"]["node"]]
                else:
                    num = self._data["num"][level]
                    if type(num) is list:
                        num = torch.cat(num).view(-1).tolist()
                    else:
                        num = num.view(-1).tolist()
            for label in self._data[level].keys():
                x = self._data[level][label]
                if split:
                    x = list(x.split(num))
                if order is not None:
                    x = [x[i] for i in order]
                if join:
                    x = torch.cat(x)
                self._data[level][label] = x

    def _reorder(self, order):
        if order is None:
            return
        self._transform_data(split=True)
        self._transform_data(order=order)
        self._transform_data(join=True)

    def _get_batch_data(self, indices):
        self._transform_data(split=True)
        batch_data = {}
        for level in (
            self._data.keys()
        ):  # node, edge, prop, num, bond, angle, dihedral, improper, pair
            batch_data[level] = {}
            for label in self._data[
                level
            ].keys():  # index, degree, atomic_num, mass, b, etc
                values = self._data[level][label][indices[0] : indices[-1] + 1]
                values = torch.cat(values)
                batch_data[level][label] = values
        self._transform_data(join=True)
        return batch_data


class TopologyNetBase(torch.nn.Module):
    def __init__(self):
        super(TopologyNetBase, self).__init__()

    def plot(self, graph):
        types = graph.get_data(self.name, "type")
        X = graph.get_data(self.name, self.coord)
        E = graph.get_data(self.name, "E")
        df = pd.DataFrame()
        if self.name == "pair":
            df["type"] = (
                types.view(-1, 2).unique(dim=0, return_inverse=True)[1].tolist()
            )
        else:
            df["type"] = types.view(-1).tolist()
        df[self.coord] = X.view(-1).tolist()
        df["E"] = E.view(-1).tolist()
        self._plot_energy_profiles(df, self.coord)

    def _constrain_parameter(self, X, X_ref, param):
        return X

    def _plot_energy_profiles(self, df, coord):
        df = df.sort_values(by=coord)
        sns.set(rc={"figure.figsize": (10, 10)})
        color_palette = itertools.cycle(sns.color_palette("bright"))
        for t in list(set(df["type"].tolist())):
            color = next(color_palette)
            X = df[df["type"] == t][coord].tolist()
            E = df[df["type"] == t]["E"].tolist()
            plt.scatter(X, E, color=color, s=10, label=t, alpha=0.5)
            plt.legend()

    def get_dataframe(self, graph, use_reference=False):
        types = graph.get_data(self.name, "type")
        X = graph.get_data(self.name, self.coord)
        df = pd.DataFrame()
        df["type"] = types.view(-1).tolist()
        df[self.coord] = X.view(-1).tolist()
        for param in self.predictors.keys():
            source_param = param
            if use_reference:
                source_param += "_ref"
            values = graph.get_data(self.name, source_param)
            df[param] = values.view(-1).tolist()
        return df

    def plot_observed_distribution(self, graph):
        df = self.get_dataframe(graph, use_reference=False)
        ax = sns.violinplot(data=df, x="type", y=self.coord, cut=0, inner=None, bw=0.2)
        plt.setp(ax.collections, alpha=0.5)
