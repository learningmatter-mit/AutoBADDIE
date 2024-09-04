import torch, copy, random
import materialbuilder
from materialbuilder.base import GraphBase
from materialbuilder.errors import TemplateError
from collections import OrderedDict
import numpy as np


class Graph(GraphBase):
    def __init__(self, parent=None, A=None):
        super(Graph, self).__init__()
        if parent is not None:
            self.parent = parent
        elif A is not None:
            self._populate_from_adjmat(A)
        self.transformations = {}
        self.typer = None
        self.node_type_label = None
        self.base_node_type_label = None

    def AddNodeLabel(self, label, values, dtype=torch.float):
        self._data["node"][label] = self._clean_tensor(values, dtype)

    def AddEdgeLabel(self, label, values, dtype=torch.float):
        self._data["edge"][label] = self._clean_tensor(values, dtype)

    def AddProperty(self, label, values, dtype=torch.float):
        self._data["prop"][label] = self._clean_tensor(values, dtype)

    def AddNum(self, label, values, dtype=torch.float):
        self._data["num"][label] = self._clean_tensor(values, dtype)

    def AddTransformation(self, transformation):
        name = type(transformation).__name__
        if name not in self.transformations.keys():
            self.transformations[name] = []
        self.transformations[name].append(transformation)
        transformation.ApplyTo(self)

    def DefineBaseNodeTypes(self, label):
        self.base_node_type_label = label
        base_types = torch.nonzero(self._data["node"][label], as_tuple=False)[:, [1]]
        self.AddNodeLabel("base_type", base_types, torch.long)
        for topology in self.topologies.values():
            topology.apply_to(self, type_keys=["base_type"])

    def DefineNodeTypes(self, label):
        self.node_type_label = label
        types = torch.nonzero(self._data["node"][label], as_tuple=False)[:, [1]]
        self.AddNodeLabel("type", types, torch.long)
        for topology in self.topologies.values():
            topology.apply_to(self, type_keys=["type"])

    def AddTyper(self, typer):
        self.typer = typer
        self.AddNodeLabel(typer.name, typer(self), torch.long)
        for topology in self.topologies.values():
            topology.apply_to(self, type_keys=[typer.name])

    def AddTopology(self, topology, device):
        topology.apply_to(self, device)
        self.topologies.update({topology.name: topology})

    def GetTopology(self, top, create_graph=False):
        return self.topologies[top](self, create_graph)


class GraphBatch(Graph):
    def __init__(self):
        super(GraphBatch, self).__init__(A=None)
        self.graphs = []

    @property
    def batch_size(self):
        return len(self.graphs)

    @property
    def template(self):
        return self.graphs[0].template

    def Add(self, graph):
        self._check_template(graph)
        self.graphs.append(graph)

    def _finalize(self):
        self._initialize_data(template=self.graphs[0].template)
        for b in range(self.batch_size):
            self._add_data_from_graph(self.graphs[b])
        self._transform_data(join=True)

    def Close(self):
        self._finalize()


class Ensemble(GraphBatch):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.parent_graph = None
        self.child_graphs = []
        self.expanded = False

    def Add(self, graph):
        self._check_template(graph)
        if self.parent_graph is None:
            self.parent_graph = graph
            self.graphs.append(self.parent_graph)
        else:
            self._check_template(graph)
            self.child_graphs.append(graph)

    def Unzip(self):
        self._transform_data(split=True)
        parent_data = {}
        for level in self._data.keys():
            parent_data[level] = {}
            for label in self._data[level].keys():
                parent_data[level][label] = self._data[level][label][0]
        for child_graph in self.child_graphs:
            child_data = {}
            child_data["node"]["xyz"] = child_graph._data["node"]["xyz"]
            child_data["prop"]["E"] = child_graph._data["prop"]["E"]
            for level in self._data.keys():
                for label in self._data[level].keys():
                    self._data[level][label].insert(1, child_data[level][label])
        self._transform_data(join=True)
        for topology in self.topologies.values():
            topology.apply_values_to(self)


class GraphDataset:
    def __init__(self):
        self.container_batch = GraphBatch()
        self.batches = []
        self.ensembles = []
        self.transformations = OrderedDict({})
        self.batch_size = None
        self.order = None
        self.details = {}

    @property
    def dataset_size(self):
        dataset_size = len(self.container_batch.graphs)
        for ensemble in self.ensembles:
            if ensemble.expanded:
                dataset_size += len(ensemble.child_graphs)
        return dataset_size

    @property
    def num_batches(self):
        return len(self.batches)

    @property
    def template(self):
        return self.container_batch.template

    @property
    def _data(self):
        return self.container_batch._data

    @property
    def data(self):
        return self.container_batch.data

    def AddGraph(self, graph):
        graph.AddProperty("ensemble", -1, torch.long)
        self.container_batch.Add(graph)

    def Add(self, graph):
        self.AddGraph(graph)

    def AddEnsemble(self, ensemble):
        ensemble.parent_graph.AddProperty("ensemble", len(self.ensembles), torch.long)
        self.container_batch.Add(ensemble.parent_graph)
        self.ensembles.append(ensemble)

    def Close(self):
        self.container_batch.Close()

    def AddTopology(self, topology, device):
        self.container_batch.AddTopology(topology, device)

    def AddTransformation(self, transformation, device):
        """
        self is the current dataset (a subset of all possible molecules)
        while tranformation comes from the template dataset (containing all species)
        transformation = for transformation in template_dataset.transformations[_transformation]
            where _transformation = 'OneHotEncoding' or 'GraphDistinctNodes'
        """
        name = type(transformation).__name__
        if name not in self.transformations.keys():
            self.transformations[name] = []
        self.transformations[name].append(transformation)
        transformation.ApplyTo(self.container_batch, device)

    def DefineBaseNodeTypes(self, label):
        self.container_batch.DefineBaseNodeTypes(label)

    def DefineNodeTypes(self, label):
        self.container_batch.DefineNodeTypes(label)

    def AddTyper(self, typer):
        self.container_batch.AddTyper(typer)

    def UnzipEnsembles(self, Qs=None):
        ensemble_counters = self._data["prop"]["ensemble"]
        if type(ensemble_counters) is not list:
            ensemble_counters = ensemble_counters.view(-1).tolist()
        self.container_batch._transform_data(split=True)
        for graph_counter in range(self.dataset_size):
            ensemble_counter = ensemble_counters[graph_counter]
            if ensemble_counter != -1:
                ensemble = self.ensembles[ensemble_counter]
                ensemble._finalize()
                parent_data = {}
                for level in (
                    self._data.keys()
                ):  # node, edge, prop, num, bond, angle, dihedral, improper, pair
                    parent_data[level] = {}
                    for label in self._data[level].keys():
                        parent_data[level][label] = self._data[level][label][
                            graph_counter
                        ]

                for graph_counter, child_graph in enumerate(ensemble.child_graphs):
                    child_data = {}
                    for level in (
                        child_graph._data.keys()
                    ):  # node, edge, prop, num, bond, angle, dihedral, improper, pair
                        child_data[level] = {}
                        for label in parent_data[
                            level
                        ].keys():  # this includes transformations (ex. one_hot_graph_distinct_r3)
                            if label not in child_graph._data[level].keys():
                                child_data[level][label] = parent_data[level][label]
                            else:
                                child_data[level][label] = child_graph._data[level][
                                    label
                                ]
                    for level in (
                        self._data.keys()
                    ):  # ['node', 'edge', 'prop', 'num', 'bond', [...], 'pair']
                        for label in self._data[
                            level
                        ].keys():  # copy everything including individual bond distances
                            self._data[level][label].append(child_data[level][label])
                self.ensembles[ensemble_counter].expanded = True
        self.container_batch._transform_data(join=True)

        if Qs != None:
            # set target charge to the average of DFT charges for each AuTopology atom type
            qs = self.container_batch._data["node"]["q"].squeeze(1)
            types = np.array(
                self.container_batch._data["node"]["type"].squeeze(1).tolist()
            )
            unique_types = self.container_batch._data["node"]["type"].unique().tolist()
            q_per_type = {}
            for cur_type in unique_types:
                inx = np.where(types == cur_type)[0]
                q_per_type[cur_type] = qs[inx].mean(0)
            Q_tar = (
                torch.tensor([q_per_type[cur_type] for cur_type in types])
                .unsqueeze(0)
                .transpose(0, 1)
            )
            self._data["node"]["q"] = (
                Q_tar  # replace actual DFT charges with average for each atom type
            )

    def get_dataframe(self, level):
        df = self.container_batch.get_dataframe(level)
        return df

    def to(self, device):
        self.container_batch.to(device)
        for batch in self.batches:
            batch.to(device)
        return self

    def Zip(self):
        self.container_batch.Zip()
        for batch in self.batches:
            batch.Zip()

    def Unzip(self):
        self.container_batch.Unzip()
        for batch in self.batches:
            batch.Unzip()

    def CreateBatches(self, batch_size, shuffle=False, device="cpu"):
        if self.order is None:
            self.order = torch.arange(self.dataset_size).tolist()
        self.batch_size = batch_size
        indices = list(range(self.dataset_size))
        bins = range(0, self.dataset_size, self.batch_size)
        idx = [indices[i : i + self.batch_size] for i in bins]
        if shuffle:
            random.shuffle(self.order)
        self.container_batch._reorder(order=self.order)
        self.batches = []
        for B, indices in enumerate(idx):
            batch = GraphBatch()
            batch._data = self.container_batch._get_batch_data(indices)
            batch.topologies = self.container_batch.topologies
            for topology in batch.topologies.values():
                batch.AddTopology(topology, device)

            batch.typer = self.container_batch.typer
            for top in batch.topologies.keys():
                num = batch._data["num"][top].view(-1).tolist()
                graph_index = np.repeat(np.arange(len(num)), num, axis=0).tolist()
                graph_index = torch.tensor(graph_index).view(-1, 1)
                batch._data[top]["graph_index"] = graph_index
            self.batches.append(batch)

    def Shuffle(self, device):
        self.CreateBatches(self.batch_size, shuffle=True, device=device)
