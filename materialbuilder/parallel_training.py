import materialbuilder
from materialbuilder.dbsettings import Geom, Group, Species, MolSet, Calc, Jacobian
from materialbuilder import graphbuilder, matbuilder, topologies
import torch
from torch.utils.data import Dataset
from munch import Munch


def get_data_dict(geoms_dataframe):
    data_dict = {}
    for species_id, row in geoms_dataframe.iterrows():
        data_dict[row.smiles] = []
        species = Species.objects.get(id=species_id)
        atomicnums = species.connectivitymatrix.atomicnums
        adjmat = species.connectivitymatrix.adjmat
        geoms = Geom.objects.filter(id__in=row.geom_ids).all()
        data_dict[row.smiles] = {
            "species_id": species_id,
            "atomicnums": atomicnums,
            "adjmat": adjmat,
            "geom_data": list(geoms.values("xyz", "calcs__props__energy")),
        }
    return data_dict


class WilsDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.smiles_of, self.inner_index_of = [], []
        for _smiles in self.data_dict.keys():
            for _inner_index in range(len(self.data_dict[_smiles]["geom_data"])):
                self.smiles_of.append(_smiles)
                self.inner_index_of.append(_inner_index)

    def __len__(self):
        dataset_size = 0
        for smiles in self.data_dict.keys():
            dataset_size += len(self.data_dict[smiles]["geom_data"])
        return dataset_size

    def __getitem__(self, index):
        sample = self.data_dict[self.smiles_of[index]]["geom_data"][
            self.inner_index_of[index]
        ]
        sample["species_id"] = self.data_dict[self.smiles_of[index]]["species_id"]
        sample["adjmat"] = self.data_dict[self.smiles_of[index]]["adjmat"]
        sample["atomicnums"] = self.data_dict[self.smiles_of[index]]["atomicnums"]
        return sample


def custom_collate_fn(in_batch):
    dataset = graphbuilder.GraphDataset()
    for b in range(len(in_batch)):
        geometry = matbuilder.Geometry(
            A=in_batch[b]["adjmat"],
            atomic_nums=in_batch[b]["atomicnums"],
            xyz=torch.tensor(in_batch[b]["xyz"])[:, -3:],
        )
        geometry.AddProperty("E", in_batch[b]["calcs__props__energy"], torch.float)
        geometry.AddProperty("species_id", in_batch[b]["species_id"], torch.long)
        dataset.AddGraph(geometry)
    dataset.Close()
    dataset.AddTopology(topologies.BondTopology())
    dataset.AddTopology(topologies.AngleTopology())
    dataset.CreateBatches(batch_size=dataset.dataset_size, shuffle=False)
    batch = dataset.batches[0]
    return batch


chemprop_args = Munch()
chemprop_args.dataset_type = "regression"
chemprop_args.num_tasks = 1
chemprop_args.atom_messages = False
chemprop_args.hidden_size = 300
chemprop_args.bias = False
chemprop_args.depth = 3
chemprop_args.dropout = 0.0
chemprop_args.undirected = False
chemprop_args.features_only = False
chemprop_args.use_input_features = False
chemprop_args.activation = "ReLU"
chemprop_args.features_size = 200
chemprop_args.ffn_num_layers = 2
chemprop_args.ffn_hidden_size = 300
chemprop_args.data_path = "data.csv"
chemprop_args.smiles_column = "smiles"
chemprop_args.target_columns = ["species_id", "property"]
chemprop_args.features_path = None
chemprop_args.features_generator = ["rdkit_2d_normalized"]
chemprop_args.features_scaling = False
chemprop_args.features_size = None
chemprop_args.max_data_size = None
chemprop_args.num_workers = 8
chemprop_args.cache = False
chemprop_args.class_balance = False
chemprop_args.seed = 999
