import torch, random, copy, os
import numpy as np
import materialbuilder
from materialbuilder import utils, errors
try:
    from materialbuilder.xyz2mol import xyz2mol
except:
    pass

class CompassTyper(torch.nn.Module):
    def __init__(self, reference_geoms=None):
        super(CompassTyper, self).__init__()
        self.reference_geoms = reference_geoms
        self.patterns, self.variables, self.param_dicts = utils.read_ff_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'FFlibrary/compass.ff'))
        self.params = {}
        self.name = 'compass_type'
        for term in self.param_dicts.keys():                
            self.params[term] = {
                var: torch.tensor(
                    self.param_dicts[term][var].values.tolist()
                ) for var in self.variables[term]
            }
        
    def _types_from_graph(self, graph):
        atomic_nums = graph.get_data('node', 'atomic_num').view(-1)
        xyz = graph.get_data('node', 'xyz')
        rdkit_mol = xyz2mol.xyz2mol(
            atomic_nums.tolist(), 0, xyz.tolist(), True, True, False)
        conformer = list(rdkit_mol.GetConformers())[0]
        rdkit_xyz = conformer.GetPositions()
        rdkit_xyz = torch.tensor(rdkit_xyz.tolist())
        inv_perm = (rdkit_xyz == xyz.unsqueeze(1)).to(torch.long)
        inv_perm = inv_perm.prod(dim=2).nonzero()[:,1]
        T_list = []
        for atom_type, pattern in self.patterns.items():
            is_H = True if (atom_type[0] == 'h') else False
            substructures = utils.get_substructures(rdkit_mol, pattern, is_H)
            T_list.append(substructures)
        T = torch.stack(T_list).t()
        num_types = len(self.patterns)
        T = torch.cat([T, torch.zeros(len(T),1).to(torch.long)], dim=1)
        T[(T[:,:num_types].sum(1)==0).nonzero(),num_types] += 1
        T = T.nonzero()
        T[(T[:,1]==num_types).nonzero(),1] = -1
        counts = T[:,0].unique(return_counts=True)[1]
        T = list(T.split(counts.tolist()))
        T = torch.stack([t[-1] for t in T])
        types = T[:,1]
        if len(types) != rdkit_mol.GetNumAtoms():
            raise errors.MissingAtomTypeError
        types = types[inv_perm]
        return(types)
        
    def __call__(self, graph):
        if self.reference_geoms is not None:
            types = []
            for ensemble_counter in graph._data['prop']['ensemble'].view(-1).tolist():
                reference_geom = self.reference_geoms[ensemble_counter]['geometry']
                types.append(self._types_from_graph(reference_geom))
            types = torch.cat(types)
        else:
            if 'graphs' in graph.__dict__.keys():
                types = []
                for _graph in graph.graphs:
                    types.append(self._types_from_graph(_graph))
                types = torch.cat(types)
            else:
                types = self._types_from_graph(graph)
        return(types)
    
    def get_types(self, graph, top):
        node_types = graph.get_data('node', self.name).view(-1)
        top_types = node_types[graph.get_data(top, 'index')]
        if len(top_types) == 0:
            return(torch.tensor([]).view(-1,1).to(torch.long))
        if top == 'improper':
            top_types = torch.stack([
                top_types[:,[0,1,2,3]],
                top_types[:,[0,1,3,2]],
                top_types[:,[2,1,0,3]],
                top_types[:,[2,1,3,0]],
                top_types[:,[3,1,0,2]],
                top_types[:,[3,1,2,0]]
            ], dim=1)
        else:
            top_types = torch.stack([top_types, top_types.flip(1)], dim=1)
        ref_types = self.param_dicts[top].drop(columns=self.variables[top])
        ref_types = torch.tensor(ref_types.values.tolist()).to(torch.long)
        top_types = (top_types.unsqueeze(1) == ref_types.unsqueeze(0).unsqueeze(2))
        top_types = top_types.to(torch.long).prod(3).sum(2)
        num_types = ref_types.shape[0]
        num_indices = ref_types.shape[1]
        top_types = torch.cat([top_types,torch.zeros(
            len(top_types),1).to(torch.long)], dim=1)
        top_types[(top_types[:,:num_types].sum(1)==0).nonzero(),num_types] += 1
        top_types = top_types.nonzero()
        if (top_types[:,0].unique(return_counts=True)[1]>1).any():
            raise Exception('Multiple type matches.')
        top_types[(top_types[:,1]==num_types).nonzero(),1] = -1
        top_types = top_types[:,[1]]
        return(top_types)
    
    def get_params(self, top, param, types, graph):
        if top == 'pair':
            if param == 'delta':
                _delta = self.params['bond']['delta']
                bond_types = graph.get_data('bond', self.name)
                delta = torch.zeros_like(_delta[bond_types])
                mask = bond_types>=0
                delta[mask] += _delta[bond_types][mask]
                return(delta, mask)
            elif param == 'charge':
                charge = torch.zeros_like(graph._data['node']['mass'])
                mask = charge>1
                return(charge, mask)
            else:
                types = graph.get_data('node', self.name)
        mask = types>=0
        X = torch.zeros_like(self.params[top][param][types])
        X[mask] = self.params[top][param][types][mask]
        return(X, mask)