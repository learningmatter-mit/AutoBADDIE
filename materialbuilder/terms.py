import torch
import numpy as np
import materialbuilder
from materialbuilder.base import TopologyNetBase
from materialbuilder import terms
from collections import OrderedDict

class ParameterPredictor(torch.nn.Module):
    def __init__(self, L_in, L_hidden, L_out):
        super(ParameterPredictor, self).__init__()
        modules = torch.nn.ModuleList()
        Lh = [L_in] + L_hidden
        for Lh_in, Lh_out in [Lh[i:i+2] for i in range(len(Lh)-1)]:
            modules.append(torch.nn.Linear(Lh_in, Lh_out))
            modules.append(torch.nn.Tanh())
        modules.append(torch.nn.Linear(Lh[-1], L_out))
        self.net = torch.nn.Sequential(*modules)
        
    def forward(self, x):
        return(self.net(x))

class TopologyNet(TopologyNetBase):
    def __init__(self, name, coord, details):
        super(TopologyNet, self).__init__()
        self.name = name
        self.coord = coord
        self.crosscoords = []
        self.details = details
        self.predictors = torch.nn.ModuleDict({})
        self.default_reference_params = {}
        self.prefactors = {}
        self.reference_statistics = {}
        self.mode = 'train'
        self.shifts = {}
        
    def train(self):
        self.mode = 'train'
        
    def eval(self):
        self.mode = 'eval'
        
    def E(self, index, graph, **args):
        return(E)
    
    def _get_topology_vectors(self, node_vectors, index):
        return(topology_vectors)
    
    def _get_params(self, graph, _base_node_input, _node_input,
                    types=None, learn=True, freeze_known=False, freeze_eq=False):
        params = {}
        device = graph.get_data('node', 'index').device
        for param in self.predictors.keys():
            base_node_input = _base_node_input
            node_input = _node_input
            if graph.typer is not None:
                X_ref, type_mask = graph.typer.get_params(self.name, param, types, graph)
                X_ref = X_ref.to(device)
                if param in self.reference_statistics.keys():
                    X_ref[~type_mask] = self.reference_statistics[param][
                        graph.get_data(self.name, 'type')][~type_mask].to(device)
                else:
                    X_ref[~type_mask] = self.default_reference_params[param]
                prefactor = self.base_prefactors[param]
                X_ref[~type_mask] += prefactor*self.base_predictors[param](
                    base_node_input)[~type_mask]
            else:
                if param in self.reference_statistics.keys():
                    X_ref = self.reference_statistics[param][
                        graph.get_data(self.name, 'type')].to(device)
                else:
                    X_ref = self.default_reference_params[param]
            #X_ref = 0.0*X_ref
            X = 0.0
            if learn:
                if not (param in self.reference_statistics.keys() and freeze_eq):
                    X = self.predictors[param](node_input)
                    '''
                    if self.name == 'pair':
                        z = graph.get_data('node', 'atomic_num')
                        if param == 'sigma':
                            with torch.no_grad():
                                OPLS_mask = (z==1)|(z==6)
                                _X_ref = self.OPLS_sigma.view(-1).to(z.device)[z]
                                _X_ref[~OPLS_mask] += X_ref
                                X_ref = _X_ref
                        elif param == 'epsilon':
                            with torch.no_grad():
                                OPLS_mask = (z==1)|(z==6)
                                _X_ref = self.OPLS_epsilon.view(-1).to(z.device)[z]
                                _X_ref[~OPLS_mask] += X_ref
                                X_ref = _X_ref
                        _types = graph.get_data('node', 'type')
                        X = self.direct_params[param][_types].view(-1,1)
                        if param in ['sigma', 'epsilon']: ################################
                            X[OPLS_mask] = 0.0 
                    else:
                        X = self.direct_params[param][types].view(-1,1)
                    if graph.typer is not None:
                        if param in ['sigma', 'epsilon']:
                            _freeze_known = freeze_known['LJ']
                        elif param in ['delta', 'charge']:
                            _freeze_known = freeze_known['coulomb']
                        else:
                            _freeze_known = freeze_known
                        if _freeze_known:
                            X[type_mask] = 0.0
                    '''
            X = self._constrain_parameter(X, X_ref, param)
            if self.name == 'pair':
                graph._data['node'][param] = X
                X = X.view(-1)[graph.get_data('pair', 'index')]
            else:
                graph._data[self.name][param] = X
            params[param] = X
        if self.name == 'angle':
            for param in ['kb1b2', 'kb1a', 'kb2a']:
                if param not in self.predictors.keys():
                    graph._data[self.name][param] = torch.zeros_like(
                        graph._data[self.name]['theta0'])
        elif self.name == 'dihedral':
            for param in ['k1b1d', 'k2b1d', 'k3b1d',
                          'k1b2d', 'k2b2d', 'k3b2d',
                          'k1b3d', 'k2b3d', 'k3b3d', 
                          'k1a1d', 'k2a1d', 'k3a1d',
                          'k1a2d', 'k2a2d', 'k3a2d',
                          'ka1a2d', 'kb1b3']:
                if param not in self.predictors.keys():
                    graph._data[self.name][param] = torch.zeros_like(
                        graph._data[self.name]['k1'])
        return(params)
    
    def forward(self, graph, settings):
        if settings.get_forces and self.name in ['bond', 'angle', 'dihedral', 'improper', 'pair']:
            topology_dict = graph.GetTopology(self.name, create_graph=False)
        elif settings.get_forces:
            raise Exception()
            topology_dict = graph.GetTopology(self.name, create_graph=True)
        else:
            topology_dict = graph.GetTopology(self.name, create_graph=False)
        index = topology_dict['index']
        if True: #learn:
            with torch.no_grad(): 
                base_node_vectors = graph._data['node'][
                    self.details.base_input_label].to(torch.float)
                if self.name == 'dihedral':
                    base_topology_vectors = self._get_topology_vectors(
                        base_node_vectors, index, use_base=True)
                else:
                    base_topology_vectors = self._get_topology_vectors(
                        base_node_vectors, index)
                node_vectors = graph._data['node'][self.details.input_label].to(torch.float)
                topology_vectors = self._get_topology_vectors(node_vectors, index)
        else:
            base_topology_vectors = None
            topology_vectors = None
        if graph.typer is not None:
            type_name = graph.typer.name
        else:
            type_name = 'type'
        params = self._get_params(
            graph, base_topology_vectors, topology_vectors, 
            topology_dict[type_name].to(torch.long) if type_name in topology_dict.keys() else None,
            True,
            self.details.freeze_knowns[self.name], settings.freeze_eq)
        args = {**{self.coord: topology_dict[self.coord]}, 
                **{param: params[param] for param in self.predictors.keys()},
                **{xcoord: topology_dict[xcoord] for xcoord in self.crosscoords}}
        E, F = self.E(index, graph, settings, **args)
        subterms = list(E.keys())
        for subterm in subterms:
            graph._data[self.name]['E_'+subterm] = E[subterm]
            num = topology_dict['num'].view(-1).tolist()
            geom_index = graph._data[self.name]['graph_index']
            geom_index = geom_index.view(-1).to(index.device)
            if len(E[subterm]) != 0:
                E[subterm] = torch.zeros(len(num)).to(index.device).scatter_add(
                    0, geom_index, E[subterm].view(-1)).view(-1,1)
        E['total'], F['total'] = 0.0, 0.0
        for subterm in subterms:
            E['total'] += E[subterm]
            if settings.get_forces and self.name in ['bond', 'angle', 'dihedral', 'improper', 'pair']:
                F['total'] += F[subterm]
            elif settings.get_forces and E[subterm].sum().item() != 0:
                raise Exception()
                F['total'] += -1.0*torch.autograd.grad(
                    E[subterm].sum(),
                    graph._data['node']['xyz'],
                    create_graph=True)[0]
        return(E, F)
    
class BondNet(TopologyNet):
    def __init__(self, details):
        super(BondNet, self).__init__('bond', 'b', details)
        self.base_predictors = torch.nn.ModuleDict({})
        self.predictors = torch.nn.ModuleDict({})
        for param in ['r0', 'k2', 'k3', 'k4']:
            self.base_predictors[param] = ParameterPredictor(self.details.base_Fr, self.details.Lh, 1)
            self.predictors[param] = ParameterPredictor(self.details.Fr, self.details.Lh, 1)
        self.default_reference_params = {'r0': 0, 'k2': 0, 'k3': 0, 'k4': 0}
        self.base_prefactors = {'r0': 0, 'k2': 0, 'k3': 0, 'k4': 0}
        self.prefactors = {'r0': 1, 'k2': 10, 'k3': 1, 'k4': 1}
        #self.prefactors = {'r0': 0.01, 'k2': 400, 'k3': 1, 'k4': 1}
        params = ['r0', 'k2', 'k3', 'k4']
        self.direct_params = torch.nn.ParameterDict({
            param: torch.nn.Parameter(torch.zeros(100,1).to(torch.float)) for param in params})
        
    def E(self, index, graph, settings, b, k2, k3, k4, r0):
        E = {}
        E['bond'] = 0.0
        if self.name in settings.terms:
            if self.details.style == 'OPLS':
                E['bond'] += k2*(b-r0).pow(2)
            elif self.details.style == 'compass':
                E['bond'] += k2*(b-r0).pow(2)
                E['bond'] += k3*(b-r0).pow(3)
                E['bond'] += k4*(b-r0).pow(4)
        else:
            E['bond'] += 0*k2*(b-r0).pow(2)
        F = self.forces(index, graph, settings, b, k2, k3, k4, r0)
        return(E, F)
    
    def forces(self, index, graph, settings, b, k2, k3, k4, r0):
        F = {}
        if settings.get_forces:
            F['bond'] = torch.zeros(graph.N, 3).to(index.device)
            if self.name in settings.terms:
                f = 0.0
                if self.details.style == 'OPLS':
                    f += 2*k2*(b-r0).pow(1)
                elif self.details.style == 'compass':
                    f += 2*k2*(b-r0).pow(1)
                    f += 3*k3*(b-r0).pow(2)
                    f += 4*k4*(b-r0).pow(3)
                db = graph.get_data('bond', 'db').view(-1,2,3)
                index = index.view(-1,1).expand(2*len(index),3)
                if len(index) != 0:
                    f = (f.unsqueeze(1)*db).view(-1,3)
                    F['bond'] -= torch.zeros_like(
                        F['bond']).scatter_add(0, index, f)
        return(F)
    
    def _get_topology_vectors(self, node_vectors, index):
        topology_vectors = node_vectors[index].sum(1)
        return(topology_vectors)
    
    def _constrain_parameter(self, X, X_ref, param):
        deltaX = self.prefactors[param]*X
        X = (X_ref+deltaX).pow(2)
        return(X)
    
class AngleNet(TopologyNet):
    def __init__(self, details):
        super(AngleNet, self).__init__('angle', 'theta', details)
        self.crosscoords = ['b']
        self.base_predictors = torch.nn.ModuleDict({})
        self.predictors = torch.nn.ModuleDict({})
        for param in ['theta0', 'k2', 'k3', 'k4']:
            self.base_predictors[param] = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, 1)
            self.predictors[param] = ParameterPredictor(2*self.details.Fr, self.details.Lh, 1)
        self.default_reference_params = {'theta0': 0, 'k2': 0, 'k3': 0, 'k4': 0} #50
        self.base_prefactors = {'theta0': 0, 'k2': 0, 'k3': 0, 'k4': 0}
        self.prefactors = {'theta0': 0.1, 'k2': 0.01, 'k3': 1, 'k4': 1}
        #self.prefactors = {'theta0': 20, 'k2': 100, 'k3': 1, 'k4': 1}
        if 'bond_bond' in self.details.crossterms[self.name]:
            for param in ['kb1b2']:
                self.base_predictors[param] = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, 1)
                self.predictors[param] = ParameterPredictor(2*self.details.Fr, self.details.Lh, 1)
                self.default_reference_params[param] = 0.0
                self.base_prefactors[param] = 1000.0
                self.prefactors[param] = 1000.0
        if 'bond_angle' in self.details.crossterms[self.name]:
            for param in ['kb1a', 'kb2a']:
                self.base_predictors[param] = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, 1)
                self.predictors[param] = ParameterPredictor(2*self.details.Fr, self.details.Lh, 1)
                self.default_reference_params[param] = 0.0
                self.base_prefactors[param] = 1000.0 
                self.prefactors[param] = 1000.0
        params = ['theta0', 'k2', 'k3', 'k4']
        if 'bond_bond' in self.details.crossterms[self.name]:
            params += ['kb1b2']
        if 'bond_angle' in self.details.crossterms[self.name]:
            params += ['kb1a', 'kb2a']
        self.direct_params = torch.nn.ParameterDict({
            param: torch.nn.Parameter(torch.zeros(100,1).to(torch.float)) for param in params})
    
    def E(self, index, graph, settings, theta, k2, k3, k4, theta0, **crossargs):
        theta0 = theta0*(np.pi/180)
        cos_theta = torch.cos(theta).unsqueeze(1)
        E = {}
        E['angle'] = 0.0
        if self.name in settings.terms:
            if self.details.style == 'OPLS':
                E['angle'] += k2*(theta-theta0).pow(2)
            elif self.details.style == 'compass':
                E['angle'] += k2*(theta-theta0).pow(2)
                E['angle'] += k3*(theta-theta0).pow(3)
                E['angle'] += k4*(theta-theta0).pow(4)
        else:
            E['angle'] += 0*k2*(theta-theta0).pow(2)
        if self.details.style == 'compass' and settings.crossterms[self.name]:
            bonds = graph.get_data(self.name, 'bond')
            r0 = graph.get_data('bond', 'r0').view(-1)[bonds]
            b = crossargs['b']
            if 'bond_bond' in settings.crossterms:
                E['bond_bond'] = crossargs['kb1b2']*(b[:,[0]]-r0[:,[0]])*(b[:,[1]]-r0[:,[1]])
            if 'bond_angle' in settings.crossterms:
                E['bond_angle'] = 0.0
                E['bond_angle'] += crossargs['kb1a']*(b[:,[0]]-r0[:,[0]])*(theta-theta0)
                E['bond_angle'] += crossargs['kb2a']*(b[:,[1]]-r0[:,[1]])*(theta-theta0)
        F = self.forces(index, graph, settings, theta, k2, k3, k4, theta0, **crossargs)
        return(E, F)
    
    def forces(self, index, graph, settings, theta, k2, k3, k4, theta0, **crossargs):
        F = {}
        if settings.get_forces:
            #theta0 = theta0*(np.pi/180)
            cos_theta = torch.cos(theta).unsqueeze(1)
            F['angle'] = torch.zeros(graph.N, 3).to(index.device)
            if self.name in settings.terms:
                f = 0.0
                if self.details.style == 'OPLS':
                    f += 2*k2*(theta-theta0).pow(1)
                elif self.details.style == 'compass':
                    f += 2*k2*(theta-theta0).pow(1)
                    f += 3*k3*(theta-theta0).pow(2)
                    f += 4*k4*(theta-theta0).pow(3)
                dtheta = graph.get_data(self.name, 'dtheta').view(-1,3,3)
                index = index.view(-1,1).expand(3*len(index),3)
                if len(index) != 0:
                    f = (f.unsqueeze(1)*dtheta).view(-1,3)
                    F['angle'] -= torch.zeros_like(F['angle']).scatter_add(0, index, f)
                if self.details.style == 'compass' and settings.crossterms[self.name]:
                    F['bond_bond'] = torch.zeros(graph.N, 3).to(index.device)
                    F['bond_angle'] = torch.zeros(graph.N, 3).to(index.device)
                    bonds = graph.get_data(self.name, 'bond')
                    r0 = graph.get_data('bond', 'r0').view(-1)[bonds]
                    b = crossargs['b']
                    db = graph.get_data(self.name, 'db').view(-1,2,3,3)
                    if 'bond_bond' in settings.crossterms[self.name]:
                        f = torch.zeros(len(theta),3,3).to(index.device)
                        f += crossargs['kb1b2'].unsqueeze(1)*((b[:,[0]]-r0[:,[0]]).unsqueeze(1)*db[:,1])
                        f += crossargs['kb1b2'].unsqueeze(1)*((b[:,[1]]-r0[:,[1]]).unsqueeze(1)*db[:,0])
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['bond_bond'] -= torch.zeros_like(F['bond_bond']).scatter_add(0, index, f)
                    if 'bond_angle' in settings.crossterms[self.name]:
                        f = torch.zeros(len(theta),3,3).to(index.device)
                        f += crossargs['kb1a'].unsqueeze(1)*(
                            (b[:,[0]]-r0[:,[0]]).unsqueeze(1)*dtheta + (theta-theta0).unsqueeze(1)*db[:,0])
                        f += crossargs['kb2a'].unsqueeze(1)*(
                            (b[:,[1]]-r0[:,[1]]).unsqueeze(1)*dtheta + (theta-theta0).unsqueeze(1)*db[:,1])
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['bond_angle'] -= torch.zeros_like(F['bond_angle']).scatter_add(0, index, f)
        return(F)
    
    def _get_topology_vectors(self, node_vectors, index):
        topology_vectors = torch.cat([
            node_vectors[index[:,[0,2]]].sum(1),
            node_vectors[index[:,1]]
        ], dim=1)
        return(topology_vectors)
    
    def _constrain_parameter(self, X, X_ref, param):
        deltaX = self.prefactors[param]*X
        X = (X_ref+deltaX).pow(2)
        return(X)
    
class DihedralNet(TopologyNet):
    def __init__(self, details):
        super(DihedralNet, self).__init__('dihedral', 'phi', details)
        self.crosscoords = ['b', 'theta']
        self.base_nonlinear = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, self.details.Lh[-1])
        self.nonlinear = ParameterPredictor(2*self.details.Fr, self.details.Lh, self.details.Lh[-1])
        self.base_predictors = torch.nn.ModuleDict({})
        self.predictors = torch.nn.ModuleDict({})
        if self.details.style == 'compass':
            params = ['k1', 'k2', 'k3']
            self.base_prefactors = {'k1': 0, 'k2': 0, 'k3': 0}
            self.prefactors = {'k1': 1, 'k2': 1, 'k3': 1}
        elif self.details.style == 'OPLS':
            params = ['k1', 'k2', 'k3', 'k4']
            self.base_prefactors = {'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0}
            self.prefactors = {'k1': 50, 'k2': 10, 'k3': 5, 'k4': 5}
        for param in params:
            self.base_predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1) 
            self.predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)        
        self.default_reference_params = {k: 0.0 for k in params}
        if self.details.style == 'compass':
            if 'bond_dihedral' in self.details.crossterms[self.name]:
                for param in ['k1b1d', 'k2b1d', 'k3b1d', 
                              'k1b2d', 'k2b2d', 'k3b2d',
                              'k1b3d', 'k2b3d', 'k3b3d']:
                    self.base_predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.default_reference_params[param] = 0.0
                    self.base_prefactors[param] = 1.0
                    self.prefactors[param] = 1.0
            if 'angle_dihedral' in self.details.crossterms[self.name]:
                for param in ['k1a1d', 'k2a1d', 'k3a1d',
                              'k1a2d', 'k2a2d', 'k3a2d']:
                    self.base_predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.default_reference_params[param] = 0.0
                    self.base_prefactors[param] = 1.0
                    self.prefactors[param] = 1.0
            if 'angle_angle_dihedral' in self.details.crossterms[self.name]:
                for param in ['ka1a2d']:
                    self.base_predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.default_reference_params[param] = 0.0
                    self.base_prefactors[param] = 1.0
                    self.prefactors[param] = 1.0
            if 'bond1_bond3' in self.details.crossterms[self.name]:
                for param in ['kb1b3']:
                    self.base_predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.predictors[param] = ParameterPredictor(self.details.Lh[-1], self.details.Lh, 1)
                    self.default_reference_params[param] = 0.0
                    self.base_prefactors[param] = 1.0
                    self.prefactors[param] = 1.0
        params = ['k1', 'k2', 'k3', 'k4']
        if 'bond_dihedral' in self.details.crossterms[self.name]:
            params += ['k1b1d', 'k2b1d', 'k3b1d', 
                       'k1b2d', 'k2b2d', 'k3b2d',
                       'k1b3d', 'k2b3d', 'k3b3d']
        if 'angle_dihedral' in self.details.crossterms[self.name]:
            params += ['k1a1d', 'k2a1d', 'k3a1d',
                       'k1a2d', 'k2a2d', 'k3a2d']
        if 'angle_angle_dihedral' in self.details.crossterms[self.name]:
            params += ['ka1a2d']
        if 'bond1_bond3' in self.details.crossterms[self.name]:
            params += ['kb1b3']
        self.direct_params = torch.nn.ParameterDict({
            param: torch.nn.Parameter(torch.zeros(100,1).to(torch.float)) for param in params})
                    
    def E(self, index, graph, settings, phi, k1, k2, k3, **crossargs):
        E = {}
        E['dihedral'] = 0.0
        if self.name in settings.terms:
            cos_phi = graph._data['dihedral']['cos_phi']
            cos_2phi = graph._data['dihedral']['cos_2phi']
            cos_3phi = graph._data['dihedral']['cos_3phi']
            cos_4phi = graph._data['dihedral']['cos_4phi']
            if self.details.style == 'OPLS':
                k4 = crossargs['k4']
                E['dihedral'] += 0.5*k1*(1+cos_phi)
                E['dihedral'] += 0.5*k2*(1-cos_2phi)
                E['dihedral'] += 0.5*k3*(1+cos_3phi)
                E['dihedral'] += 0.5*k4*(1-cos_4phi)
            elif self.details.style == 'compass':
                E['dihedral'] += k1*(1-cos_phi)
                E['dihedral'] += k2*(1-cos_2phi)
                E['dihedral'] += k3*(1-cos_3phi)
        else:
            E['dihedral'] += 0*k1
        if self.details.style == 'compass' and settings.crossterms[self.name]:
            bonds = graph.get_data(self.name, 'bond')
            r0 = graph.get_data('bond', 'r0').view(-1)[bonds]
            b = crossargs['b']
            angles = graph.get_data(self.name, 'angle')
            theta0 = graph.get_data('angle', 'theta0').view(-1)[angles]
            theta0 = theta0*(np.pi/180)
            theta = crossargs['theta']
            if 'bond_dihedral' in settings.crossterms[self.name]:
                E['bond_dihedral'] = 0.0
                E['bond_dihedral'] += (b[:,[0]]-r0[:,[0]])*(
                    crossargs['k1b1d']*cos_phi
                  + crossargs['k2b1d']*cos_2phi
                  + crossargs['k3b1d']*cos_3phi)
                E['bond_dihedral'] += (b[:,[1]]-r0[:,[1]])*(
                    crossargs['k1b2d']*cos_phi
                  + crossargs['k2b2d']*cos_2phi
                  + crossargs['k3b2d']*cos_3phi)
                E['bond_dihedral'] += (b[:,[2]]-r0[:,[2]])*(
                    crossargs['k1b3d']*cos_phi
                  + crossargs['k2b3d']*cos_2phi
                  + crossargs['k3b3d']*cos_3phi)
            if 'angle_dihedral' in settings.crossterms[self.name]:
                E['angle_dihedral'] = 0.0
                E['angle_dihedral'] += (theta[:,[0]]-theta0[:,[0]])*(
                    crossargs['k1a1d']*cos_phi
                  + crossargs['k2a1d']*cos_2phi
                  + crossargs['k3a1d']*cos_3phi)
                E['angle_dihedral'] += (theta[:,[1]]-theta0[:,[1]])*(
                    crossargs['k1a2d']*cos_phi
                  + crossargs['k2a2d']*cos_2phi
                  + crossargs['k3a2d']*cos_3phi)
            if 'angle_angle_dihedral' in settings.crossterms[self.name]:
                E['angle_angle_dihedral'] = 0.0
                E['angle_angle_dihedral'] += crossargs['ka1a2d']*(
                    theta[:,[0]]-theta0[:,[0]])*(
                    theta[:,[1]]-theta0[:,[1]])*cos_phi
            if 'bond1_bond3' in settings.crossterms[self.name]:
                E['bond1_bond3'] = 0.0
                E['bond1_bond3'] += crossargs['kb1b3']*(
                    b[:,[0]]-r0[:,[0]])*(
                    b[:,[2]]-r0[:,[2]])
        F = self.forces(index, graph, settings, phi, k1, k2, k3, **crossargs)
        return(E, F)
       
    def forces(self, index, graph, settings, phi, k1, k2, k3, **crossargs):
        F = {}
        if settings.get_forces:
            F['dihedral'] = torch.zeros(graph.N, 3).to(index.device)
            if self.name in settings.terms:
                f = 0.0
                cos_phi = torch.cos(graph._data['dihedral']['phi'])
                phi = torch.acos(cos_phi)
                cos_phi = graph._data['dihedral']['cos_phi']
                cos_2phi = graph._data['dihedral']['cos_2phi']
                cos_3phi = graph._data['dihedral']['cos_3phi']
                cos_4phi = graph._data['dihedral']['cos_4phi']
                sin_phi = graph._data['dihedral']['sin_phi']
                sin_2phi = graph._data['dihedral']['sin_2phi']
                sin_3phi = graph._data['dihedral']['sin_3phi']
                sin_4phi = graph._data['dihedral']['sin_4phi']
                if self.details.style == 'OPLS':
                    k4 = crossargs['k4']
                    f += 0.5*k1*(-1)*1*sin_phi
                    f += 0.5*k2*(+1)*2*sin_2phi
                    f += 0.5*k3*(-1)*3*sin_3phi
                    f += 0.5*k4*(+1)*4*sin_4phi
                elif self.details.style == 'compass':
                    f += k1*1*sin_phi
                    f += k2*2*sin_2phi
                    f += k3*3*sin_3phi
                dphi = graph._data['dihedral']['dphi'].view(-1,4,3)
                index = index.view(-1,1).expand(4*len(index),3)
                if len(index) != 0:
                    f = (f.unsqueeze(1)*dphi).view(-1,3)
                    F['dihedral'] -= torch.zeros_like(F['dihedral']).scatter_add(0, index, f)
                if self.details.style == 'compass' and settings.crossterms[self.name]:
                    F['bond_dihedral'] = torch.zeros(graph.N, 3).to(index.device)
                    F['angle_dihedral'] = torch.zeros(graph.N, 3).to(index.device)
                    F['angle_angle_dihedral'] = torch.zeros(graph.N, 3).to(index.device)
                    F['bond1_bond3'] = torch.zeros(graph.N, 3).to(index.device)
                    bonds = graph.get_data(self.name, 'bond')
                    r0 = graph.get_data('bond', 'r0').view(-1)[bonds]
                    b = crossargs['b']
                    db = graph.get_data(self.name, 'db').view(-1,3,4,3)
                    angles = graph.get_data(self.name, 'angle')
                    theta0 = graph.get_data('angle', 'theta0').view(-1)[angles]
                    theta0 = theta0*(np.pi/180)
                    theta = crossargs['theta']
                    dtheta = graph.get_data(self.name, 'dtheta').view(-1,2,4,3)
                    if 'bond_dihedral' in settings.crossterms[self.name]:
                        f = 0.0
                        f = torch.zeros(len(phi),4,3).to(index.device)
                        f += ((b[:,[0]]-r0[:,[0]])*(
                            -1*crossargs['k1b1d']*sin_phi
                            -2*crossargs['k2b1d']*sin_2phi
                            -3*crossargs['k3b1d']*sin_3phi
                        )).unsqueeze(1)*dphi
                        f += (
                            crossargs['k1b1d']*cos_phi+
                            crossargs['k2b1d']*cos_2phi+
                            crossargs['k3b1d']*cos_3phi
                        ).unsqueeze(1)*db[:,0]
                        f += ((b[:,[1]]-r0[:,[1]])*(
                            -1*crossargs['k1b2d']*sin_phi
                            -2*crossargs['k2b2d']*sin_2phi
                            -3*crossargs['k3b2d']*sin_3phi
                        )).unsqueeze(1)*dphi
                        f += (
                            crossargs['k1b2d']*cos_phi+
                            crossargs['k2b2d']*cos_2phi+
                            crossargs['k3b2d']*cos_3phi
                        ).unsqueeze(1)*db[:,1]
                        f += ((b[:,[2]]-r0[:,[2]])*(
                            -1*crossargs['k1b3d']*sin_phi
                            -2*crossargs['k2b3d']*sin_2phi
                            -3*crossargs['k3b3d']*sin_3phi
                        )).unsqueeze(1)*dphi
                        f += (
                            crossargs['k1b3d']*cos_phi+
                            crossargs['k2b3d']*cos_2phi+
                            crossargs['k3b3d']*cos_3phi
                        ).unsqueeze(1)*db[:,2]
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['bond_dihedral'] -= torch.zeros_like(
                                F['bond_dihedral']).scatter_add(0, index, f)
                    if 'angle_dihedral' in settings.crossterms[self.name]:
                        f = 0.0
                        f = torch.zeros(len(phi),4,3).to(index.device)
                        f += ((theta[:,[0]]-theta0[:,[0]])*(
                            -1*crossargs['k1a1d']*sin_phi
                            -2*crossargs['k2a1d']*sin_2phi
                            -3*crossargs['k3a1d']*sin_3phi
                        )).unsqueeze(1)*dphi
                        f += (
                            crossargs['k1a1d']*cos_phi+
                            crossargs['k2a1d']*cos_2phi+
                            crossargs['k3a1d']*cos_3phi
                        ).unsqueeze(1)*dtheta[:,0]
                        f += ((theta[:,[1]]-theta0[:,[1]])*(
                            -1*crossargs['k1a2d']*sin_phi
                            -2*crossargs['k2a2d']*sin_2phi
                            -3*crossargs['k3a2d']*sin_3phi
                        )).unsqueeze(1)*dphi
                        f += (
                            crossargs['k1a2d']*cos_phi+
                            crossargs['k2a2d']*cos_2phi+
                            crossargs['k3a2d']*cos_3phi
                        ).unsqueeze(1)*dtheta[:,1]
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['angle_dihedral'] -= torch.zeros_like(
                                F['angle_dihedral']).scatter_add(0, index, f)
                    if 'angle_angle_dihedral' in settings.crossterms[self.name]:
                        f = 0.0
                        f = torch.zeros(len(phi),4,3).to(index.device)
                        f += (crossargs['ka1a2d']*(
                            theta[:,[1]]-theta0[:,[1]])*(
                            cos_phi)
                        ).unsqueeze(1)*dtheta[:,0]
                        f += (crossargs['ka1a2d']*(
                            theta[:,[0]]-theta0[:,[0]])*(
                            cos_phi)
                        ).unsqueeze(1)*dtheta[:,1]
                        f += (crossargs['ka1a2d']*(
                            theta[:,[0]]-theta0[:,[0]])*(
                            theta[:,[1]]-theta0[:,[1]])*(
                            -1*sin_phi)
                        ).unsqueeze(1)*dphi
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['angle_angle_dihedral'] -= torch.zeros_like(
                                F['angle_angle_dihedral']).scatter_add(0, index, f) 
                    if 'bond1_bond3' in settings.crossterms[self.name]:
                        f = 0.0
                        f = torch.zeros(len(phi),4,3).to(index.device)
                        f += (crossargs['kb1b3']*(b[:,[0]]-r0[:,[0]])).unsqueeze(1)*db[:,2]
                        f += (crossargs['kb1b3']*(b[:,[2]]-r0[:,[2]])).unsqueeze(1)*db[:,0]
                        if len(index) != 0:
                            f = f.view(-1,3)
                            F['bond1_bond3'] -= torch.zeros_like(
                                F['bond1_bond3']).scatter_add(0, index, f)
        return(F)
        
    def _get_topology_vectors(self, node_vectors, index, use_base=False):
        if use_base:
            nonlinear_module = self.base_nonlinear
        else:
            nonlinear_module = self.nonlinear
        pair1 = nonlinear_module(torch.cat([
            node_vectors[index[:,1]],
            node_vectors[index[:,0]]
        ], dim=1))
        pair2 = nonlinear_module(torch.cat([
            node_vectors[index[:,2]],
            node_vectors[index[:,3]]
        ], dim=1))
        topology_vectors = pair1 + pair2
        return(topology_vectors)
    
    def _constrain_parameter(self, X, X_ref, param):
        deltaX = self.prefactors[param]*X
        X = X_ref+deltaX
        return(X)
    
class ImproperNet(TopologyNet):
    def __init__(self, details):
        super(ImproperNet, self).__init__('improper', 'chi', details)
        self.crosscoords = ['theta']
        self.base_predictors = torch.nn.ModuleDict({})
        self.predictors = torch.nn.ModuleDict({})
        for param in ['k2']:
            self.base_predictors[param] = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, 1) 
            self.predictors[param] = ParameterPredictor(2*self.details.Fr, self.details.Lh, 1)        
        self.default_reference_params = {'k2': 0}
        self.base_prefactors = {'k2': 0}
        self.prefactors = {'k2': 30}
        if 'angle_angle' in self.details.crossterms[self.name]:
            for param in ['ka1a2']:
                self.base_predictors[param] = ParameterPredictor(2*self.details.base_Fr, self.details.Lh, 1)
                self.predictors[param] = ParameterPredictor(2*self.details.Fr, self.details.Lh, 1)
                self.default_reference_params[param] = 0.0
                self.base_prefactors[param] = 0.1
                self.prefactors[param] = 0.1
        params = ['k2']
        self.direct_params = torch.nn.ParameterDict({
            param: torch.nn.Parameter(torch.zeros(100,1).to(torch.float)) for param in params})
        
    def E(self, index, graph, settings, chi, k2, **crossargs):
        E = {}
        E['improper'] = 0.0
        if self.name in settings.terms:
            E['improper'] = k2*chi.pow(2)
        else:
            E['improper'] = 0*k2*chi.pow(2)
        if self.details.style == 'compass' and settings.crossterms[self.name]:
            raise Exception("Must be debugged.")
            F['angle_angle'] = torch.zeros(graph.N, 3).to(index.device)
            angles = graph.get_data(self.name, 'angle')
            theta0 = graph.get_data('angle', 'theta0').view(-1)[angles]
            theta0 = theta0*(np.pi/180)
            theta = crossargs['theta']
            if 'angle_angle' in self.details.crossterms[self.name]:
                E['angle_angle'] = 0.0
                E['angle_angle'] += crossargs['ka1a2']*(theta[:,[0]]-theta0[:,[0]])
                E['angle_angle'] += crossargs['ka1a2']*(theta[:,[1]]-theta0[:,[1]])
                E['angle_angle'] += crossargs['ka1a2']*(theta[:,[2]]-theta0[:,[2]])                
        F = self.forces(index, graph, settings, chi, k2, **crossargs)
        return(E, F)
        
    def forces(self, index, graph, settings, chi, k2, **crossargs):
        F = {}
        if settings.get_forces:
            F['improper'] = torch.zeros(graph.N, 3).to(index.device)
            f = 0.0
            if self.details.style in ['compass', 'OPLS']:
                f += 2*k2*chi
            dchi = graph._data['improper']['dchi'].view(-1,4,3)
            index = index.view(-1,1).expand(4*len(index),3)
            if len(index) != 0:
                f = (f.unsqueeze(1)*dchi).view(-1,3)
                F['improper'] -= torch.zeros_like(F['improper']).scatter_add(0, index, f)
            if self.details.style == 'compass' and self.details.crossterms[self.name]:
                raise Exception("Must be debugged.")
                F['angle_angle'] = torch.zeros(graph.N, 3).to(index.device)
                angles = graph.get_data(self.name, 'angle')
                theta0 = graph.get_data('angle', 'theta0').view(-1)[angles]
                theta0 = theta0*(np.pi/180)
                theta = crossargs['theta']
                dtheta = graph.get_data(self.name, 'dtheta').view(-1,3,4,3)
                if 'angle_angle' in self.details.crossterms[self.name]:    
                    f = torch.zeros(len(theta),3,3).to(index.device)
                    f += (crossargs['ka1a2']*(theta[:,[0]]-theta0[:,[0]])*(
                        theta[:,[1]]-theta0[:,[1]])).unsqueeze(1)*dtheta
                    if len(index) != 0:
                        f = f.view(-1,3)
                        F['angle_angle'] -= torch.zeros_like(
                            F['angle_angle']).scatter_add(0, index, f)
        return(F)
        
    def _get_topology_vectors(self, node_vectors, index, use_base=False):
        topology_vectors = torch.cat([
            node_vectors[index[:,[0,2,3]]].sum(1),
            node_vectors[index[:,1]]
        ], dim=1)
        return(topology_vectors)
    
    def _constrain_parameter(self, X, X_ref, param):
        deltaX = self.prefactors[param]*X
        X = X_ref+deltaX
        return(X)
    
class PairNet(TopologyNet):
    def __init__(self, details):
        super(PairNet, self).__init__('pair', 'D', details)
        self.base_predictors = torch.nn.ModuleDict({
            'charge': ParameterPredictor(self.details.base_Fr, self.details.Lh, 1),
            'sigma': ParameterPredictor(self.details.base_Fr, self.details.Lh, 1),
            'epsilon': ParameterPredictor(self.details.base_Fr, self.details.Lh, 1)})
        self.predictors = torch.nn.ModuleDict({
            'charge': ParameterPredictor(self.details.Fr, self.details.Lh, 1),
            'sigma': ParameterPredictor(self.details.Fr, self.details.Lh, 1),
            'epsilon': ParameterPredictor(self.details.Fr, self.details.Lh, 1)})
        self.default_reference_params = {'charge': 0, 'sigma': 3, 'epsilon': 0.05}
        self.base_prefactors = {'charge': 0, 'sigma': 0, 'epsilon': 0}
        self.prefactors = {'charge': 0.1, 'sigma': 0.1, 'epsilon': 0.01}
        self.coulomb_constant = 332.0636 #kcal*Angstrom/mol
        params = ['charge', 'sigma', 'epsilon']
        self.direct_params = torch.nn.ParameterDict({
            param: torch.nn.Parameter(torch.zeros(100,1).to(torch.float)) for param in params})
        self.OPLS_sigma = torch.zeros(100,1).to(torch.float)
        self.OPLS_sigma[6] = 3.5 #3.8540
        self.OPLS_sigma[1] = 2.5 #2.8780
        self.OPLS_epsilon = torch.zeros(100,1).to(torch.float)
        self.OPLS_epsilon[6] = 0.066 #0.0620
        self.OPLS_epsilon[1] = 0.03 #0.0230

    def E(self, index, graph, settings, D, charge, sigma, epsilon):
        E, F = {}, {}
        E['LJ'], E['coulomb'] = 0.0, 0.0
        F['LJ'], F['coulomb'] = {}, {}
        F['LJ'] = torch.zeros(graph.N, 3).to(index.device)
        F['coulomb'] = torch.zeros(graph.N, 3).to(index.device)
        inv_D = graph._data['pair']['inv_D']
        inv_D_2 = graph._data['pair']['inv_D_2']
        C = graph._data['pair']['C'].view(-1,2,3)
        charge_product = charge.prod(1).view(-1,1)
        index = index.view(-1,1).expand(2*len(index),3) 
        if self.details.style == 'compass':
            sigma_mixed = sigma.pow(6).mean(1).pow(1.0/6).view(-1,1)
            epsilon_mixed = (2*epsilon.prod(1).pow(0.5)*(
                sigma.pow(3).prod(1)/sigma.pow(6).sum(1))).view(-1,1)            
        elif self.details.style == 'OPLS':
            #sigma_mixed = sigma.prod(1).pow(0.5).view(-1,1)
            sigma_mixed = (sigma+0.0001).prod(1).pow(0.5).view(-1,1) ################
            #epsilon_mixed = epsilon.prod(1).pow(0.5).view(-1,1) 
            epsilon_mixed = (epsilon+0.00001).prod(1).pow(0.5).view(-1,1) ################
        D_inv_scal = sigma_mixed*inv_D
        D_inv_scal_3 = D_inv_scal.pow(3)
        D_inv_scal_6 = D_inv_scal_3.pow(2)
        if self.name in settings.terms:
            E['coulomb'] = charge_product*inv_D*self.coulomb_constant
            f = -self.coulomb_constant*charge_product*inv_D_2
            if self.details.style == 'OPLS':
                f_1_4 = graph._data['pair']['f_1_4']
                E['coulomb'] *= f_1_4
                f *= f_1_4
            f = (f.unsqueeze(1)*C).view(-1,3)
            F['coulomb'] -= torch.zeros_like(F['coulomb']).scatter_add(0, index, f) ################       
            if self.details.style == 'OPLS':
                f_1_4 = graph._data['pair']['f_1_4']
                E['LJ'] = 4*epsilon_mixed*(D_inv_scal_6.pow(2)-D_inv_scal_6)
                E['LJ'] *= f_1_4
                f = -24*(epsilon_mixed*inv_D)*D_inv_scal_6*(2*D_inv_scal_6-1)
                f *= f_1_4
                if len(index) != 0:
                    f = (f.unsqueeze(1)*C).view(-1,3)
                    F['LJ'] -= torch.zeros_like(F['LJ']).scatter_add(0, index, f) ################
            elif self.details.style == 'compass':
                E['LJ'] = epsilon_mixed*(2*D_inv_scal_6*D_inv_scal_3-3*D_inv_scal_6)
                f = -18*(epsilon_mixed*inv_D)*D_inv_scal_6*(D_inv_scal_3-1)
                if len(index) != 0:
                    f = (f.unsqueeze(1)*C).view(-1,3)
                    F['LJ'] -= torch.zeros_like(F['LJ']).scatter_add(0, index, f)
        else:
            E['coulomb'] = 0*charge_product*inv_D*self.coulomb_constant
            E['LJ'] = 0*4*epsilon_mixed*(D_inv_scal_6.pow(2)-D_inv_scal_6)
        return(E, F)
        
    def _get_topology_vectors(self, node_vectors, index):
        topology_vectors = node_vectors
        return(topology_vectors)
    
    def _constrain_parameter(self, X, X_ref, param):
        deltaX = self.prefactors[param]*X
        X = X_ref+deltaX
        return(X)