import torch, random, copy
import materialbuilder
from materialbuilder import (
    topologies, terms, plotting,
    transformations, utils, units, output)
from materialbuilder.plotting import *
import numpy as np
from munch import Munch

kB = 0.0019872041

class BasePotential(torch.nn.Module):
    def __init__(self, details):
        super(BasePotential, self).__init__()
        self.details = details
        self.epoch = None
        self.split = None
        self.settings = None
        self.reset_dicts()
        
    def _set_settings(self, split, epoch):
        self.split = split
        self.epoch = epoch
        self.settings = Munch()
        self.settings.terms = copy.deepcopy(self.details.terms)
        self.settings.crossterms = copy.deepcopy(self.details.crossterms)
        for term, delay in self.details.term_delays.items():
            if self.epoch < delay:
                self.settings.terms.pop(self.settings.terms.index(term))
        for term, delay in self.details.crossterm_delays.items():
            if self.epoch < delay:
                self.settings.crossterms[term] = []
        if self.epoch < self.details.num_logloss_epochs:
            self.settings.lr = copy.deepcopy(
                self.details.logloss_learning_rate)
            self.settings.freeze_eq = True
        elif self.epoch < (self.details.num_logloss_epochs + 
                           self.details.num_offset_epochs):
            self.settings.lr = copy.deepcopy(
                self.details.offset_learning_rate)
            self.settings.freeze_eq = False
        else:
            self.settings.lr = copy.deepcopy(self.details.learning_rate)
            self.settings.freeze_eq = False
        self.settings.properties_to_calculate = copy.deepcopy(
            self.details.properties_to_calculate)
        self.settings.target_properties = copy.deepcopy(
            self.details.target_properties)
        self.settings.get_forces = 'F' in self.details.properties_to_calculate
        if 'F' in self.settings.target_properties:
            if self.epoch < self.details.force_delay:
                self.settings.target_properties.pop(
                    self.settings.target_properties.index('F'))
        self.settings.frozen_terms = []
        for term, start in self.details.term_freezes.items():
            if self.epoch >= start:
                self.settings.frozen_terms.append(term)
        
    def set_reference_statistics(self, dataset):
        graph = dataset.container_batch
        if 'bond' in self.details.terms:
            df = graph.get_dataframe('bond')[['type', 'b']]
            types = np.sort(df['type'].unique()).tolist()
            mu_b = [df[df['type']==t]['b'].mean() for t in types]
            mu_b = torch.tensor(mu_b)
            self.bondnet.reference_statistics = {'r0': mu_b}
        if 'angle' in self.details.terms:
            df = graph.get_dataframe('angle')[['type', 'theta']]
            types = np.sort(df['type'].unique()).tolist()
            mu_theta = [df[df['type']==t]['theta'].mean()*180/np.pi for t in types]
            mu_theta = torch.tensor(mu_theta)
            self.anglenet.reference_statistics = {'theta0': mu_theta}
        
    def set_lr(self, optimizer, lr):
        if optimizer is None:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    def _initialize_dicts(self):
        for _dict in [self.error, self.RMSE]:
            for key in _dict[self.split].keys():
                _dict[self.split][key][self.epoch] = []
        for _dict in [self.results]:
            for level in ['target', 'predicted']:
                for key in _dict[self.split][level].keys():
                    _dict[self.split][level][key][self.epoch] = []        
        
    def reset_dicts(self):
        template_dict = {'E': {}, 'F': {}, 'charge': {}, 'total_charge': {}}
        self.error, self.SSE, self.MSE, self.MAE, self.RMSE, self.results = {}, {}, {}, {}, {}, {}
        for _dict in [self.error, self.SSE, self.MSE, self.MAE, self.RMSE]:
            for split in ['train', 'test']:
                _dict[split] = copy.deepcopy(template_dict)
        for split in ['train', 'test']:
            self.results[split] = {}
            for status in ['target', 'predicted']:
                self.results[split][status] = copy.deepcopy(template_dict)        
        if self.details.style == 'OPLS':
            self.learned_params = {
                'bond': {'r0': [], 'k2': []},
                'angle': {'theta0': [], 'k2': []},
                'dihedral': {'k1': [], 'k2': [], 'k3': [], 'k4': []},
                'improper': {'k2': []},
                'pair': {'sigma': [], 'epsilon': [], 'charge': []}}
        elif self.details.style == 'compass':
            self.learned_params = {
                'bond': {'r0': [], 'k2': [], 'k3': [], 'k4': []},
                'angle': {'theta0': [], 'k2': [], 'k3': [], 'k4': [],
                          'kb1b2': [], 'kb1a': [], 'kb2a': []},
                'dihedral': {'k1': [], 'k2': [], 'k3': [],
                             'k1b1d': [], 'k2b1d': [], 'k3b1d': [],
                             'k1b2d': [], 'k2b2d': [], 'k3b2d': [],
                             'k1b3d': [], 'k2b3d': [], 'k3b3d': [], 
                             'k1a1d': [], 'k2a1d': [], 'k3a1d': [],
                             'k1a2d': [], 'k2a2d': [], 'k3a2d': [],
                             'ka1a2d': [], 'kb1b3': []},
                'improper': {'k2': []},
                'pair': {'sigma': [], 'epsilon': [], 'charge': []}}
        
    def _update_dicts(self, batch, skip_offset, E, F):
        for prop in self.error['train'].keys():
            if prop in self.settings.properties_to_calculate:
                if prop == 'E':
                    E_target = batch._data['prop']['E']
                    if not skip_offset:
                        E_target = E_target - self.E_avg[self.split][
                            batch._data['prop']['ensemble']]
                    target = E_target
                    predicted = E
                elif prop == 'F':
                    F_target = batch._data['node']['F']
                    target = F_target
                    predicted = F
                elif prop == 'charge':
                    if 'charge' not in batch._data['node'].keys():
                        charges = torch.cat([
                            torch.zeros_like(batch._data['node']['chelpg_charge']),
                            torch.zeros_like(batch._data['node']['chelpg_charge'])], dim=1)
                    else:
                        charges = torch.cat([
                            batch._data['node']['charge'],
                            batch._data['node']['chelpg_charge']], dim=1)
                    target = charges[:,[1]].view(-1,1)
                    predicted = charges[:,[0]].view(-1,1)
                elif prop == 'total_charge':
                    if 'charge' not in batch._data['node'].keys():
                        charges = torch.cat([
                            torch.zeros_like(batch._data['node']['chelpg_charge']),
                            torch.zeros_like(batch._data['node']['chelpg_charge'])], dim=1)
                    else:
                        charges = torch.cat([
                            batch._data['node']['charge'],
                            batch._data['node']['chelpg_charge']], dim=1)
                    charges = charges.split(
                        batch._data['num']['node'].view(-1).tolist())
                    charges = [x.sum(0) for x in charges]
                    charges = torch.stack(charges)
                    target = charges[:,[1]].view(-1,1)
                    predicted = charges[:,[0]].view(-1,1)                                
            else:
                device = batch.get_data('node', 'index').device
                target = torch.tensor(0.0).to(device)
                predicted = torch.tensor(0.0).to(device)
            self.results[self.split]['target'][prop][
                self.epoch].append(target.view(-1,1))
            self.results[self.split]['predicted'][prop][
                self.epoch].append(predicted.view(-1,1))
            error = (predicted-target).view(-1,1)
            self.error[self.split][prop][self.epoch].append(error)
    
    def _join_dicts(self):
        for prop in self.settings.properties_to_calculate:
            self.error[self.split][prop][self.epoch] = torch.cat(
                self.error[self.split][prop][self.epoch])
            for level in ['target', 'predicted']:
                self.results[self.split][level][prop][self.epoch] = torch.cat(
                    self.results[self.split][level][prop][self.epoch])
    
    def _populate_SSE(self):
        for prop in self.settings.properties_to_calculate:
            if len(self.error[self.split][prop][self.epoch]) != 0:
                self.SSE[self.split][prop][self.epoch] = self.error[
                    self.split][prop][self.epoch].pow(2).sum(0)

    def _populate_MSE(self):
        for prop in self.settings.properties_to_calculate:
            if len(self.error[self.split][prop][self.epoch]) != 0:
                self.MSE[self.split][prop][self.epoch] = self.error[
                    self.split][prop][self.epoch].pow(2).mean(0)
    
    def _populate_MAE(self):
        for prop in self.settings.properties_to_calculate:
            if len(self.error[self.split][prop][self.epoch]) != 0:
                self.MAE[self.split][prop][self.epoch] = self.error[
                    self.split][prop][self.epoch].abs().mean(0)
    
    def _populate_RMSE(self):
        for prop in self.settings.properties_to_calculate:
            if len(self.error[self.split][prop][self.epoch]) != 0:
                self.RMSE[self.split][prop][self.epoch] = self.error[
                    self.split][prop][self.epoch].pow(2).mean(0).pow(0.5)
        
    def _finalize_dicts(self):
        self._join_dicts()
        self._populate_SSE()
        self._populate_MSE()
        self._populate_MAE()
        self._populate_RMSE()
        for prop in self.settings.properties_to_calculate:
            self.SSE[self.split][prop][self.epoch] = self.SSE[
                self.split][prop][self.epoch].item()
            self.MSE[self.split][prop][self.epoch] = self.MSE[
                self.split][prop][self.epoch].item()
            self.MAE[self.split][prop][self.epoch] = self.MAE[
                self.split][prop][self.epoch].item()
            self.RMSE[self.split][prop][self.epoch] = self.RMSE[
                self.split][prop][self.epoch].item()
            self.error[self.split][prop][self.epoch] = self.error[
                self.split][prop][self.epoch].tolist()
        for level in ['target', 'predicted']:
            for prop in self.settings.properties_to_calculate:
                self.results[self.split][level][prop][self.epoch] = self.results[
                    self.split][level][prop][self.epoch].view(-1).tolist()
        
    def _update_learned_params(self, dataset):
        for top in self.details.terms:
            if top == 'pair':
                for param in ['sigma', 'epsilon', 'charge']:
                    self.learned_params[top][param].append(
                        dataset.batches[0]._data['node'][param].cpu().tolist())
            else:
                for param in self.learned_params[top].keys():
                    self.learned_params[top][param].append(
                        dataset.batches[0]._data[top][param].cpu().tolist()) 
    
    def to_lammps(self, batch, filename, smiles):
        output.write_lammps_data_file(batch, filename, smiles, self.details.style)
        
    def plot_params(self, top):
        figsize=(20,10)
        for param in self.learned_params[top].keys():
            fig = plt.figure()
            X = self.learned_params[top][param]
            X = [torch.tensor(x) for x in X]
            X = torch.stack(X).squeeze(2)
            X = X.unique(dim=1)
            X = X.tolist()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param)
            ax.plot(list(range(len(X))), X)
        
    def plot(self, epoch, hexbin=False):
        figsize=(20,10)
        fig = plt.figure(figsize=figsize)
        for counter, prop in enumerate(self.details.target_properties):   
            ax = fig.add_subplot(1, 2, counter+1)
            for split in ['test']:
                x = self.results[split]['predicted'][prop][epoch]
                y = self.results[split]['target'][prop][epoch]
                if hexbin:
                    extent=(min(x), max(x), min(y), max(y))
                    ax.hexbin(x=x, y=y, bins='log', cmap=colormap['test'],
                              gridsize=100, extent=extent, linewidths=0.01)                    
                else:
                    ax.scatter(x=x, y=y, s=20, alpha=0.55)
            if prop == 'E':
                ax.set_title('Energy (kcal/mol)')
            elif prop == 'F':
                ax.set_title('Force (kcal/mol-$\\AA$)')
            lim = torch.tensor(x+y).abs().max().item()
            ax.plot((-lim, lim), (-lim, lim), ls="--", c=".3", lw=3)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Target')
            textstr = 'RMSE_test = {:.3f}'.format(
                list(self.RMSE['test'][prop].values())[epoch])
            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
            ax.text(0.05, 0.955, textstr, transform=ax.transAxes,
                fontsize=16, verticalalignment='top', bbox=props)