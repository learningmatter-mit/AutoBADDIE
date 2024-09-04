import torch, random, copy
import materialbuilder
from materialbuilder import (
    topologies, terms, plotting, basepotential,
    transformations, utils, units, output)
from materialbuilder.plotting import *
import numpy as np
from munch import Munch

kB = 0.0019872041

class AuToPotential(basepotential.BasePotential):
    def __init__(self, details, convolve=None, model=None):
        super(AuToPotential, self).__init__(details)
        self.bondnet = terms.BondNet(self.details)
        self.anglenet = terms.AngleNet(self.details)
        self.dihedralnet = terms.DihedralNet(self.details)
        self.impropernet = terms.ImproperNet(self.details) 
        self.pairnet = terms.PairNet(self.details)
        self.E_offset = torch.nn.Parameter(torch.zeros(self.details.num_molecule_types))
        self.E_avg = torch.zeros(len(self.E_offset))
        self.convolve = convolve
        self.model = model
        self.device = torch.device(details.device)
        
    def FWD(self, batch):
        E = torch.zeros_like(batch._data['num']['node']).to(torch.float)
        F = torch.zeros_like(batch._data['node']['xyz']).to(torch.float)
        if 'bond' in self.details.terms:
            if 'bond' in self.settings.frozen_terms:
                torch.set_grad_enabled(False)
            E_bond, F_bond = self.bondnet(batch, self.settings)
            if len(E_bond['total']) != 0:
                E += E_bond['total']
            F += F_bond['total']
            torch.set_grad_enabled(True)
        if 'angle' in self.details.terms:
            if 'angle' in self.settings.frozen_terms:
                torch.set_grad_enabled(False)
            E_angle, F_angle = self.anglenet(batch, self.settings)
            if len(E_angle['total']) != 0:
                E += E_angle['total']
            F += F_angle['total']
            torch.set_grad_enabled(True)
        if 'dihedral' in self.details.terms:
            if 'dihedral' in self.settings.frozen_terms:
                torch.set_grad_enabled(False)
            E_dihedral, F_dihedral = self.dihedralnet(batch, self.settings)
            if len(E_dihedral['total']) != 0:
                E += E_dihedral['total']
            F += F_dihedral['total']
            torch.set_grad_enabled(True)
        if 'improper' in self.details.terms:
            if 'improper' in self.settings.frozen_terms:
                torch.set_grad_enabled(False)
            E_improper, F_improper = self.impropernet(batch, self.settings)
            if len(E_improper['total']) != 0:
                E += E_improper['total']
            F += F_improper['total']
            torch.set_grad_enabled(True)
        if 'pair' in self.details.terms:
            if 'pair' in self.settings.frozen_terms:
                torch.set_grad_enabled(False)
            E_pair, F_pair = self.pairnet(batch, self.settings)
            if len(E_pair['total']) != 0:
                E += E_pair['total']
            F += F_pair['total']
            torch.set_grad_enabled(True)
        logloss, eqloss = 0.0, 0.0
        return(E, F, logloss, eqloss)  
        
    def forward(self, datasets, split, optimizer, epoch, skip_offset=False):
        self._set_settings(split, epoch)
        dataset = datasets[self.split]
        if type(dataset) != torch.utils.data.dataloader.DataLoader:
            if self.split == 'train' and self.details.shuffle:
                dataset.Shuffle()
            batch_indices = list(range(dataset.num_batches))
            random.shuffle(batch_indices)
        self._initialize_dicts()
        if self.split == 'train':
            optimizer.zero_grad()
        if type(dataset) != torch.utils.data.dataloader.DataLoader:
            iter_object = batch_indices
        else:
            iter_object = dataset
        for batch_counter, batch in enumerate(iter_object):
            #print(batch_counter)
            if type(dataset) != torch.utils.data.dataloader.DataLoader:
                batch = dataset.batches[batch]
            loss = 0.0
            batch = batch.to(self.device)
            if self.details.base_input_label == 'encoding' or self.details.input_label == 'encoding':
                chemprop_batch = datasets['chemprop_batches']['all']
                fingerprints, encodings = self.model(chemprop_batch.batch_graph())
                ordered_species_ids = datasets['chemprop_batches']['species_ids']
                encodings = list(encodings.split(datasets['chemprop_batches']['N']))
                fingerprints = list(fingerprints.split([1 for n in datasets['chemprop_batches']['N']]))
                encodings = {ordered_species_ids[i]: encodings[i] for i in range(len(encodings))}
                fingerprints = {ordered_species_ids[i]: fingerprints[i] for i in range(len(fingerprints))}
                species_ids = batch._data['prop']['species_id'].view(-1).tolist()
                encodings = torch.cat([encodings[species_id] for species_id in species_ids])
                fingerprints = torch.cat([fingerprints[species_id] for species_id in species_ids])
                batch._data['node']['encoding'] = encodings
                batch._data['prop']['fingerprint'] = fingerprints
            if self.convolve is not None:
                batch._data['node']['mpnn'], f = self.convolve(batch, self.details)
            self.set_lr(optimizer, self.settings.lr)
            E, F, logloss, eqloss = self.FWD(batch)
            if not skip_offset:
                raise Exception()
                #print('MUST FIX BECAUSE ENSEMBLE COUNTERS ARE DIFFERENT FOR TRAIN/TEST')
                #print('AND THE ENSEMBLES ARENT EVEN RIGHT')
                ensemble_index = batch.get_data('prop', 'ensemble').view(-1).tolist()
                E += self.E_offset[ensemble_index].view(-1,1)
            self._update_dicts(batch, skip_offset, E, F)
            if self.split == 'train':
                if self.epoch < self.details.num_logloss_epochs:
                    loss = eqloss
                else:
                    for prop in self.settings.target_properties:
                        loss += self.details.loss_prefactors[prop]*self.error[
                            self.split][prop][self.epoch][-1].pow(2).mean(0)
                loss.backward()
                if self.details.batch_multiplier == 1:
                    optimizer.step()
                elif batch_counter > 0:
                    if batch_counter % self.details.batch_multiplier == 0:
                        optimizer.step()
        self._finalize_dicts()
        RMSE = 0.0
        for prop in self.settings.properties_to_calculate:
            RMSE += self.RMSE[self.split][prop][self.epoch]
        #if self.epoch % self.details.frequency['update_parameters'] == 0:
        #    if (self.split == 'train') and (self.epoch > 0):
        #        self._update_learned_params(datasets['train'])
        return(RMSE, eqloss)
    
    def infer(self, dataset, skip_offset=False):
        batch_indices = list(range(dataset.num_batches))
        random.shuffle(batch_indices)
        self.epoch = 0
        self.reset_dicts()
        self._initialize_dicts()
        for batch_counter, batch_index in enumerate(batch_indices):
            batch = dataset.batches[batch_index]
            E, F, logloss, eqloss = self(batch)
            if not skip_offset:
                ensemble_index = batch.get_data('prop', 'ensemble').view(-1).tolist()
                E += 100*self.E_offset[ensemble_index].view(-1,1)
            self._update_dicts(batch, skip_offset, E, F)
        self._finalize_dicts()
