import torch, itertools
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

class Hook():
    def __init__(self, module, direction='forward'):
        if direction == 'forward':
            self.hook = module.register_forward_hook(self.hook_fn)
        elif direction == 'backward':
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            raise Exception("The direction must be either forward or backward.")
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        
    def close(self):
        self.hook.remove()
        
class GenericForward(torch.nn.Module):
    def __init__(self):
        super(GenericForward, self).__init__()
        self.modules = OrderedDict()
        self.hooks = {'forward': OrderedDict(), 'backward': OrderedDict()}
        
    def add_layer_hooks(self, hooks, layer, direction):
        if type(layer[1]) != torch.nn.modules.container.Sequential:
            hooks[layer[0]] = Hook(layer[1], direction)
        else:
            hooks[layer[0]] = OrderedDict()
            for inner_layer in list(layer[1]._modules.items()):
                hooks[layer[0]] = self.add_layer_hooks(
                    hooks[layer[0]], inner_layer, direction)
        return(hooks)
        
    def initialize_hooks(self, directions=['forward']):
        for direction in directions:
            for layer in list(self._modules.items()):
                self.hooks[direction] = self.add_layer_hooks(
                    self.hooks[direction], layer, direction)
        
    def remove_hooks(self, directions=['forward', 'backward']):
        raise Exception("Debug this.")
        for direction in directions:
            for hook in self.hooks[direction].items():
                hook[1].close()
            self.hooks[direction] = OrderedDict()
            
    def clean_input_params(self):
        if self.depth is None:
            self.depth = len(self.L_hidden)+1
        else:
            self.L_hidden = [L_hidden for l in range(self.depth-1)]
            
    def build_modules(self, first_in_features, last_out_features, p=None):
        if p is None:
            for i in range(self.depth-1):
                if i == 0:
                    in_features = first_in_features
                else:
                    in_features = self.L_hidden[i-1]
                out_features = self.L_hidden[i]
                self.modules[str(2*i)] = torch.nn.Linear(
                    in_features, out_features)
                self.modules[str(2*i+1)] = torch.nn.Tanh()
            self.modules[str(2*(i+1))] =  torch.nn.Linear(
                self.L_hidden[-1], last_out_features)
        else:
            for i in range(self.depth-1):
                if i == 0:
                    in_features = first_in_features
                else:
                    in_features = self.L_hidden[i-1]
                out_features = self.L_hidden[i]
                self.modules[str(3*i)] = torch.nn.Linear(
                    in_features, out_features)
                self.modules[str(3*i+1)] = torch.nn.Tanh()
                self.modules[str(3*i+2)] = torch.nn.Dropout(p=p[i])
            self.modules[str(3*(i+1))] =  torch.nn.Linear(
                self.L_hidden[-1], last_out_features)
    
    def get_activations(self, direction='forward', exclude_linear_layers=False):
        activations = OrderedDict()
        for hook in self.hooks[direction].items():
            if type(hook[1]) == OrderedDict:
                activations[hook[0]] = OrderedDict()
                for inner_hook in self.hooks[direction][hook[0]].items():
                    if exclude_linear_layers:
                        if 'out_features' in self._modules[
                            hook[0]]._modules[inner_hook[0]].__dict__:
                            continue
                    activations[hook[0]][inner_hook[0]] = inner_hook[1].output
            else:
                if exclude_linear_layers:
                    if 'out_features' in self._modules[hook[0]].__dict__:
                        continue
                activations[hook[0]] = hook[1].output
        if type(self).__name__ == 'Aggregate':
            layer_index = str(int(list(activations['message_function'].items())[-1][0])+1)
            activations['message_function'][layer_index] = self.r_aggregated
        elif type(self).__name__ == 'Combine':
            layer_index = str(int(list(activations['update_function'].items())[-1][0])+1)
            activations['update_function'][layer_index] = self.r_updated
        elif type(self).__name__ == 'Readout':
            layer_index = str(int(list(activations['fingerprint_function'].items())[-1][0])+1)
            activations['fingerprint_function'][layer_index] = self.f
        elif type(self).__name__ == 'Predictor':
            layer_index = str(int(list(activations['predictor_function'].items())[-1][0])+1)
            activations['predictor_function'][layer_index] = self.y
        return(activations)

    def set_target_activations(self, module_types):
        self.target_activations = OrderedDict()
        for module_type in module_types:
            if module_type == 'predictor_function':
                container = self
            elif module_type == 'message_function':
                container = self.aggregate
            elif module_type == 'update_function':
                container = self.combine
            elif module_type == 'fingerprint_function':
                container = self.readout
            self.target_activations[module_type] = OrderedDict()
            for layer_index, layer_activations in container.get_activations()[module_type].items():
                if layer_index in container.modules.keys():
                    if hasattr(container.modules[layer_index], 'out_features'):
                        self.target_activations[module_type][layer_index] = torch.randn_like(layer_activations)
                    else:
                        self.target_activations[module_type][layer_index] = container.modules[layer_index](
                            torch.randn_like(layer_activations))
                else:
                    self.target_activations[module_type][layer_index] = torch.randn_like(layer_activations)       

    def get_pretrain_loss(self, reset=False):
        loss = 0.0
        if type(self).__name__ == 'Predictor':
            module_types = ['predictor_function']
            if (not hasattr(self, 'target_activations')) or reset:
                self.set_target_activations(module_types)
            for module_type in module_types:
                for layer_index, layer_activations in self.get_activations()[module_type].items():
                    loss += (self.target_activations[module_type][layer_index]-layer_activations).pow(2).mean()
        elif type(self).__name__ == 'Convolve':
            module_types = ['message_function', 'update_function', 'fingerprint_function']
            if (not hasattr(self, 'target_activations')) or reset:
                self.set_target_activations(module_types)
            for module_type in module_types:
                if module_type == 'predictor_function':
                    container = self
                elif module_type == 'message_function':
                    container = self.aggregate
                elif module_type == 'update_function':
                    container = self.combine
                elif module_type == 'fingerprint_function':
                    container = self.readout
                for layer_index, layer_activations in container.get_activations()[module_type].items():
                    loss += (self.target_activations[module_type][layer_index]-layer_activations).pow(2).mean()
        elif type(self).__name__ == 'Convolutions':
            for convolve in self.convolutions:
                loss += convolve.get_pretrain_loss(reset)
        return(loss)
    
class Aggregate(GenericForward):
    def __init__(self, Fr, Fe, L_hidden, L_message, depth=None):
        super(Aggregate, self).__init__()
        self.Fr = Fr
        self.Fe = Fe
        self.L_hidden = L_hidden
        self.L_message = L_message
        self.depth = depth
        self.clean_input_params()
        self.build_modules(
            first_in_features=self.Fr+self.Fe,
            last_out_features=self.L_message
        )
        self.message_function = torch.nn.Sequential(self.modules)
        self.sequential = {'message_function': self.message_function}
        self.initialize_hooks()
        
    def forward(self, r, e, a, d):
        edge_node = torch.cat([e, r[a[:,1]]], dim=1)
        pre_messages = self.message_function(edge_node)
        pre_messages_list = list(torch.split(pre_messages, d.tolist(), dim=0))
        messages = [pre_messages.sum(0) for pre_messages in pre_messages_list]
        self.r_aggregated = torch.stack(messages)
        return(self.r_aggregated)
    
class Combine(GenericForward):
    def __init__(self, Fr, Fe, L_message, L_hidden, Fr_out, depth=None):
        super(Combine, self).__init__()
        self.Fr = Fr
        self.Fe = Fe
        self.L_message = L_message
        self.L_hidden = L_hidden
        self.Fr_out = Fr_out
        self.depth = depth
        self.clean_input_params()
        self.build_modules(
            first_in_features=self.Fr+self.L_message,
            last_out_features=self.Fr_out
        )
        self.update_function = torch.nn.Sequential(self.modules)
        self.sequential = {'update_function': self.update_function}
        self.initialize_hooks()
        
    def forward(self, r, r_aggregated):
        node_and_message = torch.cat([r, r_aggregated], dim=1)
        self.r_updated = self.update_function(node_and_message)
        return(self.r_updated)
    
class Readout(GenericForward):
    def __init__(self, Fr, L_hidden, L_fingerprint, depth=None):
        super(Readout, self).__init__()
        self.Fr = Fr
        self.L_hidden = L_hidden
        self.L_fingerprint = L_fingerprint
        self.depth = depth
        self.clean_input_params()
        self.build_modules(
            first_in_features=self.Fr,
            last_out_features=self.L_fingerprint
        )
        self.fingerprint_function = torch.nn.Sequential(self.modules)
        self.sequential = {'fingerprint_function': self.fingerprint_function}
        self.initialize_hooks()
        
    def forward(self, r, N, softmax=False, tanh=True):
        f = self.fingerprint_function(r)
        if softmax:
            f = torch.nn.Softmax(dim=1)(f)
        f = list(torch.split(f, N.tolist()))
        f = [f[b].sum(0) for b in range(len(N))]
        self.f = torch.stack(f)
        if tanh:
            self.f = torch.nn.Tanh()(self.f)
        return(self.f)
    
    def visualize_fingerprints(self):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        imgplot = ax.imshow(
            self.f.t().cpu().detach(),
            vmin=-1, vmax=1, aspect='equal')
        imgplot.set_cmap('binary')
        ax.tick_params(labelcolor='none',
            top=False, bottom=False, left=False, right=False)
        ax.grid(False)
        
    def get_squared_pairwise_distances(self):
        N = self.f.shape[0]
        L_fingerprint = self.f.shape[1]
        displacements = self.f.unsqueeze(1).expand(N,N,L_fingerprint)
        displacements = -(displacements-displacements.transpose(0,1))
        D2 = displacements.pow(2).sum(2)
        D2 = D2.sum()
        return(D2)
    
class Predictor(GenericForward):
    def __init__(self, L_in, L_hidden, L_out, p=None, depth=None):
        super(Predictor, self).__init__()
        self.L_in = L_in
        self.L_hidden = L_hidden
        self.L_out = L_out
        self.p = p
        self.depth = depth
        self.clean_input_params()
        self.build_modules(
            first_in_features=self.L_in,
            last_out_features=self.L_out,
            p=self.p
        )
        self.predictor_function = torch.nn.Sequential(self.modules)
        self.sequential = {'predictor_function': self.predictor_function}
        self.initialize_hooks()
        
    def forward(self, x):
        self.y = self.predictor_function(x)
        return(self.y)
    
class Convolve(GenericForward):
    def __init__(self, aggregate, combine, readout=None):
        super(Convolve, self).__init__()
        self.aggregate = aggregate
        self.combine = combine
        self.readout = readout
        self.initialize_hooks()
        
    def forward(self, r, e, a, d, N):
        r_aggregated = self.aggregate(r, e, a, d)
        r_updated = self.combine(r, r_aggregated)
        if self.readout is not None:
            f = self.readout(r_updated, N)
        else:
            f = None
        return(r_updated, f)
    
class Convolutions(GenericForward):
    def __init__(self, convolutions):
        super(Convolutions, self).__init__()
        self.convolutions = torch.nn.ModuleList(convolutions)
        self.initialize_hooks()
        
    def forward(self, batch, job_details):
        r = batch.get_data('node', job_details.mpnn_input_label).to(torch.float)
        a = batch.get_data('edge', 'index')
        e = torch.zeros_like(a[:,[0]]).to(torch.float)
        d = batch.get_data('node', 'degree').view(-1)
        N = batch.get_data('num', 'node').view(-1)        
        for k in range(len(self.convolutions)):
            r, f = self.convolutions[k](r, e, a, d, N)
        return(r, f)