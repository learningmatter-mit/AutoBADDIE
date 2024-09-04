'''
def visualize(self, module_types=None, scatter=False, index=None):
    color = itertools.cycle('bgrcmyk')
    if module_types is None:
        if type(self).__name__ == 'Predictor':
            module_types = ['predictor_function']
        elif type(self).__name__ == 'Convolve':
            module_types = ['message_function', 'update_function', 'fingerprint_function']
    for module_type in module_types:
        if module_type == 'message_function':
            parent_module = self.aggregate
        elif module_type == 'update_function':
            parent_module = self.combine
        elif module_type == 'fingerprint_function':
            parent_module = self.readout
        elif module_type == 'predictor_function':
            parent_module = self
        num_plots = len(parent_module.get_activations()[module_type])
        fig = plt.figure(figsize=(14,6))
        fig.suptitle(module_type)
        plt.rc('axes', labelsize=10)
        for layer_index, layer in parent_module.get_activations()[
            module_type].items():
            if index is not None:
                if index != int(layer_index):
                    continue
                else:
                    gridspec = {
                        'shape': (1,1),
                        'loc': (0,0),
                        'rowspan': 1, 'colspan': 1}
                    s = 1
            else:
                gridspec = {
                    'shape': (1,num_plots),
                    'loc': (0,int(layer_index)),
                    'rowspan': 1, 'colspan': 1}
                s = 0.2
            ax = plt.subplot2grid(**gridspec, fig=fig)
            avg = layer.mean(0)
            std = layer.std(0)
            lim = torch.stack([
                (avg-std).min().abs(),(avg+std).max()]).max().item()
            ax.set_xlim(-3,3)
            if scatter:
                _color = next(color)
                for n in range(layer.shape[0]):
                    ax.scatter(
                        x=np.array(layer[n,:].tolist()),
                        y=np.arange(layer.shape[1]),
                        color=_color,
                        s=s)
            else:
                ax.errorbar(
                    x=np.array(avg.tolist()),
                    y=np.arange(len(avg)),
                    xerr=np.array(std.tolist()),
                    yerr=None,
                    color=next(color),
                    fmt='none')
            ax.tick_params(
                labelcolor='black',
                top=False, bottom=False, left=False, right=False)
            for ylabel in ax.get_yticklabels():
                ylabel.set_fontsize(0.0)
                ylabel.set_visible(False)
            for xlabel in ax.get_xticklabels():
                xlabel.set_fontsize(0.0)
                xlabel.set_visible(False)
            ax.set_xlabel('{:.3f}'.format(lim))
            ax.grid(False)
'''