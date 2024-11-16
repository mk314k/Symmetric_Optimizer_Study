import torch

class Layer_Optimizer:
    def __init__(self, model, optims, lr_func):
        """
        Initializes the Layer_Optimizer.

        Args:
            model (torch.nn.Module): The PyTorch model with named parameters.
            optims (function/class or list): Optimizer function/class (e.g., `torch.optim.Adam`) 
                or a list of such optimizers for each layer.
            lr_func (function): A function that takes a layer index and returns a learning rate.
        """
        self.optims = {}
        
        # Handle single optimizer function/class
        if not isinstance(optims, list):
            optims = [optims]  # Use the same optimizer for all layers
        
        for name, param in model.named_parameters():
            layer_index = self.extract_layer_index(name)
            if layer_index is not None:
                # Determine which optimizer to use for the layer
                optim_func = optims[layer_index] if layer_index < len(optims) else optims[-1]

                if layer_index not in self.optims:
                    self.optims[layer_index] = optim_func(
                        [param], lr=lr_func(layer_index)
                    )
                else:
                    self.optims[layer_index].param_groups[0]['params'].append(param)

    def extract_layer_index(self, name):
        """
        Extracts the layer index from the parameter name.

        Args:
            name (str): Parameter name (e.g., "layer0.weight").

        Returns:
            int or None: The layer index if found, otherwise None.
        """
        parts = name.split('.')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        """
        for optim in self.optims.values():
            optim.zero_grad()

    def step(self):
        """
        Performs a single optimization step for all layers.
        """
        for optim in self.optims.values():
            optim.step()
