from functools import partial
import torch


class SaveFeatures:
    def __init__(self) -> None:
        self.features = {}
        self.gradients = {}  # Store gradients
        self.hooks = []

    def hook_fn(self, module, input, output, path) -> None:
        """Save feature maps (activations) from the forward pass."""
        self.features[path] = output

    def save_gradient(self, module, grad_input, grad_output, path) -> None:
        """Save gradients during the backward pass."""
        self.gradients[path] = grad_output[0]  # Store gradient for the module's output

    def register_hooks(self, module, parent_path, verbose):
        for name, child in module.named_children():
            current_path = f"{parent_path}.{name}" if parent_path else name

            if verbose:
                print(f"Registering Hook: {current_path}")

            # Register forward hook for activations
            hook = child.register_forward_hook(hook=partial(self.hook_fn, path=current_path))
            self.hooks.append(hook)

            # Register backward hook for gradients
            backward_hook = child.register_backward_hook(hook=partial(self.save_gradient, path=current_path))
            self.hooks.append(backward_hook)

            # Recursively register hooks on child modules
            self.register_hooks(child, current_path, verbose)

    def close(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
