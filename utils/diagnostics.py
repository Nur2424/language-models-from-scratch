"""
utils/diagnostics.py
Reusable visualization functions for inspecting trained PyTorch networks.
 
All four functions follow the same interface convention:
    save_path (str or None): if provided, the figure is saved to this path.
    show (bool): if True, plt.show() is called after plotting.
 
Functions and their expected inputs
------------------------------------
 
plot_activation_distributions(layers, save_path=None, show=False)
    Forward-pass activation histograms for every Tanh layer in `layers`.
    Requires: each Tanh layer has a .out attribute populated from a recent
    forward pass (the Tanh class in this repo sets self.out = x.tanh() inside
    __call__).
    Prints: mean, std, and saturation fraction (|out| > 0.97) per layer.
    What to look for: mean near 0, std around 0.65, saturation below 5%.
    A shrinking std across layers means activations are collapsing (gain too
    low); a growing std means they are exploding (gain too high).
 
plot_gradient_distributions(layers, save_path=None, show=False)
    Backward-pass gradient histograms for every Tanh layer in `layers`.
    Requires: a forward pass in which retain_grad() was called on each Tanh
    layer's output, followed by backward(). The Tanh class in this repo calls
    retain_grad() automatically inside __call__ when requires_grad is True.
    Uses layer.out.grad. Skips any layer whose .out.grad is None.
    Prints: mean and std per layer.
    What to look for: all lines at the same width. Vanishing gradients show
    as early layers having much narrower curves than later ones. Exploding
    gradients show as early layers having much wider curves.
 
plot_weight_grad_ratios(parameters, save_path=None, show=False)
    Gradient histograms for every 2D parameter (weight matrices only).
    Skips 1D tensors: biases, BatchNorm gammas, and betas.
    Requires: backward() has been called so p.grad is populated.
    Prints: shape, gradient mean, gradient std, and grad:data ratio per weight.
    What to look for: all histogram lines overlapping. A line that stands out
    means that parameter receives disproportionately strong or weak updates.
 
plot_update_ratios(ud_log, parameters, save_path=None, show=False)
    Update/data ratio plotted over training steps for each 2D parameter.
 
    ud_log[step][i] = log10((lr * p_i.grad).std() / p_i.data.std()) recorded
    during training for the i-th entry in `parameters`. The ratio answers: how
    large is the actual weight change relative to the weight itself?
 
    Only 2D parameters (weight matrices) are plotted. A horizontal dashed
    reference line is drawn at y = -3 (target: updates roughly 0.1% of weight
    magnitude per step). Lines sitting around -2 to -2.5 indicate the learning
    rate could be slightly lower; lines at -4 or below indicate barely any
    learning is happening.
"""
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
 
# Shared color palette: drawn in order for each traced line.
# Matches the palette used across the language-model series.
_PALETTE = ['steelblue', 'darkorange', 'seagreen', 'crimson', '#999999', '#7c3aed', '#0891b2']
 
 
def plot_activation_distributions(layers, save_path=None, show=False):
    """
    Histogram of tanh output values for all Tanh layers after a forward pass.
 
    Parameters
    ----------
    layers : list
        Layer objects from a trained network. Any layer whose class name is
        'Tanh' and that has a populated .out attribute is included.
    save_path : str or None
        File path to save the figure. Nothing is written when None.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_labels = []
    color_idx = 0
    for i, layer in enumerate(layers):
        if type(layer).__name__ != 'Tanh':
            continue
        if not hasattr(layer, 'out'):
            continue
        t = layer.out.detach().cpu()
        sat = (t.abs() > 0.97).float().mean().item() * 100
        print(
            f'layer {i:2d} (Tanh): '
            f'mean {t.mean():+.2f}  std {t.std():.2f}  '
            f'saturated {sat:.2f}%'
        )
        hy, hx = torch.histogram(t, density=True, bins=50)
        ax.plot(hx[:-1].numpy(), hy.numpy(),
                color=_PALETTE[color_idx % len(_PALETTE)], linewidth=1.5)
        legend_labels.append(f'layer {i}')
        color_idx += 1
    ax.set_title('Forward Pass Activation Distributions')
    ax.set_xlabel('tanh output value')
    ax.set_ylabel('density')
    if legend_labels:
        ax.legend(legend_labels, fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
 
 
def plot_gradient_distributions(layers, save_path=None, show=False):
    """
    Histogram of tanh output gradients for all Tanh layers after backward().
 
    Requires: backward() called after a forward pass where retain_grad() was
    invoked on each Tanh layer's output. The Tanh class in this repo does this
    automatically in __call__ when requires_grad is True.
 
    Parameters
    ----------
    layers : list
        Layer objects. Any layer whose class name is 'Tanh' and whose
        .out.grad is not None is included.
    save_path : str or None
        File path to save the figure.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_labels = []
    color_idx = 0
    for i, layer in enumerate(layers):
        if type(layer).__name__ != 'Tanh':
            continue
        if not hasattr(layer, 'out') or layer.out.grad is None:
            continue
        t = layer.out.grad.detach().cpu()
        print(
            f'layer {i:2d} (Tanh): '
            f'mean {t.mean():+.9f}  std {t.std():e}'
        )
        hy, hx = torch.histogram(t, density=True, bins=50)
        ax.plot(hx[:-1].numpy(), hy.numpy(),
                color=_PALETTE[color_idx % len(_PALETTE)], linewidth=1.5)
        legend_labels.append(f'layer {i}')
        color_idx += 1
    ax.set_title('Backward Pass Gradient Distributions')
    ax.set_xlabel('gradient value')
    ax.set_ylabel('density')
    if legend_labels:
        ax.legend(legend_labels, fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
 
 
def plot_weight_grad_ratios(parameters, save_path=None, show=False):
    """
    Gradient distribution histograms for 2D weight matrices only.
 
    1D tensors (biases, BatchNorm gammas and betas) are skipped. Requires
    backward() to have been called so p.grad is populated.
 
    Parameters
    ----------
    parameters : list of torch.Tensor
        All trainable parameters of the network. Only ndim == 2 tensors are
        included.
    save_path : str or None
        File path to save the figure.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_labels = []
    color_idx = 0
    for i, p in enumerate(parameters):
        if p.ndim != 2 or p.grad is None:
            continue
        ratio = p.grad.std() / p.data.std()
        print(
            f'weight {str(tuple(p.shape)):>12s}  '
            f'mean {p.grad.mean():+.6f}  '
            f'std {p.grad.std():.6e}  '
            f'grad:data {ratio:.6e}'
        )
        g_cpu = p.grad.detach().cpu()
        hy, hx = torch.histogram(g_cpu, density=True, bins=50)
        ax.plot(hx[:-1].numpy(), hy.numpy(),
                color=_PALETTE[color_idx % len(_PALETTE)], linewidth=1.5)
        legend_labels.append(f'param {i} {tuple(p.shape)}')
        color_idx += 1
    ax.set_title('Weight Gradient Distributions')
    ax.set_xlabel('gradient value')
    ax.set_ylabel('density')
    if legend_labels:
        ax.legend(legend_labels, fontsize=7)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
 
 
def plot_update_ratios(ud_log, parameters, save_path=None, show=False):
    """
    Update/data ratio over training steps for each 2D parameter.
 
    log10((lr * grad).std() / data.std()) should hover near -3 in a
    well-calibrated network (updates ~0.1% of weight magnitude per step).
    Above -3: updates too large, risk of instability.
    Below -3: updates too small, barely learning.
 
    Parameters
    ----------
    ud_log : list of list of float
        ud_log[step][i] = log10 ratio for the i-th entry in `parameters` at
        that step. Indices must align with `parameters`.
    parameters : list of torch.Tensor
        All trainable parameters. Only ndim == 2 tensors are plotted.
    save_path : str or None
        File path to save the figure.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    legend_labels = []
    color_idx = 0
    for i, p in enumerate(parameters):
        if p.ndim != 2:
            continue
        trace = [ud_log[step][i] for step in range(len(ud_log))]
        ax.plot(trace, color=_PALETTE[color_idx % len(_PALETTE)],
                alpha=0.8, linewidth=1.2)
        legend_labels.append(f'param {i} {tuple(p.shape)}')
        color_idx += 1
    # reference line: target update/data ratio is 10^-3 = 0.001
    ax.axhline(y=-3, color='#555555', linestyle='--', linewidth=1.5)
    ax.set_title('Update to Data Ratio (log10)')
    ax.set_xlabel('step')
    ax.set_ylabel('log10(update/data)')
    ax.set_ylim(-6, 0)
    if legend_labels:
        ax.legend(legend_labels + ['target -3'], fontsize=7)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)