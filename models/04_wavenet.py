# 04  WAVENET CHARACTER-LEVEL LANGUAGE MODEL
# After van den Oord et al. 2016 "WaveNet: A Generative Model for Raw Audio"
# Applied to character-level name generation
#
# Trains five models in sequence on names.txt and produces:
#   outputs/plots/04_loss_comparison.png
#   outputs/plots/04_model_comparison.png
#   outputs/plots/04_update_ratios.png
#   outputs/generated_names/04_wavenet_generated.txt
#
# Run from repo root:
#   python models/04_wavenet.py
 
import sys
import pathlib
import random
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
# make utils importable from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.diagnostics import plot_update_ratios
 
# ---- Hyperparameters ----
 
n_embd      = 10    # embedding dim for flat MLP and standard WaveNet models
n_hidden    = 200   # hidden units for flat MLP models
n_wn        = 68    # hidden units for standard WaveNet models (22k params)
n_embd_sc   = 24    # embedding dim for scaled WaveNet
n_hidden_sc = 128   # hidden units for scaled WaveNet (76k params)
batch_size  = 32
max_iters   = 200_000
lr_high     = 0.1
lr_low      = 0.01
 
# Flat MLPs decay LR earlier (simpler model converges faster)
lr_flat_decay = 100_000
 
# WaveNet models use a longer high-LR phase -- the tree structure needs more
# steps to propagate gradients all the way back to the embedding layer
lr_wn_decay   = 150_000
 
# ---- Data loading ----
 
with open("../language-models-from-scratch/data/names.txt", 'r') as file:
    words = file.read().splitlines()
chars = sorted(set(''.join(words)))
stoi  = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)   # 27: 26 letters + '.' as 0
 
print(f'Words: {len(words)}  Vocabulary: {vocab_size} symbols')
 
# ---- Dataset builder ----
 
def build_dataset(word_list, block_size):
    """Build (context, target) pairs with a sliding window of block_size."""
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)
 
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
 
# ---- Layer classes ----
# All implemented from scratch, no nn.Module.
 
class Linear:
    """Fully connected layer with optional bias.
 
    Kaiming-style init: weights ~ N(0, fan_in^-0.5) keeps activations
    roughly unit-variance after a linear transform, regardless of fan_in.
 
    Supports both 2D (B, C) and 3D (B, T, C) inputs via torch.matmul
    N-D broadcasting: (B, T, C) @ (C, D) = (B, T, D).
    """
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias   = torch.zeros(fan_out) if bias else None
 
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
 
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
 
 
class BatchNorm1d:
    """Batch normalization -- FIXED version that handles 3D input correctly.
 
    For 2D input (B, C): reduce over dim=0 (batch axis only).
    For 3D input (B, T, C): reduce over dim=(0,1) (batch AND time).
 
    The dim=(0,1) choice matters because both B and T are "samples" from
    the same distribution at each channel C. Reducing only over dim=0
    would give T separate statistics per channel, each estimated from just
    B=32 samples far too noisy, and the running stats end up with shape
    (1, T, C) instead of the expected (1, 1, C). See BuggyBatchNorm1d below
    for that silent failure.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps      = eps
        self.momentum = momentum
        self.training = True
        # learnable scale and shift, shape (dim,) broadcast over B [and T]
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        # running statistics accumulated during training, used at inference
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)
 
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0            # reduce over batch
            elif x.ndim == 3:
                dim = (0, 1)       # reduce over batch and time
            else:
                raise ValueError(f'BatchNorm1d: unsupported ndim {x.ndim}')
            xmean = x.mean(dim, keepdim=True)
            xvar  = x.var( dim, keepdim=True)
        else:
            # at inference use accumulated running stats (no batch available)
            xmean = self.running_mean
            xvar  = self.running_var
        xhat     = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out
 
    def parameters(self):
        return [self.gamma, self.beta]
 
 
class BuggyBatchNorm1d:
    """BatchNorm1d with the dim=0-only bug for pedagogical demonstration.
 
    For 3D input (B, T, C), reduces only over dim=0 instead of (0,1).
    This means xmean.shape = (1, T, C), so each of the T time positions
    gets its own mean and variance estimated from B=32 samples -- far too
    noisy. Running stats silently grow to shape (1, T, C) instead of (1,1,C).
 
    The model trains without errors but the running stats are wrong at
    inference time. This is the canonical silent BatchNorm shape bug.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps      = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta  = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)
 
    def __call__(self, x):
        if self.training:
            # BUG: always use dim=0, regardless of x.ndim.
            # For 3D (B, T, C) this gives xmean.shape = (1, T, C).
            xmean = x.mean(0, keepdim=True)
            xvar  = x.var( 0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        xhat     = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                # running_mean broadcasts from shape (dim,) to (1, T, C) after
                # first update -- shape mismatch is absorbed silently by PyTorch
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out
 
    def parameters(self):
        return [self.gamma, self.beta]
 
 
class Tanh:
    """Elementwise tanh. Stores output tensor for diagnostic inspection."""
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
 
    def parameters(self):
        return []
 
 
class Embedding:
    """Token lookup table: integer indices => dense vectors.
 
    Wraps the lookup as a layer so it participates in Sequential.parameters()
    and gets updated by the same SGD loop as all other weights.
    """
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
 
    def __call__(self, IX):
        # IX shape: (B, T) -> out shape: (B, T, embedding_dim)
        self.out = self.weight[IX]
        return self.out
 
    def parameters(self):
        return [self.weight]
 
 
class Flatten:
    """Flatten all post-batch dims: (B, T, C) => (B, T*C).
 
    Used in flat MLP models where the entire context window is concatenated
    before the single hidden layer. Not used in WaveNet.
    """
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
 
    def parameters(self):
        return []
 
 
class FlattenConsecutive:
    """Merge n consecutive time steps into one wider token.
 
    (B, T, C) => (B, T//n, C*n)
 
    This is the WaveNet building block. Stacking three of these with n=2
    builds a hierarchical binary tree: level-1 sees pairs of characters,
    level-2 sees pairs of level-1 outputs (4 chars), level-3 sees 8 chars.
    The receptive field doubles with each level without any parameter cost.
 
    The reshape is zero-copy because PyTorch's C-contiguous layout stores
    adjacent time steps contiguously in memory: elements [t, :] and [t+1, :]
    for a given batch are already next to each other, so .view() just
    reinterprets the strides.
 
    When T//n == 1 (only one token left after merging), squeeze(1) removes
    the time dimension so downstream Linear layers get 2D input (B, C*n).
    """
    def __init__(self, n):
        self.n = n
 
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)   # drop the singleton time dimension
        self.out = x
        return self.out
 
    def parameters(self):
        return []
 
 
class Sequential:
    """Apply a list of layers in order. Aggregate their parameters."""
    def __init__(self, layers):
        self.layers = layers
 
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
 
    def parameters(self):
        # flatten parameter lists from all child layers into one list
        return [p for layer in self.layers for p in layer.parameters()]
 
 
# ---- Training and evaluation helpers ----
 
def train(model, Xtr, Ytr, max_iters, lr_decay_at, ud_log=None, label='model'):
    """Train model with SGD and optional update/data ratio tracking.
 
    Parameters
    ----------
    model        : Sequential
    Xtr, Ytr     : training split tensors
    max_iters    : total gradient steps
    lr_decay_at  : step at which LR drops from lr_high to lr_low
    ud_log       : list to append per-step update ratios (scaled WaveNet only)
    label        : name printed in progress messages
    """
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad_(True)
 
    lossi = []   # log10(loss) per step -- 200k values, smoothed for plotting
    for step in range(max_iters):
        # LR schedule: high for first lr_decay_at steps, then drop 10x
        lr = lr_high if step < lr_decay_at else lr_low
 
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]
 
        # forward pass
        logits = model(Xb)
        loss   = F.cross_entropy(logits, Yb)
 
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
 
        # record update/data ratio BEFORE the weight update so p.data is the
        # pre-step weight; ratio = log10(|step| / |weight|), target near -3
        if ud_log is not None:
            with torch.no_grad():
                ratios = [
                    ((lr * p.grad).std() / p.data.std()).log10().item()
                    for p in parameters
                ]
                ud_log.append(ratios)
 
        # SGD weight update
        with torch.no_grad():
            for p in parameters:
                p.data -= lr * p.grad
 
        lossi.append(loss.log10().item())
 
        if (step + 1) % 20_000 == 0:
            recent_smooth = torch.tensor(lossi[-1000:]).mean().item()
            print(f'  {label}  step {step+1:6d}/{max_iters}  '
                  f'loss {loss.item():.4f}  smoothed {recent_smooth:.4f}')
 
    return lossi
 
 
@torch.no_grad()
def evaluate(model, splits):
    """Evaluate model on each split. Returns dict of {name: NLL float}."""
    for layer in model.layers:
        layer.training = False
 
    results = {}
    for name, (X, Y) in splits.items():
        logits = model(X)
        results[name] = F.cross_entropy(logits, Y).item()
 
    for layer in model.layers:
        layer.training = True
 
    return results
 
 
@torch.no_grad()
def generate_names(model, block_size, n=20, seed=2147483647):
    """Generate n names. Fixed seed ensures reproducible output."""
    torch.manual_seed(seed)
    for layer in model.layers:
        layer.training = False
 
    names = []
    for _ in range(n):
        out     = []
        context = [0] * block_size
        while True:
            x      = torch.tensor([context])       # (1, block_size)
            logits = model(x)
            probs  = F.softmax(logits, dim=-1)
            ix     = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        names.append(''.join(out))
    return names
 
 
# ---- Build datasets ----
# block_size=3 for the flat-3 baseline; block_size=8 for everything else.
 
data3 = {}
data8 = {}
for bs, store in [(3, data3), (8, data8)]:
    store['train'] = build_dataset(words[:n1],   bs)
    store['dev']   = build_dataset(words[n1:n2], bs)
    store['test']  = build_dataset(words[n2:],   bs)
 
Xtr3, Ytr3   = data3['train']
Xdev3, Ydev3 = data3['dev']
Xtr8, Ytr8   = data8['train']
Xdev8, Ydev8 = data8['dev']
 
print(f'block_size=3  train {Xtr3.shape}  dev {Xdev3.shape}')
print(f'block_size=8  train {Xtr8.shape}  dev {Xdev8.shape}')
 
# Containers for end-of-run plots
all_lossi  = {}   # model_name -> list[float] of per-step log10 loss (200k entries)
dev_losses = {}   # model_name -> float dev NLL (5 entries including buggy)
 
# ---- Model 1: Flat MLP, block_size=3 ----
# Baseline: flatten 3 embeddings (3*10=30 dims) into a single hidden layer.
# Only sees 3 characters, so receptive field is limited.
# Expected: train ~2.058, dev ~2.107
 
print('\n--- Model 1: Flat MLP (block_size=3) ---')
torch.manual_seed(42)
model_flat3 = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * 3, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
n_params = sum(p.nelement() for p in model_flat3.parameters())
print(f'Parameters: {n_params:,}')   # expected 12,097
 
lossi_flat3 = train(
    model_flat3, Xtr3, Ytr3,
    max_iters=max_iters, lr_decay_at=lr_flat_decay,
    label='flat-3',
)
res_flat3 = evaluate(model_flat3, {'train': (Xtr3, Ytr3), 'dev': (Xdev3, Ydev3)})
print(f'Flat-3  train {res_flat3["train"]:.4f}  dev {res_flat3["dev"]:.4f}')
all_lossi['Flat-3']   = lossi_flat3
dev_losses['Flat-3']  = res_flat3['dev']
 
# ---- Model 2: Flat MLP, block_size=8 ----
# Wider context: 8 embeddings flattened to 80 dims. Same hidden size.
# Longer context already helps over flat-3 even without hierarchical structure.
# Expected: train ~1.922, dev ~2.029
 
print('\n--- Model 2: Flat MLP (block_size=8) ---')
torch.manual_seed(42)
model_flat8 = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * 8, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
n_params = sum(p.nelement() for p in model_flat8.parameters())
print(f'Parameters: {n_params:,}')   # expected 22,097
 
lossi_flat8 = train(
    model_flat8, Xtr8, Ytr8,
    max_iters=max_iters, lr_decay_at=lr_flat_decay,
    label='flat-8',
)
res_flat8 = evaluate(model_flat8, {'train': (Xtr8, Ytr8), 'dev': (Xdev8, Ydev8)})
print(f'Flat-8  train {res_flat8["train"]:.4f}  dev {res_flat8["dev"]:.4f}')
all_lossi['Flat-8']   = lossi_flat8
dev_losses['Flat-8']  = res_flat8['dev']
 
# ---- Model 3: WaveNet with BUGGY BatchNorm (pedagogical demonstration) ----
# Three levels of FlattenConsecutive(2) build a binary tree over 8 chars.
# BatchNorm here uses the buggy dim=0-only reduction, producing running
# statistics with the wrong shape. The model still trains (no crash), but
# inference behavior is subtly wrong.
# Expected: train ~1.944, dev ~2.030
 
print('\n--- Model 3: WaveNet (BUGGY BatchNorm -- pedagogical) ---')
torch.manual_seed(42)
model_wn_buggy = Sequential([
    Embedding(vocab_size, n_embd),
    # level 1: pairs of 2 tokens (4 pairs from 8-char context)
    FlattenConsecutive(2), Linear(n_embd * 2, n_wn, bias=False), BuggyBatchNorm1d(n_wn), Tanh(),
    # level 2: pairs of level-1 outputs (2 pairs)
    FlattenConsecutive(2), Linear(n_wn * 2,   n_wn, bias=False), BuggyBatchNorm1d(n_wn), Tanh(),
    # level 3: final merge -- T becomes 1, squeeze fires, output is 2D
    FlattenConsecutive(2), Linear(n_wn * 2,   n_wn, bias=False), BuggyBatchNorm1d(n_wn), Tanh(),
    Linear(n_wn, vocab_size),
])
n_params = sum(p.nelement() for p in model_wn_buggy.parameters())
print(f'Parameters: {n_params:,}')   # expected 22,397
 
lossi_wn_buggy = train(
    model_wn_buggy, Xtr8, Ytr8,
    max_iters=max_iters, lr_decay_at=lr_wn_decay,
    label='wn-buggy',
)
res_wn_buggy = evaluate(model_wn_buggy, {'train': (Xtr8, Ytr8), 'dev': (Xdev8, Ydev8)})
print(f'WaveNet buggy  train {res_wn_buggy["train"]:.4f}  dev {res_wn_buggy["dev"]:.4f}')
 
# Expose the silent bug: the FIRST BuggyBatchNorm1d received (B, 4, 68) input.
# Its running_mean started at shape (68,) but grew to (1, 4, 68) because
# xmean from dim=0 reduction has shape (1, 4, 68), and PyTorch broadcasts
# the running-stat update silently. Result: 272 separate statistics estimated
# from only 32 samples each -- much too noisy.
for layer in model_wn_buggy.layers:
    if type(layer).__name__ == 'BuggyBatchNorm1d':
        rm = layer.running_mean
        print(f'\n[BN BUG] running_mean.shape = {tuple(rm.shape)} -- expected (1, 1, {n_wn})')
        print(f'         {rm.numel()} independent statistics estimated from {batch_size} values each.')
        print(f'         Fixed version uses dim=(0,1) for 3D input.')
        break
 
# Buggy model excluded from the loss-curve plot but included in bar chart
dev_losses['WaveNet-buggy'] = res_wn_buggy['dev']
 
# ---- Model 4: WaveNet with FIXED BatchNorm ----
# Same architecture, but BatchNorm1d now uses dim=(0,1) for 3D input.
# Running stats stay shape (1, 1, C) -- correct.
# Expected: train ~1.912, dev ~2.020
 
print('\n--- Model 4: WaveNet (fixed BatchNorm) ---')
torch.manual_seed(42)
model_wn = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_wn, bias=False), BatchNorm1d(n_wn), Tanh(),
    FlattenConsecutive(2), Linear(n_wn * 2,   n_wn, bias=False), BatchNorm1d(n_wn), Tanh(),
    FlattenConsecutive(2), Linear(n_wn * 2,   n_wn, bias=False), BatchNorm1d(n_wn), Tanh(),
    Linear(n_wn, vocab_size),
])
n_params = sum(p.nelement() for p in model_wn.parameters())
print(f'Parameters: {n_params:,}')   # expected 22,397
 
lossi_wn = train(
    model_wn, Xtr8, Ytr8,
    max_iters=max_iters, lr_decay_at=lr_wn_decay,
    label='wn-fixed',
)
res_wn = evaluate(model_wn, {'train': (Xtr8, Ytr8), 'dev': (Xdev8, Ydev8)})
print(f'WaveNet fixed  train {res_wn["train"]:.4f}  dev {res_wn["dev"]:.4f}')
all_lossi['WaveNet']   = lossi_wn
dev_losses['WaveNet']  = res_wn['dev']
 
# ---- Model 5: WaveNet scaled (n_embd=24, n_hidden=128) ----
# Scale up embedding and hidden dims to fully utilize the hierarchical
# structure. ~76k parameters vs 22k before.
# Expected: train ~1.767, dev ~1.985
 
print('\n--- Model 5: WaveNet scaled (n_embd=24, n_hidden=128) ---')
 
# Shape trace: verify tensor shapes flow correctly through the tree before
# committing to a full 200k-step training run.
print('\nShape trace (batch=4):')
torch.manual_seed(42)
_model_trace = Sequential([
    Embedding(vocab_size, n_embd_sc),
    FlattenConsecutive(2), Linear(n_embd_sc * 2,    n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden_sc * 2,  n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden_sc * 2,  n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    Linear(n_hidden_sc, vocab_size),
])
x_probe = Xtr8[:4]
for layer in _model_trace.layers:
    x_probe = layer(x_probe)
    print(f'  {type(layer).__name__:20s}: {tuple(x_probe.shape)}')
 
# Re-seed so the actual training model gets the same initial weights as
# a fresh construction.
ud_log = []   # ud_log[step][i] = log10((lr*grad_i).std() / data_i.std())
 
torch.manual_seed(42)
model_wn_sc = Sequential([
    Embedding(vocab_size, n_embd_sc),
    FlattenConsecutive(2), Linear(n_embd_sc * 2,    n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden_sc * 2,  n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden_sc * 2,  n_hidden_sc, bias=False), BatchNorm1d(n_hidden_sc), Tanh(),
    Linear(n_hidden_sc, vocab_size),
])
n_params_sc = sum(p.nelement() for p in model_wn_sc.parameters())
print(f'\nParameters: {n_params_sc:,}')   # expected 76,579
 
lossi_wn_sc = train(
    model_wn_sc, Xtr8, Ytr8,
    max_iters=max_iters, lr_decay_at=lr_wn_decay,
    ud_log=ud_log,
    label='wn-scaled',
)
res_wn_sc = evaluate(model_wn_sc, {'train': (Xtr8, Ytr8), 'dev': (Xdev8, Ydev8)})
print(f'WaveNet scaled  train {res_wn_sc["train"]:.4f}  dev {res_wn_sc["dev"]:.4f}')
all_lossi['WaveNet-scaled']   = lossi_wn_sc
dev_losses['WaveNet-scaled']  = res_wn_sc['dev']
 
# ---- Summary ----
 
print('\n======== Final Results ========')
expected = {
    'Flat-3':        (2.0583, 2.1065),
    'Flat-8':        (1.9215, 2.0294),
    'WaveNet-buggy': (1.9442, 2.0302),
    'WaveNet':       (1.9119, 2.0197),
    'WaveNet-scaled':(1.7669, 1.9845),
}
all_res = {
    'Flat-3':         res_flat3,
    'Flat-8':         res_flat8,
    'WaveNet-buggy':  res_wn_buggy,
    'WaveNet':        res_wn,
    'WaveNet-scaled': res_wn_sc,
}
for name, res in all_res.items():
    exp_tr, exp_dev = expected[name]
    print(f'  {name:16s}  train {res["train"]:.4f} (exp {exp_tr})  '
          f'dev {res["dev"]:.4f} (exp {exp_dev})')
 
# ---- Save outputs ----
 
pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)
pathlib.Path('outputs/generated_names').mkdir(parents=True, exist_ok=True)
 
print('\nGenerating plots...')
 
# Plot 1: Smoothed training loss comparison (4 models, buggy excluded)
# Raw lossi has 200,000 noisy points. Reshape to (200, 1000) and take
# row means -> 200 smooth points, one per 1000-step window.
 
palette = {
    'Flat-3':         'steelblue',
    'Flat-8':         'darkorange',
    'WaveNet':        'seagreen',
    'WaveNet-scaled': 'crimson',
}
 
fig, ax = plt.subplots(figsize=(10, 4.5))
for name, lossi in all_lossi.items():
    smooth = torch.tensor(lossi).view(-1, 1000).mean(1)
    ax.plot(smooth.numpy(), label=name, color=palette[name])
ax.set_title('Training Loss -- Smoothed (1000-step windows)')
ax.set_xlabel('step (x1000)')
ax.set_ylabel('log10 NLL')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/plots/04_loss_comparison.png', dpi=110)
plt.close(fig)
print('Saved outputs/plots/04_loss_comparison.png')
 
# Plot 2: Model comparison bar chart (dev NLL, all 5 models)
# WaveNet-buggy uses #999999 (muted gray) to signal it is the broken baseline.
 
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(dev_losses.keys())
dev_nlls    = list(dev_losses.values())
bar_colors  = ['steelblue', 'darkorange', '#999999', 'seagreen', 'crimson']
bars = ax.barh(model_names, dev_nlls, color=bar_colors)
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
ax.set_xlabel('Dev NLL')
ax.set_title('Model Comparison -- Dev NLL (lower is better)')
ax.set_xlim(1.5, 2.3)
ax.invert_yaxis()   # best model at top
plt.tight_layout()
plt.savefig('outputs/plots/04_model_comparison.png', dpi=110)
plt.close(fig)
print('Saved outputs/plots/04_model_comparison.png')
 
# Plot 3: Update/data ratio for scaled WaveNet
# ud_log[step][i] = log10((lr*grad_i).std() / weight_i.std())
# Target: lines hovering near -3 (updates ~0.1% of weight magnitude per step).
 
plot_update_ratios(
    ud_log,
    model_wn_sc.parameters(),
    save_path='outputs/plots/04_update_ratios.png',
)
print('Saved outputs/plots/04_update_ratios.png')
 
# ---- Generated names ----
 
print('\nGenerating names with scaled WaveNet (seed=2147483647)...')
names = generate_names(model_wn_sc, block_size=8, n=20, seed=2147483647)
 
with open('outputs/generated_names/04_wavenet_generated.txt', 'w') as f:
    f.write(f'Model    : WaveNet character-level language model (van den Oord et al. 2016)\n')
    f.write(f'Train NLL: {res_wn_sc["train"]:.4f}  |  Dev NLL: {res_wn_sc["dev"]:.4f}  |  Parameters: {n_params_sc:,}\n')
    f.write('--------------------------------\n')
    for name in names:
        f.write(name + '\n')
print('Saved outputs/generated_names/04_wavenet_generated.txt')
 
print('\nSample names:')
for name in names[:10]:
    print(f'  {name}')
 
print('\nDone')