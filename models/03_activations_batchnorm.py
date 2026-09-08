# 03 ACTIVATIONS, GRADIENTS, AND BATCH NORMALIZATION

# Runs the full progression from the notebook:
#   1. Baseline (no fix)          => initial loss ~26, val NLL ~2.12
#   2. Fix 1: scale W2, zero b2   => val NLL ~2.13
#   3. Fix 2: scale W1, b1        => val NLL ~2.10
#   4. Kaiming initialization     => val NLL ~2.10
#   5. Inline Batch Normalization  => val NLL ~2.11
#   6. Deep 6-layer network        => val NLL ~2.40 (context bottleneck)

# Produces:
#   outputs/plots/03_initialization_comparison.png
#   outputs/plots/03_deep_training_loss.png
#   outputs/plots/03_forward_activations.png
#   outputs/plots/03_backward_gradients.png
#   outputs/plots/03_weight_grad_ratios.png
#   outputs/plots/03_update_ratios.png
#   outputs/generated_names/03_deep_network_generated.txt


import sys
import pathlib
import random
import datetime
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# make utils importable from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.diagnostics import (
    plot_activation_distributions,
    plot_gradient_distributions,
    plot_weight_grad_ratios,
    plot_update_ratios,
)

# ---- Hyperparameters ----

block_size       = 3
n_embd           = 10
n_hidden_s       = 200       # hidden units for shallow (1-layer) models
n_hidden_d       = 100       # hidden units for the deep (6-layer) model
batch_size       = 32
max_iters        = 200_000
lr_high          = 0.1
lr_low           = 0.01
lr_decay_shallow = 100_000   # LR step for shallow models
lr_decay_deep    = 150_000   # LR step for the deep model (note: different)
INIT_SEED        = 2147483647

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

def build_dataset(word_list):
    """Build (context, target) pairs using a sliding window of block_size"""
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
Xtr,  Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])
print(f'Train: {Xtr.shape}  Dev: {Xdev.shape}  Test: {Xte.shape}')

# ------------ Layer classes ----------------
# These mirror torch.nn internals exactly. Each class owns its weights and
# implements __call__ for the forward pass and parameters() for the optimizer

class Linear:
    """
    Fully connected linear layer with Kaiming initialization built in.
    bias=False is used when BatchNorm follows: BatchNorm subtracts the mean,
    so any constant bias offset is immediately canceled and wastes a parameter.
    """
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming init: divide by sqrt(fan_in) to cancel the natural sqrt(fan_in)
        # growth from matrix multiplication, keeping activations at unit variance
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
        self.bias   = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """
    Batch normalization over the feature dimension (dim 1)

    During training: normalizes using the current mini-batch mean and variance,
    then updates running statistics with a slow exponential moving average.

    During inference (training=False): uses the accumulated running statistics
    so single example inference gives the same result as batch inference.

    gamma (scale) and beta (shift) are learnable parameters that let the
    network undo the normalization if that is what the loss requires.
    They start at the identity (gamma=1, beta=0) so BatchNorm does nothing at
    initialization and the network learns how much normalization to apply.

    eps=1e-5 prevents division by zero if a neuron's variance is exactly 0,
    momentum=0.1 means each step: 90% old estimate + 10% current batch.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps          = eps
        self.momentum     = momentum
        self.training     = True
        self.gamma        = torch.ones(dim)    # learnable scale
        self.beta         = torch.zeros(dim)   # learnable shift
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)   # mean across batch, per neuron
            xvar  = x.var(0, keepdim=True)    # var across batch, per neuron
        else:
            xmean = self.running_mean
            xvar  = self.running_var
        # normalize: zero mean, unit variance (plus eps for numerical safety)
        xhat     = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                # running stats: not trained by gradients, updated manually
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    """
    Tanh non-linearity. Stores .out for forward activation inspection.
    Calls retain_grad() on .out so backward-pass gradients are accessible
    for diagnostic visualization without extra code in the training loop.
    No trainable parameters.
    """
    def __call__(self, x):
        self.out = x.tanh()
        if self.out.requires_grad:
            # retain_grad() tells PyTorch to keep this intermediate tensor's
            # gradient after backward() instead of freeing it immediately
            self.out.retain_grad()
        return self.out

# ------------ Helper functions ----------------

@torch.no_grad()
def eval_shallow(C, W1, b1, W2, b2, X, Y):
    """Full split NLL for the non-BN shallow model"""
    emb    = C[X]
    h      = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


@torch.no_grad()
def eval_shallow_bn(C, W1, W2, b2, bngain, bnbias, bnmean_r, bnstd_r, X, Y):
    """Full-split NLL for the inline-BN shallow model using running stats."""
    emb     = C[X]
    embcat  = emb.view(-1, n_embd * block_size)
    hpreact = embcat @ W1
    # use accumulated running stats, not the current-batch stats
    hpreact = bngain * (hpreact - bnmean_r) / bnstd_r + bnbias
    h       = torch.tanh(hpreact)
    logits  = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()


def run_shallow(C, W1, b1, W2, b2, g, lr_decay_at=None, tag=''):
    """
    Train the non-BN shallow MLP for max_iters steps

    Uses p.grad = None before backward (frees gradient tensor, not just zeros
    it is faster and uses less memory than zero_grad())

    LR schedule: lr_high for steps 0..lr_decay_at-1, lr_low thereafter
    Returns log10 loss list
    """
    if lr_decay_at is None:
        lr_decay_at = lr_decay_shallow
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    lossi = []
    for i in range(max_iters):
        ix  = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb    = C[Xb]
        # view() is zero copy: reinterprets tensor layout without allocation
        h      = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
        logits = h @ W2 + b2
        loss   = F.cross_entropy(logits, Yb)
        for p in parameters:
            p.grad = None   # frees gradient tensor rather than zeroing it
        loss.backward()
        lr = lr_high if i < lr_decay_at else lr_low
        for p in parameters:
            p.data += -lr * p.grad
        lossi.append(loss.log10().item())
        if i % 50_000 == 0:
            print(f'  [{tag}] step {i:6d}  loss {loss.item():.4f}')
    return lossi


def run_shallow_bn(C, W1, W2, b2, bngain, bnbias, g, tag=''):
    """
    Train the inline-BN shallow MLP for max_iters steps

    b1 is deliberately absent: BatchNorm subtracts the mean after the linear
    layer, so any constant offset added by b1 is immediately cancelled,
    bnbias (beta) handles the learned offset role

    Running stats (bnmean_r, bnstd_r) are updated with slow momentum=0.001
    matching the notebook's inline implementation (not the class version's 0.1)
    At inference, use bnmean_r and bnstd_r in place of batch statistics

    Returns (lossi, bnmean_r, bnstd_r)
    """
    parameters  = [C, W1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True
    bnmean_r = torch.zeros((1, n_hidden_s))
    bnstd_r  = torch.ones ((1, n_hidden_s))
    lossi    = []
    for i in range(max_iters):
        ix  = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        emb     = C[Xb]
        embcat  = emb.view(-1, n_embd * block_size)
        hpreact = embcat @ W1   # no b1, BatchNorm makes it redundant
        # normalize across the batch for each of the n_hidden_s neurons
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi  = hpreact.std(0,  keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
        with torch.no_grad():
            # slow EMA: 0.999 * old + 0.001 * current batch
            bnmean_r = 0.999 * bnmean_r + 0.001 * bnmeani
            bnstd_r  = 0.999 * bnstd_r  + 0.001 * bnstdi
        h      = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss   = F.cross_entropy(logits, Yb)
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = lr_high if i < lr_decay_shallow else lr_low
        for p in parameters:
            p.data += -lr * p.grad
        lossi.append(loss.log10().item())
        if i % 50_000 == 0:
            print(f'  [{tag}] step {i:6d}  loss {loss.item():.4f}')
    return lossi, bnmean_r, bnstd_r

# ------------ Phase 1: Baseline (no initialization fix) ----------------

# Default randn weights give wildly uneven logits at step 0,
# Softmax sees large differences and assigns near-100% probability to one
# character, making the model catastrophically overconfident before any data

print('\n== Phase 1: Baseline (no fix) ==')
g  = torch.Generator().manual_seed(INIT_SEED)
C  = torch.randn((vocab_size, n_embd),          generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden_s), generator=g)
b1 = torch.randn(n_hidden_s,                    generator=g)
W2 = torch.randn((n_hidden_s, vocab_size),      generator=g)
b2 = torch.randn(vocab_size,                    generator=g)

with torch.no_grad():
    emb    = C[Xtr[:batch_size]]
    h      = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    init_loss_nofix = F.cross_entropy(logits, Ytr[:batch_size]).item()
print(f'  initial loss: {init_loss_nofix:.4f}  (expected ~3.30 if uniform, {-torch.log(torch.tensor(1/27.0)).item():.4f})')

for p in [C, W1, b1, W2, b2]:
    p.requires_grad = True

lossi_nofix = run_shallow(C, W1, b1, W2, b2, g, tag='no-fix')
train_nll_nofix = eval_shallow(C, W1, b1, W2, b2, Xtr, Ytr)
val_nll_nofix   = eval_shallow(C, W1, b1, W2, b2, Xdev, Ydev)
print(f'  train NLL: {train_nll_nofix:.4f}  val NLL: {val_nll_nofix:.4f}')

# ------------ Phase 2: Fix 1 ------------ scale W2 by 0.01, zero b2 ------------

# W2 * 0.01 makes all logits near zero at init, so softmax sees tiny differences
# and assigns roughly equal probability (1/27) to all 27 characters

# This gives initial loss near the theoretical baseline of 3.29 = -log(1/27)
# b2 = zeros: no systematic bias in the output layer at init

print('\n== Phase 2: Fix 1 (W2 * 0.01, b2 = zeros) ==')
g  = torch.Generator().manual_seed(INIT_SEED)
C  = torch.randn((vocab_size, n_embd),          generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden_s), generator=g)
b1 = torch.randn(n_hidden_s,                    generator=g)
W2 = torch.randn((n_hidden_s, vocab_size),      generator=g) * 0.01   # Fix 1
b2 = torch.zeros(vocab_size)                                           # Fix 1

with torch.no_grad():
    emb    = C[Xtr[:batch_size]]
    h      = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    init_loss_fix1 = F.cross_entropy(logits, Ytr[:batch_size]).item()
print(f'  initial loss: {init_loss_fix1:.4f}  (should be near 3.29)')

for p in [C, W1, b1, W2, b2]:
    p.requires_grad = True

lossi_fix1 = run_shallow(C, W1, b1, W2, b2, g, tag='fix1')
train_nll_fix1 = eval_shallow(C, W1, b1, W2, b2, Xtr, Ytr)
val_nll_fix1   = eval_shallow(C, W1, b1, W2, b2, Xdev, Ydev)
print(f'  train NLL: {train_nll_fix1:.4f}  val NLL: {val_nll_fix1:.4f}')

# ------------ Phase 3: Fix 2 ------------ scale W1 and b1 ------------

# With default W1, pre-activations (hpreact = emb @ W1 + b1) spread between
# roughly -15 and +15. tanh squashes everything outside (-3, +3) to exactly
# +/-1. Those flat regions have zero gradient: neurons stuck there never learn

# W1 * 0.2 keeps pre-activations in the active middle range of tanh
# b1 * 0.01 reduces the bias contribution to the same active range

print('\n== Phase 3: Fix 2 (W1 * 0.2, b1 * 0.01) ==')
g  = torch.Generator().manual_seed(INIT_SEED)
C  = torch.randn((vocab_size, n_embd),          generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden_s), generator=g) * 0.2    # Fix 2
b1 = torch.randn(n_hidden_s,                    generator=g) * 0.01       # Fix 2
W2 = torch.randn((n_hidden_s, vocab_size),      generator=g) * 0.01       # Fix 1
b2 = torch.zeros(vocab_size)                                               # Fix 1

with torch.no_grad():
    emb    = C[Xtr[:batch_size]]
    h      = torch.tanh(emb.view(-1, n_embd * block_size) @ W1 + b1)
    logits = h @ W2 + b2
    init_loss_fix2 = F.cross_entropy(logits, Ytr[:batch_size]).item()
print(f'  initial loss: {init_loss_fix2:.4f}')

for p in [C, W1, b1, W2, b2]:
    p.requires_grad = True

lossi_fix2 = run_shallow(C, W1, b1, W2, b2, g, tag='fix2')
train_nll_fix2 = eval_shallow(C, W1, b1, W2, b2, Xtr, Ytr)
val_nll_fix2   = eval_shallow(C, W1, b1, W2, b2, Xdev, Ydev)
print(f'  train NLL: {train_nll_fix2:.4f}  val NLL: {val_nll_fix2:.4f}')

# ------------ Phase 4: Kaiming initialization ------------

# The 0.2 in Fix 2 was a guess tuned for this architecture,
# Kaiming gives the principled formula: divide by sqrt(fan_in)
# Multiplying by (5/3) compensates for tanh's slight variance shrinkage:
# every pass through tanh reduces the spread slightly; the 5/3 gain on
# the linear layer pumps it back up to keep the spread stable across layers,
# 5/3 is the tanh gain; for ReLU the corresponding gain is sqrt(2)

print('\n== Phase 4: Kaiming initialization ==')
g  = torch.Generator().manual_seed(INIT_SEED)
fan_in_1 = n_embd * block_size   # = 30
C  = torch.randn((vocab_size, n_embd),       generator=g)
W1 = torch.randn((fan_in_1, n_hidden_s),     generator=g) * (5/3) / fan_in_1 ** 0.5
b1 = torch.randn(n_hidden_s,                 generator=g) * 0.01
W2 = torch.randn((n_hidden_s, vocab_size),   generator=g) * 0.01
b2 = torch.zeros(vocab_size)

with torch.no_grad():
    emb    = C[Xtr[:batch_size]]
    h      = torch.tanh(emb.view(-1, fan_in_1) @ W1 + b1)
    logits = h @ W2 + b2
    init_loss_kaiming = F.cross_entropy(logits, Ytr[:batch_size]).item()
print(f'  initial loss: {init_loss_kaiming:.4f}')

for p in [C, W1, b1, W2, b2]:
    p.requires_grad = True

lossi_kaiming = run_shallow(C, W1, b1, W2, b2, g, tag='kaiming')
train_nll_kaiming = eval_shallow(C, W1, b1, W2, b2, Xtr, Ytr)
val_nll_kaiming   = eval_shallow(C, W1, b1, W2, b2, Xdev, Ydev)
print(f'  train NLL: {train_nll_kaiming:.4f}  val NLL: {val_nll_kaiming:.4f}')

# ------------ Phase 5: Inline Batch Normalization ------------

# Instead of carefully initializing weights, force pre-activations to be
# roughly Gaussian (mean 0, std 1) after every linear layer automatically

# The learnable gamma (scale) and beta (shift) start at the identity and let
# gradient descent decide how much normalization each layer actually needs

print('\n== Phase 5: Inline Batch Normalization ==')
g      = torch.Generator().manual_seed(INIT_SEED)
C      = torch.randn((vocab_size, n_embd),     generator=g)
W1     = torch.randn((fan_in_1, n_hidden_s),   generator=g) * (5/3) / fan_in_1 ** 0.5
# no b1: BatchNorm subtracts the batch mean, so b1 has zero effect
W2     = torch.randn((n_hidden_s, vocab_size), generator=g) * 0.01
b2     = torch.zeros(vocab_size)
bngain = torch.ones ((1, n_hidden_s))   # gamma: starts as identity
bnbias = torch.zeros((1, n_hidden_s))   # beta:  starts as identity

with torch.no_grad():
    emb     = C[Xtr[:batch_size]]
    embcat  = emb.view(-1, fan_in_1)
    hpreact = embcat @ W1
    bni     = hpreact.mean(0, keepdim=True)
    bns     = hpreact.std(0,  keepdim=True)
    hpreact = bngain * (hpreact - bni) / bns + bnbias
    logits  = torch.tanh(hpreact) @ W2 + b2
    init_loss_bn = F.cross_entropy(logits, Ytr[:batch_size]).item()
print(f'  initial loss: {init_loss_bn:.4f}')

lossi_bn, bnmean_r, bnstd_r = run_shallow_bn(C, W1, W2, b2, bngain, bnbias, g, tag='bn')
train_nll_bn = eval_shallow_bn(C, W1, W2, b2, bngain, bnbias, bnmean_r, bnstd_r, Xtr,  Ytr)
val_nll_bn   = eval_shallow_bn(C, W1, W2, b2, bngain, bnbias, bnmean_r, bnstd_r, Xdev, Ydev)
print(f'  train NLL: {train_nll_bn:.4f}  val NLL: {val_nll_bn:.4f}')

# ------------ Summary of shallow model progression ------------

print('\n---- Shallow model progression ----')
print(f'{"Config":<22}  {"Init Loss":>9}  {"Train NLL":>9}  {"Val NLL":>9}')
print('-' * 58)
rows = [
    ('No fix',            init_loss_nofix,    train_nll_nofix,    val_nll_nofix),
    ('Fix 1: W2*0.01',    init_loss_fix1,     train_nll_fix1,     val_nll_fix1),
    ('Fix 2: W1*0.2',     init_loss_fix2,     train_nll_fix2,     val_nll_fix2),
    ('Kaiming init',      init_loss_kaiming,  train_nll_kaiming,  val_nll_kaiming),
    ('Batch Norm',        init_loss_bn,       train_nll_bn,       val_nll_bn),
]
for name, il, tr, vl in rows:
    print(f'{name:<22}  {il:>9.4f}  {tr:>9.4f}  {vl:>9.4f}')

# ------------ Phase 6: Deep 6-layer network with layer classes ------------

# With three layer classes we can build a much deeper network as a plain list

# Six Linear layers alternating with BatchNorm and Tanh, feeding cross-entropy
# directly from the last BatchNorm (no Tanh at the end)

# Post-construction init fix:
#   layers[-1].gamma *= 0.1  same motivation as W2 * 0.01 in shallow models:
#   keeps the last layer's output logits small at init. When the last layer is
#   BatchNorm, scale its gamma (not a weight matrix)
#   layer.weight *= 1.0 for hidden Linear layers the (5/3) tanh-gain slot
#   is left at 1.0 because BatchNorm handles the scale automatically after each
#   layer. In a network without BatchNorm, write * (5/3) here instead

print('\n== Phase 6: Deep 6-layer network ==')
g = torch.Generator().manual_seed(INIT_SEED)

C_deep = torch.randn((vocab_size, n_embd), generator=g)

layers = [
    Linear(n_embd * block_size, n_hidden_d, bias=False), BatchNorm1d(n_hidden_d), Tanh(),
    Linear(n_hidden_d, n_hidden_d, bias=False),           BatchNorm1d(n_hidden_d), Tanh(),
    Linear(n_hidden_d, n_hidden_d, bias=False),           BatchNorm1d(n_hidden_d), Tanh(),
    Linear(n_hidden_d, n_hidden_d, bias=False),           BatchNorm1d(n_hidden_d), Tanh(),
    Linear(n_hidden_d, n_hidden_d, bias=False),           BatchNorm1d(n_hidden_d), Tanh(),
    Linear(n_hidden_d, vocab_size, bias=False),           BatchNorm1d(vocab_size),
]

with torch.no_grad():
    # scale last layer's gamma down to fix overconfident softmax at init
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            # (5/3) tanh-gain slot: left at 1.0 because BatchNorm handles scaling
            layer.weight *= 1.0

parameters_deep = [C_deep]
for layer in layers:
    if hasattr(layer, 'parameters'):
        parameters_deep += layer.parameters()

for p in parameters_deep:
    p.requires_grad = True

n_params_deep = sum(p.nelement() for p in parameters_deep)
print(f'  Parameters: {n_params_deep}')   # 47024

# ------------ Training loop (deep model) --------------

# Two diagnostic additions beyond the standard loop:
#   retain_grad() on Tanh outputs: keep intermediate gradients after backward()
#   so visualization functions can inspect gradient flow through each layer,
#   The Tanh class calls retain_grad() automatically in __call__, so no extra
#   code is needed here.

#   ud list: at every step for each parameter record the log10 ratio of update
#   size to parameter size, Captures whether the learning rate is appropriate

lossi_deep = []
ud_log     = []

for i in range(max_iters):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass through the layer list
    emb = C_deep[Xb]               # (32, 3, 10)
    x   = emb.view(emb.shape[0], -1)   # (32, 30) zero-copy reshape
    for layer in layers:
        x = layer(x)
    logits = x                     # (32, 27)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters_deep:
        p.grad = None
    loss.backward()

    lr = lr_high if i < lr_decay_deep else lr_low
    for p in parameters_deep:
        p.data += -lr * p.grad

    lossi_deep.append(loss.log10().item())

    with torch.no_grad():
        ud_log.append([
            ((lr * p.grad).std() / p.data.std()).log10().item()
            for p in parameters_deep
        ])

    if i % 50_000 == 0:
        print(f'  [deep] step {i:6d}  loss {loss.item():.4f}')

print('Deep model training complete.')

# ------------ Evaluation (deep model) ------------

for layer in layers:
    layer.training = False   # use running stats for inference

@torch.no_grad()
def eval_deep(X, Y):
    emb = C_deep[X]
    x   = emb.view(X.shape[0], -1)
    for layer in layers:
        x = layer(x)
    return F.cross_entropy(x, Y).item()

train_nll_deep = eval_deep(Xtr,  Ytr)
val_nll_deep   = eval_deep(Xdev, Ydev)
print(f'  Deep model - train NLL: {train_nll_deep:.4f}  val NLL: {val_nll_deep:.4f}')
print('  (train ~= val means context bottleneck, not overfitting model saturated on 3-char context)')

# ------------ Diagnostic pass for gradient visualizations ------------

# The training loop's last backward() populated layer.out.grad for the last
# mini-batch, For cleaner diagnostics, run one fresh forward+backward on a
# fixed batch after setting layers back to training mode

for layer in layers:
    layer.training = True

emb_diag = C_deep[Xtr[:batch_size]]
x_diag   = emb_diag.view(batch_size, -1)
for layer in layers:
    x_diag = layer(x_diag)
loss_diag = F.cross_entropy(x_diag, Ytr[:batch_size])
loss_diag.backward()

# ------------ Diagnostic plots that require layer.out and layer.out.grad ---------

# Must come BEFORE generation: the name-generation loop calls each Tanh layer
# one token at a time (inside torch.no_grad()), which overwrites layer.out with
# a no-grad tensor and loses the gradient captured above

pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)

print('\nForward activation statistics:')
plot_activation_distributions(
    layers,
    save_path='outputs/plots/03_forward_activations.png',
)
print('Saved outputs/plots/03_forward_activations.png')

print('\nBackward gradient statistics:')
plot_gradient_distributions(
    layers,
    save_path='outputs/plots/03_backward_gradients.png',
)
print('Saved outputs/plots/03_backward_gradients.png')

# switch back to eval mode for generation
for layer in layers:
    layer.training = False

# ------------ Generate names from the deep model ------------

g_gen = torch.Generator().manual_seed(INIT_SEED)
names = []

with torch.no_grad():
    for _ in range(20):
        out     = []
        context = [0] * block_size
        while True:
            emb_s = C_deep[torch.tensor([context])]   # (1, 3, 10)
            x_s   = emb_s.view(1, -1)                 # (1, 30)
            for layer in layers:
                x_s = layer(x_s)
            probs = F.softmax(x_s, dim=1)
            ix    = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        names.append(''.join(itos[k] for k in out[:-1]))

print('\nGenerated names:')
for name in names:
    print(f'  {name}')

# ------------ Save outputs ------------

pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)
pathlib.Path('outputs/generated_names').mkdir(parents=True, exist_ok=True)

# Plot 1: Initialization comparison bar chart

# Two grouped bars per configuration: initial loss and final val NLL.
# The no-fix initial loss (~26) towers over the others, making the contrast
# between bad and good initialization immediately visible.
config_names = ['No Fix', 'Fix 1\nW2*0.01', 'Fix 2\nW1*0.2', 'Kaiming\nInit', 'Batch\nNorm']
init_losses  = [r[1] for r in rows]
val_nlls     = [r[3] for r in rows]

x_pos = np.arange(len(config_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x_pos - width/2, init_losses, width, label='Initial loss', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, val_nlls,    width, label='Final val NLL', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(config_names, fontsize=9)
ax.set_ylabel('loss / NLL')
ax.set_title('Initialization and Normalization: Effect on Training')
ax.legend()
ax.bar_label(bars1, fmt='%.2f', padding=2, fontsize=7)
ax.bar_label(bars2, fmt='%.2f', padding=2, fontsize=7)
plt.tight_layout()
plt.savefig('outputs/plots/03_initialization_comparison.png', dpi=110)
plt.close()
print('Saved outputs/plots/03_initialization_comparison.png')

# Plot 2: Deep model training loss

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(lossi_deep, alpha=0.4)
ax.axvline(x=lr_decay_deep, color='red', linestyle='--', linewidth=1,
           label=f'LR decay  {lr_high} -> {lr_low}  (step {lr_decay_deep//1000}k)')
ax.set_xlabel('step')
ax.set_ylabel('log10(mini-batch loss)')
ax.set_title('6-Layer Deep Network Training Loss')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/plots/03_deep_training_loss.png', dpi=110)
plt.close()
print('Saved outputs/plots/03_deep_training_loss.png')

# Plots 5-6: weight grad ratios and update ratios

# These read parameters_deep and ud_log, not layer.out, so ordering does not matter

print('\nWeight gradient ratios:')
plot_weight_grad_ratios(
    parameters_deep,
    save_path='outputs/plots/03_weight_grad_ratios.png',
)
print('Saved outputs/plots/03_weight_grad_ratios.png')

plot_update_ratios(
    ud_log,
    parameters_deep,
    save_path='outputs/plots/03_update_ratios.png',
)
print('Saved outputs/plots/03_update_ratios.png')

# Generated names file
today  = datetime.date.today().isoformat()
header = (
    f'Model: 6-layer deep MLP with BatchNorm\n'
    f'Train NLL: {train_nll_deep:.4f}  |  Val NLL: {val_nll_deep:.4f}  |  Parameters: {n_params_deep}\n'
    f'Generated: {today}\n'
)
with open('outputs/generated_names/03_deep_network_generated.txt', 'w') as f:
    f.write(header + '\n')
    for name in names:
        f.write(name + '\n')
print('Saved outputs/generated_names/03_deep_network_generated.txt')

print('\nDone')
