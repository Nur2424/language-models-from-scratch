# 02 MLP CHARACTER-LEVEL LANGUAGE MODEL
# After Bengio et al. 2003 "A Neural Probabilistic Language Model"


# Trains a 3-character-context MLP on names.txt and produces:
#   outputs/plots/02_lr_search.png
#   outputs/plots/02_embedding_scatter.png
#   outputs/plots/02_training_loss.png
#   outputs/generated/02_mlp_generated.txt
#
# Run from repo root:
#   python models/02_mlp_language_model.py


import random
import datetime
import pathlib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------ Hyperparameters ----------

block_size      = 3        # context length predict next char from 3 preceding
emb_dim_2d      = 2        # embedding dim for LR search + scatter plotted directly
n_hidden_2d     = 100      # hidden units for the intermediate 2D model
emb_dim         = 10       # embedding dim for the final model
n_hidden        = 200      # hidden units for the final model
batch_size      = 32
max_iters_mid   = 110_000  # steps for intermediate model (100k high + 10k low LR)
max_iters       = 200_000  # steps for final model (100k high + 100k low LR)
lr_high         = 0.1
lr_low          = 0.01
lr_decay_at_mid = 100_000
lr_decay_at     = 100_000
INIT_SEED       = 2147483647   # fixed seed for all parameter initializations

# ---- Data loading ------------

with open("../language-models-from-scratch/data/names.txt", "r", encoding="utf-8") as file:
    words = file.read().splitlines()
chars = sorted(set(''.join(words)))
stoi  = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)   # 27: 26 letters + '.' as 0

print(f'Words: {len(words)}  Vocabulary: {vocab_size} symbols')

# ---- Dataset builder ------------

def build_dataset(word_list):
    """Build (context, target) pairs using a sliding window of block_size"""
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)                                            
            context = context[1:] + [ix]  # slide drop oldest char, append newest
    return torch.tensor(X), torch.tensor(Y)

# 80/10/10 split random.seed(42) fixes the shuffle for reproducibility
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte,  Yte  = build_dataset(words[n2:])
print(f'Train: {Xtr.shape}  Dev: {Xdev.shape}  Test: {Xte.shape}')

# ---- LR search (one-time calibration) ------------
# Run once on a short-lived model.  Read the chart, pick the rate (~0.1 from
# the sweet spot near log10(lr) = -1), then discard this model state
# The LR search uses the intermediate architecture: C(27,2), hidden=100

g_lr  = torch.Generator().manual_seed(INIT_SEED)
C_lr  = torch.randn((vocab_size, emb_dim_2d),               generator=g_lr, requires_grad=True)
W1_lr = torch.randn((block_size * emb_dim_2d, n_hidden_2d), generator=g_lr, requires_grad=True)
b1_lr = torch.randn(n_hidden_2d,                            generator=g_lr, requires_grad=True)
W2_lr = torch.randn((n_hidden_2d, vocab_size),              generator=g_lr, requires_grad=True)
b2_lr = torch.randn(vocab_size,                             generator=g_lr, requires_grad=True)
params_lr = [C_lr, W1_lr, b1_lr, W2_lr, b2_lr]

lrexp_vals = torch.linspace(-3, 0, 1000)
lrs        = 10 ** lrexp_vals
lri, lossi_lr = [], []

for i in range(1000):
    ix  = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C_lr[Xtr[ix]]
    # view() reinterprets tensor shape via metadata zero copy, no allocation
    h      = torch.tanh(emb.view(-1, block_size * emb_dim_2d) @ W1_lr + b1_lr)
    logits = h @ W2_lr + b2_lr
    # F.cross_entropy works directly on logits numerically stable, no manual softmax
    loss   = F.cross_entropy(logits, Ytr[ix])
    for p in params_lr:
        p.grad = None  
    loss.backward()
    for p in params_lr:
        p.data += -lrs[i].item() * p.grad
    lri.append(lrexp_vals[i].item())
    lossi_lr.append(loss.item())

# Discard LR search model its state is not needed further
del params_lr, C_lr, W1_lr, b1_lr, W2_lr, b2_lr
print('LR search done.')

# ---- Intermediate model train C(27,2) for embedding scatter ------------
# Two-dimensional embeddings can be plotted directly.  The geometry the model
# finds (vowels clustering, '.' isolated) emerges purely from which characters
# co-occur across 32k names the model is never told about phonetics

g_mid = torch.Generator().manual_seed(INIT_SEED)
C2    = torch.randn((vocab_size, emb_dim_2d),               generator=g_mid, requires_grad=True)
W1_2  = torch.randn((block_size * emb_dim_2d, n_hidden_2d), generator=g_mid, requires_grad=True)
b1_2  = torch.randn(n_hidden_2d,                            generator=g_mid, requires_grad=True)
W2_2  = torch.randn((n_hidden_2d, vocab_size),              generator=g_mid, requires_grad=True)
b2_2  = torch.randn(vocab_size,                             generator=g_mid, requires_grad=True)
params_mid = [C2, W1_2, b1_2, W2_2, b2_2]

for i in range(max_iters_mid):
    ix  = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C2[Xtr[ix]]
    h      = torch.tanh(emb.view(-1, block_size * emb_dim_2d) @ W1_2 + b1_2)
    logits = h @ W2_2 + b2_2
    loss   = F.cross_entropy(logits, Ytr[ix])
    for p in params_mid:
        p.grad = None
    loss.backward()
    # Two-phase LR, explicit if-branch and no scheduler object
    lr = lr_high if i < lr_decay_at_mid else lr_low
    for p in params_mid:
        p.data += -lr * p.grad

print('Intermediate 2D model trained.')

# ---- Model initialization (final model) ------------
# C is the embedding table: shape (vocab_size, emb_dim).
# C[ix] == one_hot(ix) @ C embedding lookup IS a linear layer accessed by
# index instead of matmul, which is both faster and more memory efficient 

g_final = torch.Generator().manual_seed(INIT_SEED)
C  = torch.randn((vocab_size, emb_dim),              generator=g_final, requires_grad=True)
W1 = torch.randn((block_size * emb_dim, n_hidden),   generator=g_final, requires_grad=True)
b1 = torch.randn(n_hidden,                           generator=g_final, requires_grad=True)
W2 = torch.randn((n_hidden, vocab_size),             generator=g_final, requires_grad=True)
b2 = torch.randn(vocab_size,                         generator=g_final, requires_grad=True)
parameters = [C, W1, b1, W2, b2]

n_params = sum(p.nelement() for p in parameters)
print(f'Final model parameters: {n_params}')   # 11897

# ------------ Training ------------

stepi = []
lossi = []

for i in range(max_iters):
    ix  = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C[Xtr[ix]]
    # view() is free: (N, block_size, emb_dim) => (N, block_size*emb_dim), no copy
    h      = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
    logits = h @ W2 + b2
    loss   = F.cross_entropy(logits, Ytr[ix])

    for p in parameters:
        p.grad = None   # None frees the gradient tensor faster than zeroing
    loss.backward()

    # Two-phase LR: explicit if-branch matching the notebook exactly
    lr = lr_high if i < lr_decay_at else lr_low
    for p in parameters:
        p.data += -lr * p.grad

    stepi.append(i)
    lossi.append(loss.log10().item())

    if i % 20_000 == 0:
        print(f'  step {i:6d}  log10(loss) {lossi[-1]:.4f}')

print('Training complete.')

# ------------ Evaluation ------------
# Full-split NLL: one forward pass over each entire split with no gradients.
# The mini-batch loss during training is noisy; these numbers are the honest ones.

@torch.no_grad()
def evaluate(X, Y):
    emb    = C[X]
    h      = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()

train_nll = evaluate(Xtr,  Ytr)
dev_nll   = evaluate(Xdev, Ydev)
test_nll  = evaluate(Xte,  Yte)
print(f'\nFull-split NLL:  train {train_nll:.4f}  |  dev {dev_nll:.4f}  |  test {test_nll:.4f}')

# ------------ Generate names ------------

g_gen = torch.Generator().manual_seed(INIT_SEED)
names = []

for _ in range(20):
    out     = []
    context = [0] * block_size
    while True:
        emb_s    = C[torch.tensor([context])]
        h_s      = torch.tanh(emb_s.view(1, -1) @ W1 + b1)
        logits_s = h_s @ W2 + b2
        probs    = F.softmax(logits_s, dim=1)
        ix       = torch.multinomial(probs, num_samples=1, generator=g_gen).item()
        context  = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    # strip trailing '.' token before recording
    names.append(''.join(itos[k] for k in out[:-1]))

# ------------ Save outputs ------------

pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)
pathlib.Path('outputs/generated_names').mkdir(parents=True, exist_ok=True)

# Plot 1: LR search
fig, ax = plt.subplots()
ax.plot(lri, lossi_lr)
ax.set_xlabel('log10(learning rate)')
ax.set_ylabel('mini-batch loss')
ax.set_title('Learning Rate Search')
plt.tight_layout()
plt.savefig('outputs/plots/02_lr_search.png', dpi=110)
plt.close()
print('Saved outputs/plots/02_lr_search.png')

# Plot 2: 2D character embedding scatter
C2_data = C2.data   # detach for plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(C2_data[:, 0], C2_data[:, 1], s=200)
for i in range(vocab_size):
    ax.text(C2_data[i, 0].item(), C2_data[i, 1].item(), itos[i],
            ha='center', va='center', color='white')
ax.set_xlabel('embedding dim 1')
ax.set_ylabel('embedding dim 2')
ax.set_title('Learned Character Embeddings (2D)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/02_embedding_scatter.png', dpi=110)
plt.close()
print('Saved outputs/plots/02_embedding_scatter.png')

# Plot 3: Training loss curve
fig, ax = plt.subplots()
ax.plot(stepi, lossi, alpha=0.4)
ax.axvline(x=lr_decay_at, color='red', linestyle='--', linewidth=1,
           label=f'LR decay  {lr_high} -> {lr_low}')
ax.set_xlabel('step')
ax.set_ylabel('log10(mini-batch loss)')
ax.set_title('MLP Training Loss (200k steps)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/plots/02_training_loss.png', dpi=110)
plt.close()
print('Saved outputs/plots/02_training_loss.png')

# Generated names file
today  = datetime.date.today().isoformat()
header = (
    f'Model: MLP character-level language model (Bengio et al. 2003)\n'
    f'Train NLL: {train_nll:.4f}  |  Dev NLL: {dev_nll:.4f}  |  Parameters: {n_params}\n'
    f'Generated: {today}\n'
)
with open('outputs/generated_names/02_mlp_generated.txt', 'w') as f:
    f.write(header + '\n')
    for name in names:
        f.write(name + '\n')
print('Saved outputs/generated/02_mlp_generated.txt')

print('\nDone.')
