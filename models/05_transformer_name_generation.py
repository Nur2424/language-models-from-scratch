# 05  TRANSFORMER CHARACTER-LEVEL LANGUAGE MODEL
# GPT-style causal transformer applied to character-level name generation.
# After "Attention Is All You Need" (Vaswani et al. 2017) and Karpathy's makemore lecture 5.
#
# Produces:
#   outputs/plots/05_training.png
#   outputs/plots/05_grid_olivia.png
#   outputs/plots/05_grid_anna.png
#   outputs/plots/05_grid_christopher.png
#   outputs/plots/05_grid_mia.png
#   outputs/plots/05_hero_heads.png
#   outputs/plots/05_model_comparison.png
#   outputs/generated_names/05_transformer_generated.txt
#
# Run from repo root:
#   python models/05_transformer_name_generation.py
 
import sys
import pathlib
import random
from collections import defaultdict
 
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.diagnostics import plot_update_ratios
 
# ---- Hyperparameters ----
 
block_size    = 16    # longest name is 15 chars; '.' + name fits in 16 tokens
n_embd        = 64    # residual stream width
n_head        = 4     # attention heads; head_size = n_embd // n_head = 16
n_layer       = 6     # transformer blocks stacked
dropout       = 0.2   # applied inside attention and feedforward
learning_rate = 1e-3  # AdamW, no schedule
batch_size    = 64
max_iters     = 5000
eval_interval = 250   # evaluate + log update ratios every N steps
eval_iters    = 100   # mini-batches averaged for the cheap loss estimate
 
# ---- Data loading ----
 
with open("../language-models-from-scratch/data/names.txt", 'r') as file:
    words = file.read().splitlines()
 
chars = sorted(list(set(''.join(words))))
stoi  = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(itos)   # 27
 
print(f'{len(words)} words  |  vocab_size={vocab_size}')
 
def build_sequences(word_list):
    """Convert a list of names into padded (X, Y) tensors.
 
    For a name like 'emma':
      ids = [5, 13, 13, 1]
      x   = [0, 5, 13, 13, 1, 0, 0, ...] (length block_size, padded with 0/'.')
      y   = [5, 13, 13, 1, 0, -1, -1, ...] (padded with -1 so loss ignores pads)
 
    ignore_index=-1 in cross_entropy skips the padded positions, so the loss
    measures only real next-character predictions, not padding noise.
    """
    X, Y = [], []
    for w in word_list:
        ids = [stoi[c] for c in w]
        x   = [0] + ids                             # '.' start token + name
        y   = ids + [0]                             # name + '.' end token
        x   = x + [0]  * (block_size - len(x))      # pad inputs with '.' (index 0)
        y   = y + [-1] * (block_size - len(y))      # pad targets with -1 (ignored)
        X.append(x[:block_size])
        Y.append(y[:block_size])
    return torch.tensor(X), torch.tensor(Y)
 
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
 
Xtr,  Ytr  = build_sequences(words[:n1])
Xdev, Ydev = build_sequences(words[n1:n2])
Xte,  Yte  = build_sequences(words[n2:])
print(f'train {tuple(Xtr.shape)}  dev {tuple(Xdev.shape)}  test {tuple(Xte.shape)}')
# expected: (25626, 16)  (3203, 16)  (3204, 16)
 
device = 'cuda' if torch.cuda.is_available() else (
         'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f'device: {device}')
 
# ---- Model architecture ----
 
class Head(nn.Module):
    """One causal self-attention head.
 
    forward(x)                   -> out          (training / generation)
    forward(x, return_attn=True) -> out, att     (inspection; att is (B, T, T))
    """
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a trained parameter but must move to GPU with the module,
        # so register_buffer is the right API (parameter() would include it in
        # optimizer updates, plain tensor assignment would not follow .to(device))
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.attn_drop = nn.Dropout(dropout)
 
    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        k = self.key(x)                                     # (B, T, head_size)
        q = self.query(x)                                   # (B, T, head_size)
        v = self.value(x)                                   # (B, T, head_size)
 
        head_size = k.shape[-1]
        # Scale by head_size**-0.5 to keep variance of q@k^T near 1 regardless
        # of head_size. Without scaling, large head_sizes push softmax into
        # near-one-hot regions where gradients vanish.
        # Formal argument: Var(q_i @ k_i^T) = head_size * Var(q) * Var(k).
        # Dividing by sqrt(head_size) brings the variance back to ~1.
        att = q @ k.transpose(-2, -1) * head_size ** -0.5  # (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
 
        out = att @ v                                       # (B, T, head_size)
        if return_attn:
            return out, att
        return out
 
 
class MultiHeadAttention(nn.Module):
    """n_head causal heads in parallel; outputs concatenated and projected.
 
    The n_embd % n_head == 0 assertion ensures the concatenated output
    (n_head * head_size) equals exactly n_embd, so it fits back into the
    residual stream without a dimension change.
 
    The proj (projection) layer mixes information ACROSS heads after
    concatenation. Without it each head's contribution would be independent;
    proj lets the model learn which head combinations matter most.
 
    return_attn=True returns the per-head attention stacked as (B, n_head, T, T),
    used in the attention visualization section.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        # assertion ensures concatenated output fits back into n_embd residual stream
        assert n_embd % n_head == 0, 'n_embd must divide evenly into n_head heads'
        head_size = n_embd // n_head
        self.heads     = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj      = nn.Linear(n_embd, n_embd)   # cross-head mixing, projects back into stream
        self.resid_drop = nn.Dropout(dropout)
 
    def forward(self, x, return_attn=False):
        if return_attn:
            outs, atts = [], []
            for h in self.heads:
                o, a = h(x, return_attn=True)
                outs.append(o)
                atts.append(a)                            # each a: (B, T, T)
            out = torch.cat(outs, dim=-1)                 # (B, T, n_embd)
            out = self.resid_drop(self.proj(out))
            return out, torch.stack(atts, dim=1)          # attn: (B, n_head, T, T)
 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.resid_drop(self.proj(out))
        return out
 
 
class FeedForward(nn.Module):
    """Per-position MLP: expand to 4x width, ReLU, project back to n_embd.
 
    Attention is purely communication (mixing across T). FeedForward is
    purely computation (each position thinks independently about what it heard).
    The 4x expansion gives the model capacity to represent complex functions.
    """
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
 
    def forward(self, x):
        return self.net(x)   # each position processed independently, no T mixing
 
 
class Block(nn.Module):
    """One transformer block: Pre-LN attention + residual, Pre-LN FFN + residual.
 
    Pre-LN (LayerNorm before the sublayer) vs Post-LN (after):
    Pre-LN keeps the residual stream unnormalized, so gradients flow cleanly
    through the skip connection all the way to the embedding layer. Post-LN
    normalizes the residual, which can shrink gradients to early layers and
    requires careful warm-up. Pre-LN is what GPT-2 uses.
 
    return_attn=False is the default for training and generation (fast path).
    return_attn=True is used only during attention visualization.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.sa   = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
 
    def forward(self, x, return_attn=False):
        if not return_attn:
            # fast path: no attention storage
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x
        attended, att = self.sa(self.ln1(x), return_attn=True)
        x = x + attended
        x = x + self.ffwd(self.ln2(x))
        return x, att
 
 
class NameTransformer(nn.Module):
    """GPT-style transformer for character-level name generation.
 
    Architecture:
      token_emb (B,T) -> (B,T,n_embd)
      pos_emb   adds positional signal, same shape
      n_layer Blocks (causal self-attention + FFN)
      ln_f + head -> logits (B,T,vocab_size)
 
    Loss uses ignore_index=-1 to skip the padding targets written by
    build_sequences, so only real next-character predictions count.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb  = nn.Embedding(vocab_size, n_embd)
        self.pos_emb    = nn.Embedding(block_size, n_embd)
        self.blocks     = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
 
    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, a = block(x, return_attn=True)
                attn_maps.append(a)           # (B, n_head, T, T) per layer
            else:
                x = block(x)
        logits = self.head(self.ln_f(x))      # (B, T, vocab_size)
 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,   # skip -1 padding; real positions only
            )
 
        if return_attn:
            return logits, loss, attn_maps
        return logits, loss
 
# ---- Training setup ----
 
torch.manual_seed(1337)
model = NameTransformer(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
model = model.to(device)
 
# Parameter count breakdown
def _pc(mod): return sum(p.numel() for p in mod.parameters())
emb_params   = _pc(model.token_emb) + _pc(model.pos_emb)
blk_params   = _pc(model.blocks)
final_params = _pc(model.ln_f) + _pc(model.head)
total_params = _pc(model)
print(f'\n{"embeddings (token + pos)":<32}{emb_params:>8,}  ({100*emb_params/total_params:.1f}%)')
print(f'{"6 blocks":<32}{blk_params:>8,}  ({100*blk_params/total_params:.1f}%)')
print(f'{"  per block":<32}{blk_params//n_layer:>8,}')
print(f'{"final LN + head":<32}{final_params:>8,}  ({100*final_params/total_params:.1f}%)')
print(f'{"-"*46}')
print(f'{"total":<32}{total_params:>8,}')
 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
 
def get_batch(split):
    X, Y = (Xtr, Ytr) if split == 'train' else (Xdev, Ydev)
    ix   = torch.randint(len(X), (batch_size,))
    return X[ix].to(device), Y[ix].to(device)
 
@torch.no_grad()
def estimate_loss():
    """Cheap sampled NLL estimate for the training curve (eval_iters batches)."""
    out = {}
    model.eval()
    for split in ['train', 'dev']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
 
@torch.no_grad()
def full_loss(X, Y, chunk=2048):
    """Exact per-token NLL over a whole split (no approximation from sampling)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(X), chunk):
        xb  = X[i:i+chunk].to(device)
        yb  = Y[i:i+chunk].to(device)
        logits, _ = model(xb)
        # reduction='sum' so we can accumulate numerator and denominator separately;
        # dividing sum/count gives exact per-token NLL, not batch-size-weighted average
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), yb.view(-1),
            ignore_index=-1, reduction='sum'
        )
        total += loss.item()
        count += (yb != -1).sum().item()
    model.train()
    return total / count
 
def param_group(name):
    """Bucket parameter by name substring for the update-ratio plot."""
    if any(k in name for k in ['key', 'query', 'value', 'proj']): return 'attention'
    if 'ffwd' in name:                                             return 'feedforward'
    if '.ln' in name or name.startswith('ln'):                    return 'layernorm'
    if 'emb' in name or 'head' in name:                           return 'embeddings'
    return 'other'
 
# ---- Training loop ----
 
torch.manual_seed(1337)   # re-seed after all demo/setup cells; makes the run reproducible
 
train_hist, dev_hist, hist_steps = [], [], []
ud = []   # ud[i] = {'step': int, 'ratios': [(param_name, log10_ratio), ...]}
 
print(f'\nTraining {max_iters} steps on {device}...')
model.train()
for step in range(max_iters):
    # evaluate at the start of every eval_interval and at the final step
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        train_hist.append(losses['train'])
        dev_hist.append(losses['dev'])
        hist_steps.append(step)
        print(f'  step {step:4d} | train {losses["train"]:.4f} | dev {losses["dev"]:.4f}')
 
    xb, yb   = get_batch('train')
    _, loss   = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
 
    # log update ratios AFTER backward, BEFORE optimizer step so p.data is pre-step weights
    if step % eval_interval == 0 or step == max_iters - 1:
        with torch.no_grad():
            ratios = [
                (n, (learning_rate * p.grad.std() / (p.data.std() + 1e-12)).log10().item())
                for n, p in model.named_parameters() if p.grad is not None
            ]
        ud.append({'step': step, 'ratios': ratios})
 
    optimizer.step()
 
# ---- Evaluation ----
 
tr = full_loss(Xtr,  Ytr)
vl = full_loss(Xdev, Ydev)
te = full_loss(Xte,  Yte)
print(f'\nFull-split NLL   train {tr:.4f} | dev {vl:.4f} | test {te:.4f}')
print(f'Train/dev gap    {vl - tr:.4f}')
print(f'Best sampled dev {min(dev_hist):.4f} at step {hist_steps[dev_hist.index(min(dev_hist))]}')
 
# ---- Attention visualization helpers ----
 
@torch.no_grad()
def attention_maps(name):
    """Run the trained model on a single name and return per-layer attention."""
    model.eval()
    seq = [0] + [stoi[c] for c in name]         # '.' start token + name characters
    idx = torch.tensor([seq], device=device)
    _, _, maps = model(idx, return_attn=True)    # maps: list of n_layer (1, n_head, T, T)
    return [m[0].cpu().numpy() for m in maps], ['.'] + list(name)
 
 
def draw_head(ax, A, labels, title, annotate=True):
    """Render one attention head as a heatmap with optional numeric annotations."""
    ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=10)
    if annotate:
        for r in range(len(labels)):
            for c in range(len(labels)):
                if A[r, c] > 0.005:
                    ax.text(c, r, f'{A[r, c]:.2f}', ha='center', va='center',
                            color='white' if A[r, c] > 0.5 else '#333', fontsize=8)
 
# ---- Generate names ----
 
@torch.no_grad()
def generate_name(temperature=1.0):
    """Autoregressive generation: start from '.' (index 0), sample until '.' or block_size."""
    idx = torch.tensor([[0]], device=device)
    out = []
    for _ in range(block_size):
        logits, _ = model(idx[:, -block_size:])
        logits     = logits[:, -1, :] / temperature   # last position only; temperature scaling
        probs      = F.softmax(logits, dim=-1)
        nxt        = torch.multinomial(probs, num_samples=1)
        if nxt.item() == 0:    # '.' signals end of name
            break
        out.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    return ''.join(itos[i] for i in out)
 
# ---- Save outputs ----
 
pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)
pathlib.Path('outputs/generated_names').mkdir(parents=True, exist_ok=True)
 
print('\nGenerating plots...')
 
# ---- Plot 1: Training loss + update ratios ----
# Left: loss curve. Right: log10 update ratio per parameter group.
# Skip step 0 in the update plot: LayerNorm initialises to all-ones weights,
# so std(weight) == 0 at step 0 and the ratio is undefined (or -inf).
 
ud_plot  = [e for e in ud if e['step'] > 0]
ud_steps = [e['step'] for e in ud_plot]
series   = defaultdict(list)
for e in ud_plot:
    bucket = defaultdict(list)
    for name, val in e['ratios']:
        bucket[param_group(name)].append(val)
    for g in ['embeddings', 'attention', 'feedforward', 'layernorm']:
        if bucket[g]:
            series[g].append(sum(bucket[g]) / len(bucket[g]))
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
 
ax1.plot(hist_steps, train_hist, label='train', color='steelblue',  linewidth=1.8)
ax1.plot(hist_steps, dev_hist,   label='dev',   color='darkorange', linewidth=1.8)
ax1.set_xlabel('iteration')
ax1.set_ylabel('NLL loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(alpha=0.3)
 
group_colors = {
    'embeddings': 'steelblue',
    'attention':  '#7c3aed',
    'feedforward':'seagreen',
    'layernorm':  'darkorange',
}
for g in ['embeddings', 'attention', 'feedforward', 'layernorm']:
    if series[g]:
        ax2.plot(ud_steps[:len(series[g])], series[g],
                 label=g, linewidth=1.6, color=group_colors[g])
ax2.axhline(y=-3, color='#555555', linestyle='--', linewidth=1.2, label='target (-3)')
ax2.set_xlabel('iteration')
ax2.set_ylabel('log10(update ratio)')
ax2.set_title('Update Ratios by Parameter Group')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
 
fig.suptitle('Transformer Training -- Names Dataset', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('outputs/plots/05_training.png', dpi=130)
plt.close(fig)
print('Saved outputs/plots/05_training.png')
 
# ---- Plots 2-5: Attention grids (all 6 layers x 4 heads per name) ----
 
names_to_read = ['olivia', 'anna', 'christopher', 'mia']
for name in names_to_read:
    maps, labels = attention_maps(name)
    fig, axes = plt.subplots(6, 4, figsize=(13, 18))
    for L in range(6):
        for H in range(4):
            draw_head(axes[L, H], maps[L][H], labels, f'L{L} H{H}', annotate=False)
    fig.suptitle(f'"{name}": attention across all 6 layers x 4 heads', fontsize=14, y=0.997)
    plt.tight_layout()
    plt.savefig(f'outputs/plots/05_grid_{name}.png', dpi=90)
    plt.close(fig)
    print(f'Saved outputs/plots/05_grid_{name}.png')
 
# ---- Plot 6: Hero heads (3 annotated heatmaps) ----
# Three interpretable heads identified from inspection:
#   L1 H2 on 'christopher': previous-character pattern (diagonal shift-by-1)
#   L5 H0 on 'christopher': start-token anchor (column 0 dominates)
#   L5 H3 on 'olivia':      last-vowel head (late-layer vowel tracking)
 
heroes = [
    ('christopher', 1, 2, 'Layer 1, Head 2\nprevious-character'),
    ('christopher', 5, 0, 'Layer 5, Head 0\nstart-token anchor'),
    ('olivia',      5, 3, 'Layer 5, Head 3\nlast-vowel'),
]
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, (nm, L, H, title) in zip(axes, heroes):
    maps, labels = attention_maps(nm)
    draw_head(ax, maps[L][H], labels, f'"{nm}"   {title}', annotate=True)
plt.tight_layout()
plt.savefig('outputs/plots/05_hero_heads.png', dpi=120)
plt.close(fig)
print('Saved outputs/plots/05_hero_heads.png')
 
# ---- Plot 7: Model comparison bar chart ----
# Reference dev NLL values across the makemore series. Bigram, MLP, BatchNorm
# MLP, and WaveNet values are from the corresponding scripts. Transformer
# uses the exact full-split dev NLL measured above.
 
comparison = {
    'Bigram':          (2.45,  27),
    'MLP':             (2.16,  12_097),
    'BatchNorm MLP':   (2.10,  12_097),
    'WaveNet':         (2.24,  22_397),
    'Transformer':     (round(vl, 4), total_params),
}
# sort worst (highest NLL) to best (lowest NLL) so the best sits at the bottom
sorted_models = sorted(comparison.items(), key=lambda kv: kv[1][0], reverse=True)
 
model_names   = [k for k, v in sorted_models]
model_vals    = [v[0] for k, v in sorted_models]
model_params  = [v[1] for k, v in sorted_models]
 
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ['crimson', 'darkorange', '#999999', 'seagreen', 'steelblue']
bars = ax.barh(model_names, model_vals, color=bar_colors[:len(model_names)])
# label each bar with dev NLL and parameter count
for bar, val, params in zip(bars, model_vals, model_params):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}  ({params:,} params)',
            va='center', ha='left', fontsize=9)
ax.set_xlabel('Dev NLL')
ax.set_title('Dev NLL Across the makemore Series')
ax.set_xlim(1.5, 3.0)
ax.invert_yaxis()   # worst model at the top
plt.tight_layout()
plt.savefig('outputs/plots/05_model_comparison.png', dpi=130)
plt.close(fig)
print('Saved outputs/plots/05_model_comparison.png')
 
# ---- Generated names ----
 
model.eval()
torch.manual_seed(42)
names_out = [generate_name(temperature=1.0) for _ in range(30)]
 
out_path = pathlib.Path('outputs/generated_names/05_transformer_generated.txt')
with open(out_path, 'w') as f:
    f.write(f'Model    : Transformer character-level language model (Vaswani et al. 2017)\n')
    f.write(f'Train NLL: {tr:.4f}  |  Dev NLL: {vl:.4f}  |  Parameters: {total_params:,}\n')
    f.write('--------------------------------\n')
    for name in names_out:
        f.write(name + '\n')
print(f'Saved {out_path}')
 
print('\nSample names:')
for name in names_out[:10]:
    print(f'  {name}')
 
print('\nDone')