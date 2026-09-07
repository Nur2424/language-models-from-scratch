
import torch
import matplotlib.pyplot as plt
import datetime
import os


# ------------- Data loading ----------------

words = open('data/names.txt', 'r').read().splitlines()
print(f'Loaded {len(words)} names')


# ------------- Build vocabulary -------------

chars = sorted(list(set(''.join(words))))
stoi  = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(itos)   # 27


# ------------- Build bigram counts -------------

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1


# ------------- Plot and save the bigram count heatmap -------------

os.makedirs('outputs/plots', exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 16))
ax.imshow(N, cmap='Blues')
for i in range(vocab_size):
    for j in range(vocab_size):
        chstr = itos[i] + itos[j]
        ax.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=7)
        ax.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=7)
ax.set_title('Bigram Character Counts (names.txt)', fontsize=16, pad=14)
ax.axis('off')
plt.tight_layout()

plot_path = 'outputs/plots/01_bigram_counting_heatmap.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved to {plot_path}')


# ------------- Normalize +1 smoothing so no probability is exactly 0 -------------

P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)   # each row sums to 1


# ------------- Compute NLL loss on the training data -------------

log_likelihood = 0.0
n_bigrams      = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        log_likelihood += P[stoi[ch1], stoi[ch2]].log().item()
        n_bigrams      += 1

nll = -log_likelihood / n_bigrams
print(f'NLL loss (counting model, +1 smoothing): {nll:.4f}')


# ------------- Sample names -------------

g     = torch.Generator().manual_seed(2147483647)
names = []

for _ in range(20):
    out = []
    ix  = 0
    while True:
        ix  = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    names.append(''.join(out))


# ------------- Save outputs -------------

os.makedirs('outputs/generated', exist_ok=True)

out_path = 'outputs/generated/01_bigram_counting_generated.txt'
with open(out_path, 'w') as f:
    f.write(f'Model : Bigram counting (statistical)\n')
    f.write(f'NLL   : {nll:.4f}\n')
    f.write(f'Date  : {datetime.date.today()}\n')
    f.write('-' * 30 + '\n')
    for name in names:
        f.write(name + '\n')

print(f'Generated names saved to {out_path}')