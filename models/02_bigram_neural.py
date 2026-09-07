import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import os

# ------------- Data loading -------------

words = open('data/names.txt', 'r').read().splitlines()
print(f'Loaded {len(words)} names')

# ------------- Build vocabulary -------------

chars = sorted(list(set(''.join(words))))
stoi  = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(itos)   # 27


# ---- Build training set: all bigrams from all names as (input, target) pairs -----

xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f'Training examples: {num}')

# One-hot encode inputs
xenc = F.one_hot(xs, num_classes=vocab_size).float()   # (num, 27)

# ---------------------------------------------------------------------------
# Initialize the network: a single 27x27 weight matrix
# W[i, j] will learn the log-count of character j following character i
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)


# ---- Training loop: gradient descent on the full dataset ----

losses = []

for step in range(400):
    # Forward pass
    logits = xenc @ W                              # (num, 27)
    loss   = F.cross_entropy(logits, ys)
    loss   = loss + 0.01 * (W**2).mean()           # L2 regularization = smoothing

    # Backward pass
    W.grad = None
    loss.backward()

    # Update
    W.data += -50 * W.grad

    losses.append(loss.item())
    if step % 50 == 0:
        print(f'step {step:4d}: loss {loss.item():.4f}')

nll = losses[-1]
print(f'\nFinal NLL loss (neural bigram): {nll:.4f}')


# ---- Plot training loss curve and save ----

os.makedirs('outputs/plots', exist_ok=True)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('gradient step')
plt.ylabel('loss (NLL + L2)')
plt.title('Neural Bigram: training loss')
plt.tight_layout()

plot_path = 'outputs/plots/02_bigram_neural_loss.png'
plt.savefig(plot_path, dpi=150)
plt.close()
print(f'Plot saved to {plot_path}')


# ---- Sample names from the trained network ----

g     = torch.Generator().manual_seed(2147483647)
names = []

for _ in range(20):
    out = []
    ix  = 0
    while True:
        xenc_step = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits    = xenc_step @ W
        probs     = torch.softmax(logits, dim=1)
        ix        = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    names.append(''.join(out))


# ------------- Save outputs -------------

os.makedirs('outputs/generated', exist_ok=True)

out_path = 'outputs/generated/02_bigram_neural_generated.txt'
with open(out_path, 'w') as f:
    f.write(f'Model : Bigram neural network (gradient descent)\n')
    f.write(f'NLL   : {nll:.4f}\n')
    f.write(f'Date  : {datetime.date.today()}\n')
    f.write('-' * 30 + '\n')
    for name in names:
        f.write(name + '\n')

print(f'Generated names saved to {out_path}')