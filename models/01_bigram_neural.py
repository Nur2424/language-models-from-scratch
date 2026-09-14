# 01 BIGRAM NEURAL NETWORK (gradient descent)


# Produces:
#   outputs/plots/01_bigram_neural_loss.png
#   outputs/generated_names/01_bigram_neural_generated.txt

# Run from repo root:
#   python models/01_bigram_neural.py
 
import datetime
import pathlib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
# -------- Hyperparameters --------
 
max_iters  = 400
lr         = 10.0    # lr=50 overshoots on full-batch GD and bounces; 10 is stable
l2_lambda  = 0.01
INIT_SEED  = 2147483647
 
# -------- Data loading --------
 
with open("../language-models-from-scratch/data/names.txt", "r", encoding="utf-8") as file:
    words = file.read().splitlines()
chars = sorted(set(''.join(words)))
stoi  = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos  = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)   # 27
 
print(f'Words: {len(words)}  Vocabulary: {vocab_size} symbols')
 
# -------- Build bigram dataset --------
 
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
 
xs  = torch.tensor(xs)
ys  = torch.tensor(ys)
num = xs.nelement()
print(f'Training bigrams: {num}')
 
# xenc @ W is equivalent to looking up a row of W one-hot matmul selects a row
xenc = F.one_hot(xs, num_classes=vocab_size).float()   # (num, 27)
 
# -------- Model --------
 
g = torch.Generator().manual_seed(INIT_SEED)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
print(f'Parameters: {W.nelement()}')   # 729
 
# -------- Training: full-batch gradient descent --------
 
losses = []
 
for step in range(max_iters):
    logits = xenc @ W
    nll    = F.cross_entropy(logits, ys)
    loss   = nll + l2_lambda * (W ** 2).mean()   # L2 = smoothing
 
    W.grad = None
    loss.backward()
    W.data += -lr * W.grad
 
    losses.append(loss.item())
    if step % 50 == 0:
        print(f'  step {step:4d}  loss {loss.item():.4f}')
 
final_loss = losses[-1]
print(f'\nFinal loss: {final_loss:.4f}')
 
# -------- Generate 20 names --------
 
g_gen = torch.Generator().manual_seed(INIT_SEED)
names = []
 
for _ in range(20):
    out = []
    ix  = 0
    while True:
        xenc_step = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        probs = torch.softmax(xenc_step @ W, dim=1)
        ix    = torch.multinomial(probs, num_samples=1, replacement=True, generator=g_gen).item()
        if ix == 0:
            break
        out.append(itos[ix])
    names.append(''.join(out))
 
# -------- Save outputs --------
 
pathlib.Path('outputs/plots').mkdir(parents=True, exist_ok=True)
pathlib.Path('outputs/generated_names').mkdir(parents=True, exist_ok=True)
 
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses, color='steelblue', linewidth=1.5)
ax.set_xlabel('gradient step')
ax.set_ylabel('loss (NLL + L2)')
ax.set_title('Neural Bigram: Training Loss (full-batch GD, 400 steps)')
plt.tight_layout()
plt.savefig('outputs/plots/01_bigram_neural_loss.png', dpi=150)
plt.close()
print('Saved outputs/plots/01_bigram_neural_loss.png')
 
today  = datetime.date.today().isoformat()
header = (
    f'Model: Bigram neural network (gradient descent)\n'
    f'Loss  : {final_loss:.4f}  |  Parameters: {W.nelement()}\n'
    f'Generated: {today}\n'
)
with open('outputs/generated_names/01_bigram_neural_generated.txt', 'w') as f:
    f.write(header + '\n')
    for name in names:
        f.write(name + '\n')
print('Saved outputs/generated_names/01_bigram_neural_generated.txt')
 
print('\nDone.')