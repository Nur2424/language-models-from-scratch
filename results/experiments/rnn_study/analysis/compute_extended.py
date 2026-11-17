"""Model 6: extended analysis of the verified final runs (D-63). Read-only over the study records and the
one-time test result; writes analysis/extended_summary.json (exclusively). Sections: predictive performance
(per-position dev NLL, test, generalisation), hidden-state dynamics over training, temporal gradients (G2
loss-projected attenuation vs G1, W spectrum), optimisation behaviour (learning curves, gradient norms, dev
rises), seed variability (per-seed table; which findings hold in every seed), estimate-quality caveats, and,
when torch is available, generated samples. No test-split access (the test numbers are read from result.json).
Run from the repository root:
    python3 results/experiments/rnn_study/analysis/compute_extended.py
"""
import glob, json, math, os, statistics as st, sys
import numpy as np

REPO = os.getcwd()
sys.path.insert(0, REPO)  # the repository modules (generation section)
ROOT = os.path.join(REPO, 'results/experiments/rnn_study')
RUNS = os.path.join(ROOT, 'L7_research/runs/rnn')

def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2: return None
    return {"mean": st.fmean(v), "se": st.stdev(v) / math.sqrt(len(v)), "min": min(v), "max": max(v), "n": len(v)}

def load():
    out = []
    for d in sorted(glob.glob(RUNS + '/run_*')):
        pts = {}; prim = {}
        for l in open(d + '/research_diagnostics/points.jsonl'):
            r = json.loads(l); pts[r['point_id']] = r
        for l in open(d + '/research_diagnostics/primary.jsonl'):
            r = json.loads(l); prim[r['point_id']] = r
        out.append({"dir": d, "seed": json.load(open(d + '/config.json'))["seed"], "meta": json.load(open(d + '/metadata.json')),
                    "hist": [json.loads(l) for l in open(d + '/history.jsonl')],
                    "steps": [json.loads(l) for l in open(d + '/diagnostics/steps.jsonl')],
                    "o3": json.load(open(d + '/research_diagnostics/o3.json')), "pts": pts, "prim": prim})
    return out

R = load()
assert len(R) == 10 and all(r['meta']['outcome'] == 'completed' for r in R)
def pid(r, kind=None, k=None, sigma=None):
    return next(p for p, x in r['pts'].items() if (kind is None or x['kind'] == kind) and (k is None or x['completed_epochs'] == k)
                and (sigma is None or x['sigma'] == sigma))
best = lambda r: pid(r, 'epoch_end', k=r['meta']['best_epoch'])
train_points = lambda r: sorted((x['sigma'], p) for p, x in r['pts'].items() if x['kind'] in ('start', 'early', 'epoch_end'))
test = json.load(open(os.path.join(ROOT, 'test_evaluation/result.json')))
out = {"note": "Descriptive summaries of the 10 verified final runs (D-60, D-62); SE = SD/sqrt(10) over seeds."}

# ---------------------------------------------------------------- 1. predictive performance
pos = {}
for r in R:
    by_t = r['hist'][r['meta']['best_epoch']]['position_dev_nll']['by_t']
    for t, (s, n) in by_t.items():
        pos.setdefault(t, {"nll": [], "targets": n})["nll"].append(s / n)
init_pos = {t: s / n for t, (s, n) in R[0]['hist'][0]['position_dev_nll']['by_t'].items()}
# EOS share of dev targets per position is not stored; report per-position NLL and target counts
out["predictive"] = {
    "best_dev_nll": ms([min(x['dev_nll'] for x in r['hist'][1:]) for r in R]),
    "test_nll": ms([v['test_nll'] for v in test['per_seed'].values()]),
    "dev_minus_test_per_seed": ms([test['per_seed'][str(r['seed'])]['test_nll'] - min(x['dev_nll'] for x in r['hist'][1:]) for r in R]),
    "perplexity_test": math.exp(test['test_nll_mean']),
    "train_nll_at_best": ms([r['hist'][r['meta']['best_epoch']]['train_nll'] for r in R]),
    "generalisation_gap_at_best": ms([r['hist'][r['meta']['best_epoch']]['dev_nll'] - r['hist'][r['meta']['best_epoch']]['train_nll'] for r in R]),
    "dev_nll_by_position_at_best": {t: {"nll": ms(v["nll"]), "targets": v["targets"], "nll_at_init_seed0": init_pos[t]}
                                    for t, v in sorted(pos.items(), key=lambda kv: int(kv[0].rstrip('+')))},
    "uniform_baseline_ln27": math.log(27)}

# ---------------------------------------------------------------- 2. hidden-state dynamics over training
traj = {}
for r in R:
    for s, p in train_points(r):
        rows = r['prim'][p]['results']['H1']['per_t_weighted']
        late = [x for x in rows if x['t'] >= 3]
        w = [x['names'] for x in late] if 'names' in late[0] else [1] * len(late)
        c = traj.setdefault(s, {"saturation_t>=3": [], "mean_sq_dtanh_t>=3": [], "rms_t>=3": [], "epoch": r['pts'][p]['completed_epochs']})
        c["saturation_t>=3"].append(st.fmean(x['saturation'] for x in late))
        c["mean_sq_dtanh_t>=3"].append(st.fmean(x['mean_sq_dtanh'] for x in late))
        c["rms_t>=3"].append(st.fmean(x['rms'] for x in late))
strat = {}
for r in R:
    for x in r['prim'][best(r)]['results']['H1']['per_t_stratum']:
        strat.setdefault(f"{x['stratum']}:{x['t']}", []).append(x['saturation'])
out["hidden_state"] = {"over_training_mean_over_positions_t>=3": {s: {"epoch": c["epoch"], **{k: ms(v) for k, v in c.items() if k != "epoch"}}
                                                               for s, c in sorted(traj.items())},
                       "saturation_by_stratum_and_t_at_best": {k: ms(v) for k, v in sorted(strat.items())}}

# ---------------------------------------------------------------- 3. temporal gradients: G2 vs G1, W spectrum
def g1_pooled(p):
    return {x['delta']: x['log_phi_q'][1] for x in p['results']['G1']['pooled_delta_weighted']}
g2 = {}
for lab, f in (("step1", lambda r: pid(r, 'early', sigma=1)), ("epoch1", lambda r: pid(r, 'epoch_end', k=1)), ("best", best)):
    acc = {}
    for r in R:
        p = r['prim'][f(r)]; phi = g1_pooled(p)
        for x in p['results']['G2']['pooled_delta_weighted']:
            d = x['delta']
            if d == 0: continue
            c = acc.setdefault(d, {"log_Rbar": [], "log_phi_median": [], "log_Rbar_minus_log_phi": []})
            lr = math.log(x['R_bar_weighted']); c["log_Rbar"].append(lr)
            if d in phi: c["log_phi_median"].append(phi[d]); c["log_Rbar_minus_log_phi"].append(lr - phi[d])
    g2[lab] = {d: {k: ms(v) for k, v in c.items()} for d, c in sorted(acc.items())}
excl = {}
for r in R:
    for s, p in train_points(r):
        res = r['prim'][p]['results']
        if 'G2' in res:
            excl.setdefault(s, []).append(sum(res['G2']['excluded_exact_zero_by_stratum'].values()))
spec = {}
for k in (0, 1, 2, 4, 8, 16, 24, 30, 38):
    rows = [r['hist'][k] for r in R]
    spec[k] = {"sigma_max_W": ms([x['W_singular_values'][0] for x in rows]),
               "sigma_min_W": ms([x['W_singular_values'][-1] for x in rows]),
               "singular_values_above_1": ms([sum(v > 1 for v in x['W_singular_values']) for x in rows]),
               "spectral_radius_W": ms([x['W_spectral_radius'] for x in rows]),
               "orthogonality_defect": ms([x['W_orthogonality_defect'] for x in rows]),
               "param_norms": {n: ms([x['param_norms'][n] for x in rows]) for n in rows[0]['param_norms']}}
out["temporal_gradients"] = {"G2_loss_projected_vs_G1": g2, "G2_exact_zero_exclusions_total_by_sigma": {s: ms(v) or v for s, v in sorted(excl.items())},
                             "W_spectrum_by_epoch": spec}

# ---------------------------------------------------------------- 4. optimisation behaviour
curves = {}
for k in range(0, 39):
    rows = [r['hist'][k] for r in R]
    curves[k] = {"train_nll": ms([x['train_nll'] for x in rows]), "dev_nll": ms([x['dev_nll'] for x in rows]),
                 "running_train_nll": ms([x['running_train_nll'] for x in rows]) if k else None}
gn = {}
for r in R:
    for e in r['o3']['gradient_norm_profile']:
        c = gn.setdefault(e['epoch'], {"median": [], "max": [], "max_over_median": []})
        for kk in c: c[kk].append(e[kk])
first = {}
for r in R:
    for s in r['steps'][:16]:
        first.setdefault(s['step'], {}).setdefault('W', []).append(s['grad_norms']['W'])
rises = [sum(1 for e in r['o3']['epoch_events'] if e['event'] == 'dev_nll_rose') for r in R]
out["optimisation"] = {"learning_curves": curves,
                       "gradient_norm_per_epoch": {e: {k: ms(v) for k, v in c.items()} for e, c in sorted(gn.items())},
                       "W_grad_norm_first_steps": {s: ms(c['W']) for s, c in sorted(first.items())},
                       "dev_nll_rises_per_run": ms(rises), "dev_nll_rise_epochs": {r['seed']: [e['epoch'] for e in r['o3']['epoch_events'] if e['event'] == 'dev_nll_rose'] for r in R},
                       "best_epoch": ms([r['meta']['best_epoch'] for r in R]),
                       "dev_nll_epoch38_minus_best": ms([r['hist'][38]['dev_nll'] - min(x['dev_nll'] for x in r['hist'][1:]) for r in R])}

# ---------------------------------------------------------------- 5. seed variability: per-seed table and robustness
per_seed = []
for r in R:
    pb, p0, p1 = r['prim'][best(r)]['results'], r['prim'][pid(r, 'start')]['results'], r['prim'][pid(r, 'early', sigma=1)]['results']
    pend = r['prim'][pid(r, 'epoch_end', k=38)]['results']
    phi_b, phi_0 = g1_pooled(r['prim'][best(r)]), g1_pooled(r['prim'][pid(r, 'start')])
    z = np.load(r['dir'] + '/research_diagnostics/raw/point_%04d.npz' % best(r), allow_pickle=False)
    row, t, d, dn = z['c1_row'], z['c1_t'], z['c1_d'], z['c1_delta_nll']
    n = np.zeros(int(row.max()) + 1, int); np.maximum.at(n, row, t); eos = t == n[row]; m3 = (d == 3) & (t >= 10)
    per_seed.append({"seed": r['seed'], "best_epoch": r['meta']['best_epoch'],
                     "best_dev_nll": min(x['dev_nll'] for x in r['hist'][1:]), "test_nll": test['per_seed'][str(r['seed'])]['test_nll'],
                     "g_W_epoch38": pend['G1']['g_W'], "sigma_max_W_epoch38": r['hist'][38]['W_singular_values'][0],
                     "log_phi8_init": phi_0[8], "log_phi8_best": phi_b[8],
                     "slope_phi_init": (phi_0[8] - phi_0[1]) / 7, "slope_phi_best": (phi_b[8] - phi_b[1]) / 7,
                     "saturation_t>=3_best": st.fmean(x['saturation'] for x in pb['H1']['per_t_weighted'] if x['t'] >= 3),
                     "Pi_ge3_step1": p1['G3']['full']['long_share_ge3'], "Pi_ge3_epoch38": pend['G3']['full']['long_share_ge3'],
                     "F8_d4_best": next(x['mean'] for x in pb['C1']['F8'] if x['d'] == 4),
                     "F8_d8_best": next(x['mean'] for x in pb['C1']['F8'] if x['d'] == 8),
                     "EOS_d3": float(dn[m3 & eos].mean()), "letter_d3": float(dn[m3 & ~eos].mean())})
chk = lambda f: sum(1 for s in per_seed if f(s))
out["seed_variability"] = {"per_seed": per_seed, "holds_in_n_of_10_seeds": {
    "decay_slower_at_best_than_init (slope_best > slope_init)": chk(lambda s: s['slope_phi_best'] > s['slope_phi_init']),
    "phi_still_decays_at_best (log_phi8_best < 0)": chk(lambda s: s['log_phi8_best'] < 0),
    "W_amplifies (sigma_max_W_epoch38 > 1)": chk(lambda s: s['sigma_max_W_epoch38'] > 1),
    "saturation_rises (best > 0.10)": chk(lambda s: s['saturation_t>=3_best'] > 0.10),
    "F8_d8_below_0.05": chk(lambda s: s['F8_d8_best'] < 0.05),
    "EOS_dominates_at_d3 (EOS > 3 x letter)": chk(lambda s: s['EOS_d3'] > 3 * s['letter_d3']),
    "HC1_long_share_grew (Pi_epoch38 > Pi_step1)": chk(lambda s: s['Pi_ge3_epoch38'] > s['Pi_ge3_step1']),
    "test_above_dev": chk(lambda s: s['test_nll'] > s['best_dev_nll'])}}

# ---------------------------------------------------------------- 6. estimate quality caveats
cos = {}
for r in R:
    for s, p in train_points(r):
        g = r['prim'][p]['results'].get('G3')
        if g: cos.setdefault(s, []).append(g['agreement']['cos_halves'])
out["estimate_quality"] = {"cos_G_A_G_B_by_sigma": {s: ms(v) for s, v in sorted(cos.items())},
                           "note": "The probe halves estimate the same population gradient; low cosine late in training means the per-distance shares of Ĝ are noisier there, although Π̂(Δ≥3) keeps its sign in both halves."}

# ---------------------------------------------------------------- 7. generated samples (needs torch; Mac)
try:
    import torch
    from models.recurrent.rnn import RNNLanguageModel
    from utils.reproducibility import make_generator
    samples = {}
    for r in R:
        m = RNNLanguageModel.from_checkpoint(os.path.join(r['dir'], 'model_best.pt'))
        g = make_generator(12345)
        gen = m.generate(num_samples=12, max_length=30, generator=g)
        samples[r['seed']] = [{'text': x.text, 'finished': x.finished} for x in gen]
    lens = [len(x['text']) for v in samples.values() for x in v]
    out["generation"] = {"samples_temperature_1": samples, "mean_length": st.fmean(lens),
                         "unfinished": sum(1 for v in samples.values() for x in v if not x["finished"]),
                         "note": "Qualitative only: 12 samples per seed, temperature 1, max_length 30, generator seed 12345; no metric is derived."}
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise  # only a missing torch (Linux check runs) may skip generation
    out["generation"] = None

dest = os.environ.get('M6_EXTENDED_OUT') or os.path.join(ROOT, 'analysis/extended_summary.json')
with open(dest, 'x') as fh:
    json.dump(out, fh, indent=1, sort_keys=True, default=str)
print("written", dest)
