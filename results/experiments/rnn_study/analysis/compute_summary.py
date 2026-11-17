"""Model 6 final-study summaries (diag spec §9, predeclared; plus the exploratory EOS/letter split of C1).
Read-only over results/experiments/rnn_study/L7_research; writes analysis/summary.json. Run from the repository root:
    python3 results/experiments/rnn_study/analysis/compute_summary.py
"""
import glob, json, math, os, statistics as st, sys
import numpy as np
ROOT = os.path.join(os.getcwd(), 'results/experiments/rnn_study')
RUNS = os.path.join(ROOT, 'L7_research/runs/rnn')

def ms(v):
    v = [x for x in v if x is not None]
    return {"mean": st.fmean(v), "se": st.stdev(v) / math.sqrt(len(v)), "n": len(v)} if len(v) > 1 else None

def load():
    runs = []
    for d in sorted(glob.glob(RUNS + '/run_*')):
        pts = {json.loads(l)['point_id']: json.loads(l) for l in open(d + '/research_diagnostics/points.jsonl')}
        prim = {json.loads(l)['point_id']: json.loads(l) for l in open(d + '/research_diagnostics/primary.jsonl')}
        runs.append({"dir": d, "seed": json.load(open(d + '/config.json'))["seed"], "meta": json.load(open(d + '/metadata.json')),
                     "hist": [json.loads(l) for l in open(d + '/history.jsonl')], "pts": pts, "prim": prim})
    return runs

def pid(r, kind, k=None, sigma=None):
    return next(p for p, x in r['pts'].items() if x['kind'] == kind and (k is None or x['completed_epochs'] == k)
                and (sigma is None or x['sigma'] == sigma))

R = load()
assert len(R) == 10 and all(r['meta']['outcome'] == 'completed' and r['meta']['diagnostics_status'] == 'complete' for r in R)
best = lambda r: pid(r, 'epoch_end', k=r['meta']['best_epoch'])
P = {"init": lambda r: pid(r, 'start'), "epoch1": lambda r: pid(r, 'epoch_end', k=1), "best": best}
out = {"seeds": [r['seed'] for r in R], "best_epochs": [r['meta']['best_epoch'] for r in R],
       "best_dev_nll": ms([min(x['dev_nll'] for x in r['hist'][1:]) for r in R]),
       "train_nll_at_best": ms([r['hist'][r['meta']['best_epoch']]['train_nll'] for r in R])}
# RQA A1: pooled population-weighted median log phi; W-only log omega; D-based log delta-hat (median over strata)
A1 = {}
for lab, f in P.items():
    rows = {}
    for r in R:
        g1 = r['prim'][f(r)]['results']['G1']
        dh = {}
        for x in g1['per_stratum_delta']:
            dh.setdefault(x['delta'], []).append(x['log_delta_hat_q'][1])
        for x in g1['pooled_delta_weighted']:
            d = x['delta']; c = rows.setdefault(d, {"log_phi": [], "log_omega": [], "log_delta_hat": []})
            c["log_phi"].append(x['log_phi_q'][1]); c["log_omega"].append(math.log(g1['omega'][d - 1]))
            c["log_delta_hat"].append(st.median(dh[d]))
    A1[lab] = {d: {k: ms(v) for k, v in c.items()} for d, c in sorted(rows.items())}
out["RQA_A1_attenuation"] = A1
# A2 saturation and E[tanh'^2] per t (population-weighted)
A2 = {}
for lab, f in P.items():
    acc = {}
    for r in R:
        for x in r['prim'][f(r)]['results']['H1']['per_t_weighted']:
            c = acc.setdefault(x['t'], {"saturation": [], "mean_sq_dtanh": []})
            c["saturation"].append(x['saturation']); c["mean_sq_dtanh"].append(x['mean_sq_dtanh'])
    A2[lab] = {t: {k: ms(v) for k, v in c.items()} for t, c in sorted(acc.items())}
out["RQA_A2_H1"] = A2
# A3 log sigma vs log phi at E_sel (init, epoch 1, epoch 38): stratum-median of quartile-medians
A3 = {}
for lab, f in (("init", P["init"]), ("epoch1", P["epoch1"]), ("epoch38", lambda r: pid(r, 'epoch_end', k=38))):
    acc = {}
    for r in R:
        per = {}
        for x in r['prim'][f(r)]['results']['G1']['per_stratum_delta']:
            per.setdefault(x['delta'], []).append(x)
        for d, v in per.items():
            c = acc.setdefault(d, {"log_sigma": [], "log_phi": []})
            c["log_sigma"].append(st.median(y['log_sigma_q'][1] for y in v)); c["log_phi"].append(st.median(y['log_phi_q'][1] for y in v))
    A3[lab] = {d: {k: ms(v) for k, v in c.items()} for d, c in sorted(acc.items())}
out["RQA_A3_sigma_vs_phi"] = A3
out["RQA_A4_g_W"] = {k: ms([r['prim'][pid(r, 'start') if k == 0 else pid(r, 'epoch_end', k=k)]['results']['G1']['g_W'] for r in R])
                     for k in (0, 1, 2, 4, 8, 16, 24, 30, 38)}
# RQB B1: F8 / F7 at best epoch
out["RQB_B1"] = {F: {x['d']: None for x in R[0]['prim'][best(R[0])]['results']['C1'][F]} for F in ("F8", "F7")}
for F in ("F8", "F7"):
    for d in out["RQB_B1"][F]:
        v = [next(x for x in r['prim'][best(r)]['results']['C1'][F] if x['d'] == d) for r in R]
        out["RQB_B1"][F][d] = {"delta_nll": ms([x['mean'] for x in v]), "mean_clustered_se": st.fmean(x['se'] for x in v),
                               "targets": v[0]['targets'], "names": v[0]['names']}
# B2: stratum-3 (t, d) matrix pooled over seeds
cells = {}
for r in R:
    for x in r['prim'][best(r)]['results']['C1']['matrix']:
        if x['stratum'] == 3:
            c = cells.setdefault(f"{x['t']}:{x['d']}", [0.0, 0]); c[0] += x['sum']; c[1] += x['count']
out["RQB_B2_stratum3_mean_by_t_d"] = {k: v[0] / v[1] for k, v in cells.items() if v[1]}
# Exploratory: F8 set split into EOS and letter targets (raw per-target values)
split = {}
for r in R:
    z = np.load(r['dir'] + '/research_diagnostics/raw/point_%04d.npz' % best(r), allow_pickle=False)
    row, t, d, dn = z['c1_row'], z['c1_t'], z['c1_d'], z['c1_delta_nll']
    n = np.zeros(int(row.max()) + 1, int); np.maximum.at(n, row, t); eos = t == n[row]
    for dd in range(1, 9):
        m = (d == dd) & (t >= 10)
        for lab, mm in (("EOS", m & eos), ("letter", m & ~eos)):
            split.setdefault(f"{lab}:{dd}", []).append(float(dn[mm].mean()))
    split.setdefault("counts_EOS_letter", (int(((d == 1) & (t >= 10) & eos).sum()), int(((d == 1) & (t >= 10) & ~eos).sum())))
out["exploratory_F8_EOS_vs_letter"] = {k: (ms(v) if k != "counts_EOS_letter" else v) for k, v in split.items()}
# RQC C1/C2: over all early and epoch-end points
C = []
for s in sorted({x['sigma'] for x in R[0]['pts'].values() if x['kind'] in ('early', 'epoch_end')}):
    rows = []
    for r in R:
        p = pid(r, 'early', sigma=s) if s < 802 else next(q for q, x in r['pts'].items() if x['kind'] == 'epoch_end' and x['sigma'] == s)
        res = r['prim'][p]['results']; g = res['G3']; f = {x['d']: x['mean'] for x in res['C1']['F8']}
        rows.append((g['full']['long_share_ge3'], g['full']['kappa'], g['agreement']['cos_halves'],
                     g['agreement']['long_share_ge3']['spread'], g['agreement']['long_share_ge3']['sign_reproduced'], f))
    C.append({"sigma": s, "epoch": r['pts'][p]['completed_epochs'], "Pi_ge3": ms([x[0] for x in rows]), "kappa": ms([x[1] for x in rows]),
              "cos_halves": ms([x[2] for x in rows]), "Pi_half_spread_mean": st.fmean(x[3] for x in rows),
              "Pi_sign_reproduced": sum(bool(x[4]) for x in rows),
              "F8": {d: ms([x[5][d] for x in rows]) for d in range(1, 9)}})
out["RQC_over_training"] = C
dest = os.path.join(ROOT, 'analysis'); os.makedirs(dest, exist_ok=True)
with open(os.path.join(dest, 'summary.json'), 'x') as fh:
    json.dump(out, fh, indent=1, sort_keys=True, default=str)
print("written", os.path.join(dest, 'summary.json'))
