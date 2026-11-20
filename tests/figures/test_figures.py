"""The figures must show exactly the numbers the studies report, in the shared style.

These tests read the published result files; they skip if a file is absent."""
import csv
import json
import math

import pytest

from figures import data, style as st

needs = pytest.mark.skipif(not all(p.exists() for p in (data.WAVENET_PER_SEED, data.WAVENET_CONTEXT,
                                                         data.TRANSFORMER_RESULT, data.POSITION_BY_INDEX)),
                           reason="published result files not present")


@needs
def test_wavenet_comparisons_match_the_wavenet_report():
    w = data.wavenet_test_nll()
    rows = {r["comparison"]: r for r in csv.DictReader(
        (data.RESULTS / "wavenet_study/final/analysis/final/comparisons.csv").open()) if r["endpoint"] == "test_nll"}
    for name, (a, b) in {"C-A": ("C", "A"), "D-B": ("D", "B")}.items():
        d = data.paired(w[a], w[b])
        assert math.isclose(d.mean, float(rows[name]["mean_difference"]), abs_tol=1e-12)
        for got, want in zip(d.interval, (float(rows[name]["ci95_low"]), float(rows[name]["ci95_high"]))):
            assert math.isclose(got, want, abs_tol=1e-9)


@needs
def test_transformer_comparisons_match_the_analysis_file():
    t = data.transformer_test_nll()
    primary = json.loads((data.RESULTS / "transformer_study/test_evaluation/analysis.json").read_text())["primary"]
    cases = {"T-RQ1a": ("gpt2_style", "rnn", 0.95, "ci95"),
             "T-RQ1b_equivalence": ("gpt2_style", "wavenet_d", 0.90, "ci90"),
             "T-RQ2": ("transformer_2017", "gpt2_style", 0.95, "ci95")}
    for key, (a, b, level, ci) in cases.items():
        d = data.paired(t[a], t[b], level)
        assert math.isclose(d.mean, primary[key]["mean_diff"], abs_tol=1e-12)
        for got, want in zip(d.interval, primary[key][ci]):
            assert math.isclose(got, want, abs_tol=1e-9)


@needs
def test_pooled_tail_matches_the_grouped_position_analysis():
    grouped = json.loads((data.RESULTS / "transformer_study/dev_diagnostics/position_paired.json").read_text())
    by_index = data.position_by_index("gpt2_style - wavenet_d")
    want = grouped["pairs"]["gpt2_style - wavenet_d"]["i>=9"]
    for key in ("targets", "mean", "se_name_clustered"):
        assert math.isclose(by_index["9+"][key], want[key], rel_tol=1e-12)


@needs
def test_context_frequency_categories_are_complete():
    freq = data.wavenet_context_frequency()
    for cat in ("unseen", "seen:low", "seen:mid", "seen:high"):
        for comp in ("C-A", "D-B"):
            assert (cat, comp) in freq


def test_bundled_fonts_and_licence_are_present():
    for name in st.FONT_FILES:
        assert (st.FONT_DIR / name).stat().st_size > 100_000
    assert "SIL Open Font License" in (st.FONT_DIR / "OFL-LICENSE.md").read_text()


def test_style_uses_the_bundled_typeface():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    st.setup()
    assert plt.rcParams["font.family"] == [st.FAMILY]
    path = font_manager.findfont(font_manager.FontProperties(family=st.FAMILY))
    assert path.startswith(str(st.FONT_DIR))


def test_signed_uses_the_true_minus_sign():
    assert st.signed(-0.0219, 3) == "−0.022"
    assert st.signed(0.0204, 3) == "+0.020"
