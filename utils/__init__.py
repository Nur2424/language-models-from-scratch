"""Shared infrastructure for the language-models-from-scratch project.

Import submodules explicitly, e.g. ``from utils.data import load_names``.
Nothing is re-exported here so that ``import utils`` stays cheap and the
import graph stays acyclic.

Module map
----------
paths            repository-relative path resolution
io               atomic file writes, JSON / JSONL helpers
config           typed YAML configuration (load, merge, override, validate, save)
vocabulary       character vocabulary and token conventions (BOS/EOS, PAD, IGNORE_INDEX)
data             names loading, leakage-safe splitting, fingerprints, padding, context windows
batching         seeded, shuffled, without-replacement epoch ordering of example indices
reproducibility  seeding, independent RNG streams, determinism, RNG state capture
device           explicit CPU / CUDA / MPS selection and device metadata
statistics       numerically stable summaries shared by metrics and diagnostics
metrics          token NLL with explicit reduction, perplexity, parameter counts
logging_utils    console + text + JSONL logging without duplicate handlers
checkpoint       best / final / resume checkpoint primitives
run_manager      run directories, metadata, status tracking, history
generation       temperature / top-k / top-p sampling primitives
diagnostics      gradient, activation and parameter-update summaries
"""

__version__ = "0.1.0"
