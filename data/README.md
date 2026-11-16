# Data

`names.txt` is the only dataset used in this repository: 32,033 first names, one per line.

## Source

The file is taken unchanged from Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) repository (MIT licence). There it is described as the most common 32K names from the US Social Security Administration's [baby names data](https://www.ssa.gov/oact/babynames/) for the year 2018.

## Contents

| Property | Value |
|---|---|
| Lines (names) | 32,033 |
| Distinct names | 29,494 (2,539 names appear twice) |
| Characters | lowercase `a`–`z` only |
| Name length | 2 to 15 characters, mean 6.12 |
| Trailing newline | none (so `wc -l` reports 32,032) |
| SHA-256 | `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d` |

To confirm you have the same file:

```bash
shasum -a 256 data/names.txt
```

## How it is split

Splitting is done in code (`utils/data.py`), never by hand, with the settings in `configs/base.yaml`:

- 80% train, 10% dev, 10% test, taken over **distinct** names;
- every copy of a repeated name is kept in the same split, so no name appears in two splits;
- a fixed split seed (42), independent of the training seed.

| Split | Names | Predicted characters (including the end-of-name token) |
|---|---|---|
| Train | 25,640 | 182,666 |
| Dev | 3,196 | 22,801 |
| Test | 3,197 | 22,679 |

The character vocabulary is built from the training split only.
