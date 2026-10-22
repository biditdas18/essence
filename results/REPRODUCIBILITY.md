# Reproducibility

## Environment that produced every number in this paper

Created with conda (conda-forge). Recreate with:

    conda create -n essence -c conda-forge python=3.12.14 numpy=1.26.4 pandas=2.2.3 scikit-learn=1.6.1
    # the sentence-transformers step additionally needs:
    conda create -n essence-embed -c conda-forge python=3.11 pytorch sentence-transformers
    pip install socksio   # only if your network routes through a SOCKS proxy

| package | version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.17.1 |
| tqdm | 4.67.1 |
| sentence-transformers | not installed in this env |
| torch | not installed in this env |

Platform on which the reported numbers were produced: macOS-27.0-arm64-arm-64bit

## Known reproducibility gap — disclosed

The previously published Amazon Books validation figure was reported as 0.032146 at K = 10.
Two independent environments on this machine both produce **0.032075** (a relative difference of
0.22%):

- a freshly pinned environment matching `requirements.txt` (numpy 1.26.4, pandas 2.2.3,
  scikit-learn 1.6.1), and
- the repository's own pre-existing virtual environment (scikit-learn 1.7.2).

The discrepancy was localised, not guessed at. Mean items-per-cluster matches to the digit
(2.58405), which rules out any difference in the data, the user sample, or the train/validation
split, and places the difference inside K-means itself. Thread count was tested as a cause and
ruled out: the value is stable at 0.032074877 across 1, 4 and 8 BLAS threads. The original
environment is not reconstructible because **no lockfile was committed at the time**, which this
file now fixes.

Two consequences worth stating plainly:

1. The affected quantity is a validation-selection figure, not a reported test result, and the
   0.22% shift changes no conclusion: re-running the full chain in the new environment at the
   published K = 10 reproduced the published significance count exactly (12 of 18 after FDR on
   Amazon Books, 7 of 18 on MovieLens-25M).
2. **The adopted K values are invariant to it.** They are fixed by the items-per-cluster floor
   and the all-users-clustered constraint, both of which are exact arithmetic on per-user
   history lengths, and both reproduce to the digit.

## Determinism

Seeds are fixed (SEED = 42) for the bootstrap, K-means initialisation, and all sampling. Within
one environment, repeated runs reproduce bit-for-bit. Across environments, expect differences at
the 0.2% level in K-means-dependent quantities, for the reason above.
