# MASTER FINDINGS — Essence

**Source of truth for every number in the paper.** Regenerated 2026-09-28 from
the results files, not from notes. Supersedes all earlier versions. Where this document and the
paper disagree, the paper is stale.

Target venue: SOICT 2026 (Springer CCIS, single-blind, 12 pages excluding references), deadline
5 Oct 2026, notification 12 Oct 2026.

---

## 1. Headline finding: population density, not domain

Ranks at the adopted $K$, Recall@10, conservative baselines:

| dataset | users | adopted $K$ | Essence Recall@10 | rank | significant / tests |
|---|---|---|---|---|---|
| Last.fm-1K | 99 | 10 | 0.028075 | **2/10** | 6/18 |
| Amazon Books | 2,000 | 8 | 0.024619 | **3/10** | 10/18 |
| MovieLens-25M | 2,000 | 10 | 0.005511 | **8/10** | 7/18 |

**Tested, not merely described.** All 8 of 8
between-dataset contrasts of Cohen's $d$ are significant after BH, every one is a sign flip, and
no CI contains zero ($\Delta d$ 0.58--0.83). See `domain_dependence_interaction.csv`.

**The mediator is co-occurrence density, not the domain.** Demonstrated within Last.fm-1K:

| cohort | users | items heard by $\geq$2 users | Essence | CF-ItemKNN | $d$ | $q$ |
|---|---|---|---|---|---|---|
| 50--500 (published) | 99 | 10.3% | 0.028075 | 0.001756 | **+0.281** | <0.0001 |
| 50--2,000 | 374 | 28.4% | 0.002634 | 0.005475 | **-0.160** | 0.0139 |

Lifting the cap reverses the verdict and makes CF-ItemKNN the best of eight systems; Essence
falls 2nd to 5th. The same mechanism appears independently on Amazon Books, where heavy users
degrade CF-ItemKNN by 22% against 86--93% for content methods. Confound disclosed: the cap lift
raises user count *and* history length, both acting through density, and they cannot be
separated in this dataset. See `COHORT_NOT_DOMAIN.md`.

**Critical caveat for the paper.** Last.fm's classification as a sparse-co-occurrence dataset
holds *for the published 99-user cohort only*. Do not describe music as a weak-collaborative-
signal domain.

---

## 2. Central negative result: the clustering null

| test | scope | result |
|---|---|---|
| Silhouette-stratified, adopted $K$, tie-aware $p$ | 18 comparisons, no exclusions | **0/18 favour clustering** |
| Encoder robustness | 12 comparisons, 3 datasets x 2 encoders x 2 metrics | **0 favour clustering**, 1 favours recency |
| Heavy users (Amazon, 330 items/user, $K=60$) | powered to 21.6% | null holds, $q=0.31$ |
| Expanded Last.fm cohort (374 users) | -- | null holds; LT-Recall $q=0.031$ clustering **worse** |
| Global temporal cutoff | 14 comparisons | 1 exception, **not significant study-wide** |
| Seed stability | 3 $K$-means seeds x 2 user samples | holds in all 5; published seed is unfavourable to Essence |

**Study-wide:** 136 hypothesis tests in total. BH across all of them takes
significance from 59 to 56 and changes two rows, both
against clustering: the one pro-clustering exception dies ($q=0.075$), and the
**high-silhouette stratum** shows clustering significantly *worse* ($q=0.033$). Under study-wide
correction, **no comparison anywhere favours clustering**.

---

## 3. Power

`null_power_analysis.csv`: 18 cells, 17 informative,
0 reaching their own 80%-power MDE. Ranking-metric nulls are
adequately powered and carry content; several long-tail cells cannot detect a doubling and are
**not claimed as nulls**.

---

## 4. Identity shortcut (corrected)

| $K$ | shortcut share | collapse when same-artist excluded |
|---|---|---|
| 3 (as published) | 20.6% | 87.9% |
| **10 (adopted)** | **32.9%** | **95.3%** |

The published analysis used a hardcoded $K=3$. Its figures reproduce exactly at that $K$; at the
adopted $K$ the dependence is substantially larger. Report the adopted-$K$ figures.

---

## 5. Other verified results

- **Cold start:** content beats CF in 5 of 6 coldest-decile comparisons at adopted $K$. Generic
  to content-only methods, not Essence-specific.
- **Metric robustness:** Essence holds the same rank under Recall@5/@10/@20 and NDCG@10 on
  Amazon and MovieLens, with MIND and ComiRec now included. Last.fm on Recall@10 only.
- **Sequential baselines:** SASRec and GRU4Rec via RecBole, validation-tuned, 21--175x below
  Essence on Last.fm and Amazon. Not evidence Essence beats them in general.
- **Conservative convention:** each baseline at the better of published and tuned. Costs
  Essence its Last.fm ComiRec win ($d=0.065$, $q=0.54$) --- which is also underpowered, so
  "not shown to differ", never "equal".
- **Statistical defect:** the two-sided bootstrap counted tied resamples as evidence against the
  null. Fixed in all call sites; the correction moves against us.

---

## 6. Closed after the camera-ready was built (22 Oct 2026)

Two gaps the README carried are now closed, both re-run at the adopted $K$:

- **Active-cluster-selection ablation**, previously only at the earlier $K$ (10/10/15).
  At 10/8/10, recency-select leads mean-select on Recall@10 on all three datasets
  (0.0281/0.0016, 0.0246/0.0197, 0.0055/0.0045) and on LT-Recall on two of three.
  MovieLens-25M stays mixed: Essence's LT-Recall is exactly 0.0000 against mean-select's
  0.0005, which is also one of the cells section 3 declines to call a null.
  `results/scratch_ablation_lt_{lastfm_K10,amazon_K8,movielens_K10}.csv`; the script now
  takes `--lastfm-k/--amazon-k/--movielens-k` and tags its output with $K$.
- **Architecture-family test** (Recency-Weighted vs the four collaborative baselines) had
  no script behind it -- the repository's one such gap. Now
  `evaluation/recency_weighted_family_test.py` ->
  `results/recency_weighted_family_test.csv`. At the adopted $K$ the Recall@10 sign flip
  is identical to Essence's: **8/8 ahead in the sparse family, 0/4 in the dense one**
  (baselines ahead 4/4). Removing clustering does not remove the pattern. LT-Recall is
  flatter: 4/8 sparse, 2/4 on MovieLens-25M where Essence has none.

---

## 7. Not verified / out of scope

- Last.fm chain not patched for Recall@5/@20/NDCG@10.
- MovieLens baseline re-runs used $\mathrm{lr}=10^{-3}$, $K=4$: Recall@10 within $-0.8\%$ and
  $+11.2\%$ of published. Conclusion unaffected (baselines lead 8.5--12.7x).
- MovieLens baseline tuning stopped at 2 of 24 as a disclosed scope decision.
- The 975-user uncapped Last.fm cohort was still running at the time of writing.
- Amazon ComiRec's published-configuration hyperparameters were never recorded, so only its
  Recall@10 can be reported at the conservative value.
