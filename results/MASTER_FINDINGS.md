# Master Findings — Essence Robustness & Generalization Engagement

> **STALE — reflects $K{=}3$ baseline results only.** $K$-validation work
> (completed 2026-08-24) selected $K{=}10$ (Last.fm-1K), $K{=}10$ (Amazon
> Books), and $K{=}15$ (MovieLens-25M) per dataset, and found the headline
> findings — domain-dependence, the clustering-mechanism null result, the
> cold-start/singleton scope boundary — replicate at the validated $K$'s.
> Several *numbers* below changed materially (ranks, effect sizes,
> significance counts): see `paper/essence_paper_v2.tex` Sections 3-5 for
> current, authoritative results. **This file has not been updated to
> match and should not be cited for current figures.**

Reference document for the paper-writing pass. Not paper text. Every number
below is pulled fresh from the cited CSV, not recalled from conversation —
re-run the cited script if you want to reproduce it. Compiled after Step 11
(MovieLens-25M third dataset). All Recall@10 / LT-Recall@10 figures are
macro-averages (per-user ratio, then mean across users — see README's
"Averaging convention audit").

---

## 1. Last.fm-1K (99 users, 22,767 catalog items, Music)

### Headline Recall@10 (`results/evaluation_results_v7_mind_comirec.csv`)

| System | Recall@10 | LT-Recall@10 |
|---|---|---|
| Recency-Weighted | 0.0304 | 0.0151 |
| Last-Item | 0.0265 | 0.0064 |
| Avg-Last-10 | 0.0252 | 0.0134 |
| ComiRec (SA) | 0.0137 | 0.0050 |
| **Essence (K=3)** | **0.0119** | **0.0059** |
| MIND | 0.0078 | 0.0021 |
| Content (Avg Emb) | 0.0038 | 0.0029 |
| Popularity | 0.0023 | 0.0000 |
| CF (ItemKNN) | 0.0018 | 0.0000 |
| Random | 0.0001 | 0.0000 |

**Pattern: Essence loses to the 3 recency baselines, beats the original 4 baselines (Random/Popularity/CF/Content) and MIND, no significant difference vs. ComiRec.**

### Significance (`results/fdr_corrected_significance.csv`, part of the 36-comparison Last.fm+Amazon family)

Significant after BH-FDR correction (q<0.05):
- Essence beats Random (d=+0.219), CF-ItemKNN (d=+0.187), Popularity (d=+0.176), Content (d=+0.185) — all small effect sizes.
- Essence loses to Recency-Weighted (d=−0.265), Last-Item (d=−0.295), Avg-Last-10 (d=−0.266) on Recall@10.
- Essence vs MIND and vs ComiRec: **not significant** on either metric (underpowered at n=99).
- Essence vs Recency-Weighted on **LT-Recall@10**: significant at raw p (0.0486) but **flips to non-significant after FDR correction** (q=0.0875) — the single comparison in the whole 36-item family that flips.

### K-sensitivity (`results/sensitivity_results.csv`)
K=2: 0.0080 / K=3: 0.0119 / K=4: 0.0239 / K=5: 0.0251 (Recall@10). K=3 (the paper's default) is **not** near-optimal — K=4/5 clearly outperform it. Seed variance at K=3 (10 seeds): 0.0122±0.0011 recall, 0.0103±0.0038 LT — much smaller than the K=3→K=4 jump, so this isn't noise.

### Multi-cutoff (`results/multi_cutoff_lastfm.csv`)
Recency baselines beat Essence at k=5, 10, **and** 20, on both metrics, consistently — not a k=10 artifact.

### Artist-shortcut check (`results/shortcut_analysis_summary.csv`)
20.6% of Essence's recs share an artist with the active cluster. Holdout (excluding those candidates) collapses Recall@10 to 0.0014 and LT-Recall@10 to 0.0005 — same-artist candidates account for a disproportionate share of hits despite being a minority of recs.

### ItemKNN neighbor-count sweep (`results/itemknn_neighbor_sweep.csv`)
No real "neighbor count" hyperparameter existed in the canonical ItemKNN (full aggregate scoring, not top-N). Added one for this test only. All N values give essentially flat, noisy Recall@10 (0.0017–0.0020) — canonical "unrestricted" isn't meaningfully suboptimal.

### Compute cost (`results/compute_cost.csv`)
Essence K-means fit: 9.8ms/user. Essence full inference: 72.1ms/user (dict-based scoring over 22,767 items — the slowest path measured; Amazon's vectorized equivalent is 6-9x faster despite a larger catalog). MIND/ComiRec inference: 19.3ms (noisy, small sample) / 3.5ms.

---

## 2. Amazon Books (2,000 users, 61,727 catalog items, E-commerce)

### Headline Recall@10 (`results/results_amazon_peruser_mind_comirec.csv`)

| System | Recall@10 | LT-Recall@10 |
|---|---|---|
| Last-Item | 0.0311 | 0.0215 |
| Recency-Weighted | 0.0260 | 0.0179 |
| Avg-Last-10 | 0.0235 | 0.0196 |
| **Essence (K=3)** | **0.0221** | **0.0197** |
| Content (Avg Emb) | 0.0173 | 0.0124 |
| ComiRec (SA) | 0.0056 | 0.0121 |
| MIND | 0.0052 | 0.0057 |
| CF (ItemKNN) | 0.0031 | 0.0137 |
| Popularity | 0.0021 | 0.0000 |
| Random | 0.0001 | 0.0000 |

**Pattern: Essence beats Content/CF/Popularity/Random/MIND clearly, loses to Last-Item on both metrics, and is statistically tied with Avg-Last-10/Recency-Weighted on LT-Recall specifically.** This is the dataset where Essence's headline claim holds up best.

### Significance (`results/fdr_corrected_significance.csv`)
Significant after FDR: Essence beats Popularity/Random/CF-ItemKNN/Content/MIND/ComiRec on Recall@10 (d ranges 0.069–0.259); beats Popularity/Random/Content/MIND on LT-Recall@10. Essence loses to Last-Item (d=−0.090) and Recency-Weighted (d=−0.059) on Recall@10, both small effects.

### BCa robustness check (`results/bca_amazon_check.csv`)
Essence vs MIND on LT-Recall@10: robust under BCa (no conclusion change). **Essence vs ComiRec on LT-Recall@10: flips** — not significant under percentile bootstrap (CI touches 0 at −0.0001), significant under BCa (CI starts at +0.0001). Right on the boundary; flagged, not resolved.

### MIND routing bug (`experiments/mind_comirec/test_mind_routing.py`, `experiments/mind_comirec/model.py`)
Found and fixed: MIND's routing logits were zero-initialized, and because MIND shares one bilinear matrix across all K capsules, this meant all capsules were mathematically guaranteed to collapse to identical vectors — MIND's effective K was always 1 regardless of the nominal setting, for every run before the fix. Fixed to match the paper's `b ~ N(0, σ²)` initialization. **All MIND results in this document are post-fix.** The fix did not change any headline significance conclusion (Essence vs MIND remained significant on Amazon both before and after).

### Popularity-decile breakdown — cold-start vs. singleton scope boundary (`results/popularity_decile_recall.csv`, `results/decile_significance_check.csv`, `results/decile_bca_check.csv`, `results/longtail_definition_reconciliation.csv`)

Tested: Essence vs. Last-Item / Avg-Last-10 / Recency-Weighted only (**not CF-ItemKNN** — CF was never included in the Amazon decile breakdown).

- **Decile 1 contains zero singleton items** — it's 100% items with zero train interactions (true cold-start), not "rare observed items." Decile 2 has only 3.6% of the 42,878 singletons; 86% of singletons sit in deciles 3–9.
- Essence loses to Last-Item at both deciles 1 and 2 (significant after FDR, d=−0.093 and −0.065), loses to Recency-Weighted at decile 2 (significant, d=−0.075; decile-1 version significant at raw p but not after FDR), no significant difference vs. Avg-Last-10 at either decile. All 4 raw-significant comparisons confirmed robust under BCa.
- **100% of Essence's actual LT-Recall@10 hits come from deciles 3–9** (37% from decile 9 alone) — zero from deciles 1 or 2. The decile-1/2 cold-start weakness and the singleton-recall headline win are about **almost entirely disjoint item populations** — not a contradiction, a scope boundary (see README's dedicated section).

### Review-text pollution (Pass 1 vs Pass 2, `experiments/amazon_books/results_amazon_peruser.csv` vs `results_amazon_pass2_peruser.csv`)
Content: 0.0173→0.0027 (retains 15.4%). Essence: 0.0221→0.0041 (retains 18.4%). Both degrade by a similar order of magnitude when switching from metadata to review-text embeddings — a shared confound, not Essence-specific. Essence's absolute point-loss is larger (mechanical, since it starts from a higher baseline) but its proportional retention is slightly *better* than Content's.

### K-sensitivity (`results/sensitivity_results_amazon.csv`)
K=2: 0.0214 / K=3: 0.0221 / K=4: 0.0228 / K=5: 0.0242. Much flatter than Last.fm — K=3 is close to optimal here, unlike Last.fm.

### Multi-cutoff (`results/multi_cutoff_amazon.csv`)
Last-Item leads at every cutoff on both metrics, but the LT-Recall@10 gap narrows sharply with k (Last-Item's relative lead: +53% at k=5 → +9% at k=10 → +4.5% at k=20). **Essence overtakes Avg-Last-10 and Recency-Weighted on LT-Recall@20** despite trailing both on Recall@20.

### Author-shortcut check (`results/shortcut_analysis_amazon_summary.csv`)
19.8% of recs share an author with the active cluster (n=2000). Holdout collapses Recall@10 to 0.0073 and LT-Recall@10 to 0.0080 — same pattern as Last.fm.

### ItemKNN neighbor sweep
Not run on Amazon (Last.fm only, per original scope).

### Compute cost (`results/compute_cost.csv`)
Essence K-means: 10.8ms/user. Essence full inference (vectorized): 10.5ms/user — the vectorized candidate-matrix approach used for Amazon is far faster than Last.fm's dict-based equivalent despite a 2.7x larger catalog.

---

## 3. MovieLens-25M (2,000-user subsample, 7,654 catalog items with fetched TMDb text, Movies)

### Headline Recall@10 (`results/results_movielens_peruser_full.csv`)

| System | Recall@10 | LT-Recall@10 |
|---|---|---|
| CF (ItemKNN) | 0.0850 | 0.0037 |
| ComiRec (SA) | 0.0706 | 0.0014 |
| Popularity | 0.0517 | 0.0000 |
| MIND | 0.0423 | 0.0000 |
| Last-Item | 0.0091 | 0.0031 |
| Recency-Weighted | 0.0076 | 0.0046 |
| Avg-Last-10 | 0.0064 | 0.0026 |
| **Essence (K=3)** | **0.0048** | **0.0020** |
| Content (Avg Emb) | 0.0043 | 0.0020 |
| Random | 0.0012 | 0.0000 |

**Pattern: Essence loses clearly — 8th of 10 systems, barely above Content and Random.** A third, distinct pattern from both prior datasets.

### Significance (`results/fdr_corrected_significance_movielens.csv`, 18-comparison family, separate from the main 36)
7/18 significant after FDR, all on Recall@10: Essence beats only Random (d=+0.122); loses to CF-ItemKNN (**d=−0.558**), ComiRec (**d=−0.522**), Popularity (**d=−0.447**), MIND (**d=−0.388**), Last-Item (d=−0.092), Recency-Weighted (d=−0.081). **These are the four largest-magnitude effect sizes in the entire engagement** — everywhere else across Last.fm/Amazon, |d| stayed under 0.3. All 7 confirmed robust under BCa (`results/bca_movielens_check.csv`, zero conclusion changes). LT-Recall@10: no significant differences anywhere, but the total hit counts are vanishingly small (Essence: 1 hit total across 2,000 users; CF: 3 hits total) — underpowered, not evidence of a tie.

### Popularity-decile breakdown (`results/popularity_decile_recall_movielens.csv`, `results/decile1_bootstrap_movielens.csv`, `results/decile2_bootstrap_movielens.csv`, `results/longtail_definition_reconciliation_movielens.csv`)
Tested: Essence vs. **CF-ItemKNN only** (the dominant baseline here; MIND/ComiRec excluded — no persisted model checkpoint, would need retraining).

- CF's overall dominance is **not uniform** — it comes almost entirely from decile 10 (CF: 0.1177, the single highest number in any decile analysis in this engagement), while CF is weak-to-zero at every other decile. Essence stays low and roughly flat (0.0000–0.0063, exactly zero at deciles 3 and 4) across all deciles including decile 10 (0.0050) — no comparable spike anywhere.
- **At decile 1 (true cold-start), the pattern reverses: CF scores exactly 0.0000 (no co-occurrence signal for zero-train-interaction items) while Essence scores 0.0056** — Essence numerically and marginally-significantly beats CF here (P(diff>0)=0.979, one-sided; observed diff +0.0056, 95% CI [+0.0003,+0.0130]). At decile 2, no significant difference (diff +0.0016, P=0.560).
- Singleton items here concentrate in deciles 2–5 (32,32.8% in deciles 3–4 alone), not 3–9 like Amazon — dataset-specific decile geometry, reported factually.

### Genre-shortcut check — flagged as non-transferable, not a finding (`results/shortcut_analysis_movielens_summary.csv`)
99.85% of Essence's recs share a genre with the active cluster (vs. ~20% for artist/author on Last.fm/Amazon). This number is **not comparable** to the artist/author checks — genre is a coarse ~20-value categorical tag, so near-universal overlap is the expected base rate by chance, not evidence of a content shortcut. The holdout variant (Recall@10 collapses to 0.0006) mostly reflects the candidate pool being starved to near-nothing, not a meaningful ablation. Do not cite this number as comparable to the Last.fm/Amazon shortcut findings.

### TMDb data acquisition (`experiments/movielens/`)
7,724 candidate items (from 2,000 sampled users, seed=42, same convention as Amazon's `preprocess_amazon.py`), 7,654 fetched (99.1% success), 70 failed. (A prior draft attributed the 70 failures to "55×404, 5 transient network errors" — those two numbers sum to 60, not 70, and no persisted fetch log exists to reconcile or verify the split; the cause breakdown is untraceable and has been dropped rather than restated.) All 10 sanity-checked sample texts were legitimate.

---

## 4. Cross-dataset synthesis

### The "beats CF at cold-start" invariant — validated, but NOT Essence-specific

**Second update, correcting the first**: the previous version of this section validated "Essence beats CF-ItemKNN at cold-start" across all three datasets (table below is factually correct and unchanged) but implied this was evidence for Essence's architecture specifically. A follow-up scope check (`results/coldstart_invariant_scope_check.csv`, `results/coldstart_invariant_scope_check_extended.csv`) shows **this is wrong** — Recency-Weighted, Last-Item, and Avg-Last-10 all show the *same* pattern, just as strongly or more so:

| Dataset | Decile | Essence | CF-ItemKNN | Diff | Cohen's d | q (FDR) | Significant? |
|---|---|---|---|---|---|---|---|
| Last.fm | 1 | 0.0086 | 0.0000 | +0.0086 | 0.222 | 0.0003 | Yes |
| Last.fm | 2 | 0.0172 | 0.0000 | +0.0172 | 0.215 | <0.001 | Yes |
| Amazon | 1 | 0.0175 | 0.0000 | +0.0175 | 0.201 | <0.001 | Yes |
| Amazon | 2 | 0.0176 | 0.0004 | +0.0172 | 0.185 | <0.001 | Yes |
| MovieLens | 1 | 0.0056 | 0.0000 | +0.0056 | 0.081 | 0.0499 | Yes (marginal) |
| MovieLens | 2 | 0.0063 | 0.0047 | +0.0016 | 0.017 | 0.879 | No |

**But run the identical test with Recency-Weighted, Last-Item, or Avg-Last-10 in Essence's place, and 15 of 18 comparisons are also significant** — including *every* Last.fm and Amazon comparison for all three baselines (p<0.0001 throughout), and 3/3 on MovieLens decile 1. The pattern is mechanical, not architectural: CF-ItemKNN's co-occurrence score is **exactly 0.0000** for any item with zero training interactions, by construction — no method that scores items via content embeddings can fail to beat that. This says nothing about clustering, peer-isolation, or Essence's design; it says CF cannot do cold-start scoring at all, and any content-based method (Essence included) trivially clears that bar.

**Correct framing for the paper**: "Essence, like every content-embedding-based method tested, beats collaborative filtering at true cold-start" — not "Essence uniquely solves cold-start." The distinction matters for how the reframing document positions this finding (see `paper/REFRAMING_PROPOSAL.md`, corrected below).

### Domain-dependence — this IS well-supported across all three datasets

| Dataset | Essence rank (of 10 systems, Recall@10) | Strength of collaborative/popularity signal in this domain |
|---|---|---|
| Amazon Books | 4th (beats 6, loses to all 3 recency baselines) | Weak — CF/Popularity are near-bottom (CF=0.0031, Popularity=0.0021) |
| Last.fm-1K | 5th (beats 5, loses to 3 recency baselines + tied w/ ComiRec) | Weak — CF/Popularity near-bottom (CF=0.0018, Popularity=0.0023) |
| MovieLens-25M | 8th (beats only Content, Random) | **Strong** — CF/Popularity/ComiRec/MIND are the top 4 systems |

Essence's relative standing tracks inversely with how strong collaborative/popularity signal is in the domain. This pattern is consistent and well-evidenced across all three datasets (unlike the cold-start claim above).

### K-sensitivity, side by side
Last.fm: K=3 clearly suboptimal (K=4/5 much better). Amazon: K=3 roughly optimal (flat curve). MovieLens: not tested (K-sweep was Last.fm/Amazon-only per original scope).

### Effect-size magnitude, side by side
All Last.fm/Amazon effect sizes are "small" by Cohen's convention (|d|<0.3). MovieLens's Essence-losing comparisons are the only "medium"-approaching effects in the whole engagement (|d| up to 0.558).

### Statistical machinery applied consistently across all three datasets
Paired bootstrap (10,000 resamples) → BH-FDR correction (as a separate family per dataset/analysis, never pooled) → Cohen's-d-equivalent effect size → BCa cross-check for boundary cases. Same `evaluation/paired_bootstrap.py` script used unmodified across all three datasets and the decile sub-analyses.
