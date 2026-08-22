# Reviewer Coverage Audit — RecSys 2026 R&P Notes, Submission 1165

Source: the actual decision email and 3 reviews for "Peer-Isolated Recommendation:
A Personal Embedding Cluster Approach to Long-Tail Recovery" (weak reject:
Review 1 = Accept(2), Review 2 = Weak Reject(-1), Review 3 = Weak Reject(-1)).
Every concern below is quoted or closely paraphrased directly from the review
text supplied by the user — nothing summarized from memory of this engagement.
"Closed" requires a specific results file that directly answers the concern;
"Partially closed" means real work was done but a gap remains (stated
explicitly); "Open" means nothing in this engagement addressed it.

---

## Metareview

| Concern | Status | Justification |
|---|---|---|
| "Missing recent-content baselines" | **Closed** | `results/paired_bootstrap_lastfm_mind_comirec.csv`, `paired_bootstrap_amazon_mind_comirec.csv` — Last-Item, Avg-Last-10, Recency-Weighted implemented and tested on all 3 datasets. |
| "Limited and uneven empirical evidence" | **Partially closed** | A third dataset (MovieLens-25M, `results/MASTER_FINDINGS.md` §3) was added, substantially broadening the evidence base. But the specific unevenness the reviewers flagged — Last.fm's thin long-tail signal — was never fixed, only more rigorously *quantified* (see Review 1 §3 below). |
| "Insufficient positioning against related multi-interest methods" | **Closed** | MIND and ComiRec hand-implemented (RecBole doesn't ship either — verified against PyPI + GitHub master), trained, and statistically compared on all 3 datasets. `experiments/mind_comirec/`. |
| "Weak reproducibility" | **Partially closed** | `requirements.txt` now pinned; README documents exact repro commands for every table; extensive new scripts exist for every analysis. But whether this repo is actually attached to/released with a resubmission is a packaging decision outside this engagement's scope — see Review 2 §1 below. |

---

## Review 1 (Score: 2, Accept — leaning accept with minor revisions)

| # | Concern (quoted/paraphrased) | Status | Justification |
|---|---|---|---|
| W1 | "MIND and ComiRec are cited but neither compared against nor clearly distinguished from Essence... should better position its contribution and, if feasible, include these as baselines." | **Closed** | Both implemented, trained (4 configs initially, expanded to 14 MIND reruns after a routing bug fix), and compared via paired bootstrap + FDR + BCa on all 3 datasets. **Caveat**: found and fixed a real bug in the MIND implementation (routing logits zero-initialized, causing capsule collapse) — see `experiments/mind_comirec/test_mind_routing.py`. All reported MIND numbers are post-fix. |
| W2 | "Key design choices (K=3, r=10) are fixed without sensitivity analysis, and no variance across K-means random initializations is reported." | **Partially closed** | K-sensitivity: `results/sensitivity_results.csv` (Last.fm), `results/sensitivity_results_amazon.csv` (Amazon) — K=3 is *not* near-optimal on Last.fm (K=4/5 clearly better), roughly optimal on Amazon. Seed variance (K-means random init): same files, 10 seeds each, both datasets — variance is small relative to the K-sweep spread, so K=3 was a stable-but-suboptimal choice on Last.fm. **Gap**: `r=10` (the recency-window size) itself was never swept — only K was. |
| W3 | "Last.fm-1K results are based on only two long-tail hits, making Amazon Books the primary source of empirical support." | **Open** (quantified, not fixed) | Verified directly: `results/evaluation_results_v5.csv` — Essence has exactly 2 users with LT-Recall@10 > 0 out of 92 eligible on Last.fm. This is still true today; Last.fm's user count was never increased. The sensitivity/seed-variance work (above) makes the instability of this specific number more visible, but doesn't resolve the underlying sparsity. |
| S1 | "Broader empirical validation across additional datasets." | **Closed** | MovieLens-25M added as a third dataset — full baseline suite, MIND/ComiRec, paired bootstrap + FDR + BCa, decile breakdown, shortcut check. `results/MASTER_FINDINGS.md` §3. |
| S2 | "Adding comparison against a wider set of related methods would substantially strengthen confidence in the generality of results." | **Closed** | Same as W1 + the 3 recency baselines + real ItemKNN CF (already present pre-engagement). |
| S3 | "Paper simply lists references at the end... add relevant citations in the text." | **Open** | This is a paper-formatting/writing concern; nothing in this engagement touches paper text (explicit constraint throughout). Needs to be done during the actual paper-editing pass. |

## Review 2 (Score: -1, Weak Reject)

| # | Concern (quoted/paraphrased) | Status | Justification |
|---|---|---|---|
| W1 | "No code or preprocessing scripts are provided, and several implementation details are missing." | **Partially closed** | The repo now contains complete preprocessing/embedding/evaluation code for all 3 datasets, MIND/ComiRec implementations, and every statistical script — far more complete than what the reviewer saw. **Gap**: whether this code is actually linked/released alongside a resubmission is a submission-packaging decision, not something this engagement can close on its own. |
| W2 | "The statistical comparison is not valid. Essence having a CI above zero does not show it beats the content baseline. Its mean falling outside another method's CI is also not a test of the difference." | **Partially closed — real gap, be precise about it** | `evaluation/paired_bootstrap.py` was built specifically to fix this (paired bootstrap on the per-user difference, not independent CIs) and is now used throughout: `results/paired_bootstrap_*.csv` for all 3 datasets, plus FDR correction (`results/fdr_corrected_significance*.csv`) and BCa cross-checks (`results/bca_*_check.csv`). **However — verified directly against the current `.tex`**: the paper's abstract (line 88) and CI table narrative (lines 606–610, 916, 922) **still exclusively use the "overlapping CIs" framing** the reviewer explicitly flagged as invalid. The correct statistical machinery exists and has been run; it has not been propagated into the paper text. This is the single most direct, still-live instance of a reviewer's core objection being unaddressed in the actual manuscript. |
| W3 | "The most important baselines are missing: last item, average of last ten, recency-weighted average. Without these it is unclear whether clustering helps at all." | **Closed, and the answer is uncomfortable** | All 3 implemented (`models/recommenders.py`: `last_item_recommend`, `avg_last10_recommend`, `recency_weighted_recommend`) and tested on all 3 datasets. The reviewer's implicit hypothesis was largely **confirmed, not refuted**: these baselines beat Essence on raw Recall@10 on Last.fm (all 3) and Amazon (Last-Item), and match Essence's own performance level on MovieLens. See `results/MASTER_FINDINGS.md` §1–3. This needs honest treatment in any resubmission, not a "we added the baselines and still win" framing — the results don't support that framing on 2 of 3 datasets. |
| W4 | "Track name plus artist and book title plus author may create an easy shortcut. The method may mainly retrieve another item by the same artist or author." | **Closed as an investigation — concern confirmed as partially real** | `evaluation/shortcut_check.py` (Last.fm), `experiments/amazon_books/shortcut_check_amazon.py` (Amazon): ~20% of Essence's recs share an artist/author with the active cluster, and excluding those candidates collapses both Recall@10 and LT-Recall@10 by 4–8x — same-artist/author candidates account for a disproportionate share of actual hits despite being a minority of recommendations. The reviewer's concern has real teeth and should be disclosed, not dismissed. (MovieLens's genre-based version of this check is flagged as non-transferable — genre's low cardinality produces near-universal overlap by chance, unlike a specific artist/author name — see `results/shortcut_analysis_movielens_summary.csv`.) |
| R-body | "Essence's main new point is the removal of cross-user interaction signals, not the use of multiple interests itself... a recent-item embedding or average of last ten may produce the same or better result." | **Directly investigated; findings support the reviewer** | See W3. On Last.fm and Amazon, simple recency baselines frequently match or beat Essence's clustering mechanism. |
| R-body | "Some claims are too strong... 'statistically substantiated,' 'structurally miss,' 'peer-isolation property' make the contribution sound broader than the evidence supports." | **Open** | Writing/tone concern about the paper text; not something an empirical engagement resolves. Relevant to Step D's reframing proposal (`paper/REFRAMING_PROPOSAL.md`) but not yet applied to the `.tex`. |

## Review 3 (Score: -1, Weak Reject)

| # | Concern (quoted/paraphrased) | Status | Justification |
|---|---|---|---|
| W1 | "Two datasets are discussed, however only results for one of them are presented, despite there being sufficient space." | **Closed in the current draft** | Verified directly against `.tex`: both Last.fm-1K (§ line 566, Table at 574) and Amazon Books (§ line 614, Table at 620) have full result tables present in the current `essence_paper_v2.tex`. Unclear whether this was already fixed before this engagement started or reflects a version the reviewer didn't have — either way, current state is closed. |
| W2 | "Averaging the last ten interactions' embeddings is problematic... recent interactions are typically expected to contribute with different weights rather than equally." | **Closed as an investigation — concern has merit** | Directly tested via `recency_weighted_recommend` (exponential decay, not uniform averaging) on all 3 datasets. Recency-Weighted frequently *outperforms* both Essence and the uniform Avg-Last-10 baseline (e.g. Last.fm: Recency-Weighted 0.0304 vs Avg-Last-10 0.0252 vs Essence 0.0119) — directly validates the reviewer's intuition that uniform weighting is suboptimal. |
| W3 | "The empirical evaluation could be strengthened further." | **Closed** (as far as a generic ask can be) | FDR correction, effect sizes, BCa cross-checks, multi-cutoff (@5/10/20), popularity-decile breakdown, compute-cost disclosure, a third full dataset, K-sensitivity, seed variance — see `results/MASTER_FINDINGS.md` for the complete inventory. |
| R-body | "With respect to baselines, it is good to compare to other embeddings (Matryoshka/Semantic IDs/etc)." | **Open** | Not attempted anywhere in this engagement — every dataset uses `all-MiniLM-L6-v2` sentence-transformer embeddings throughout, no alternative embedding scheme was tested. |

---

## Summary

| Status | Count |
|---|---|
| Closed | 8 |
| Partially closed | 5 |
| Open | 5 |

**The single most important open-but-fixable item**: Review 2's core statistical-validity objection has a working fix (`paired_bootstrap.py` + FDR + BCa) that has never been applied to the paper text itself — the abstract and CI table still use the exact "overlapping CIs" framing that was called out as invalid. This is the highest-leverage fix available before any resubmission, and it's a `.tex` edit, which stays gated per your standing instruction.
