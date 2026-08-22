# Reframing Proposal — Draft for Review

**Status: draft only. Not applied to `essence_paper_v2.tex`. No sign-off has
been given on any of this.** Every claim below cites the exact result file it
comes from — see `results/MASTER_FINDINGS.md` for the full evidence base.

## Update #2 (corrects Update #1 below in spirit, though not deleting it):
## the cold-start pattern is real but NOT an Essence-specific finding

Update #1 (left below for the audit trail) said the cold-start invariant was
now validated and safe to use as a headline claim. **That was incomplete.**
A follow-up scope check (`results/coldstart_invariant_scope_check.csv`,
`_extended.csv`) ran the identical decile-1/2-vs-CF test with
Recency-Weighted, Last-Item, and Avg-Last-10 in Essence's place. **15 of 18
comparisons are also significant, most at p<0.0001** — every Last.fm and
Amazon comparison for all three non-Essence baselines beats CF-ItemKNN at
cold-start too. The mechanism is mechanical, not architectural:
CF-ItemKNN's co-occurrence score is exactly 0.0000 for any item with zero
training interactions, so *any* content-embedding-based method — clustering
or not — clears that bar trivially.

**Do not write this into the paper as "Essence solves cold-start" or as
evidence for clustering/peer-isolation specifically.** The defensible claim
is narrower: "Essence, like every content-based method tested, beats
collaborative filtering at true cold-start" — a statement about content
embeddings vs. co-occurrence scoring in general, not about what makes
Essence's design distinct. Separately (`results/essence_vs_recencyweighted_consolidated.csv`):
**Essence never significantly beats Recency-Weighted on Recall@10 in any of
the three datasets** — it loses significantly in all three. Any reframing
that leans on "Essence recovers cold-start / long-tail items other methods
miss" needs to be explicit that the simpler recency-only baselines recover
them too, often better.

The abstract and contributions list below (and Update #1's text) were
drafted before this correction and should not be used as-is — the whole
"domain-dependence + cold-start" framing needs a pass that is honest about
which parts are Essence-specific (the domain-dependence *ranking* pattern,
which — unlike cold-start — has no evidence yet of being generic to other
methods, since the other baselines weren't ranked against the full 10-system
field per domain the same way) and which parts are shared by any
content-based competitor (cold-start recovery).

## Update #1 (superseded by Update #2 above — kept for the audit trail)

**This section originally flagged the cold-start invariant as unsupported.
That gap has since been closed** (`results/MASTER_FINDINGS.md` §4,
`results/coldstart_invariant_cross_dataset.csv`): Essence vs. CF-ItemKNN at
decile 1 (true cold-start) is now significant, after FDR correction, on
**all three datasets** (strongly on Last.fm and Amazon, marginally on
MovieLens at q=0.0499). At decile 2 it holds on Last.fm and Amazon but not
MovieLens. The abstract and contributions list below were drafted *before*
this was confirmed and therefore undersell it — they should be revised to
state the cold-start invariant as a validated headline finding, not omitted
or hedged. I have not gone back and rewritten the draft text below to
reflect this; treat everything after this notice as needing a pass to
integrate the now-confirmed cold-start claim before you take it as final.

**[This update was itself incomplete — see Update #2 above, which
supersedes this.]**

---

## Proposed title

> **Peer-Isolated Recommendation: When Content-Only Personalization Beats
> Collaborative Signal, and When It Doesn't**

(Current title — "A Personal Embedding Cluster Approach to Long-Tail
Recovery" — commits to a universal long-tail-recovery claim that MovieLens
contradicts. The proposed title makes the domain-dependence the headline
instead of something a reader discovers three datasets in.)

## Proposed abstract

> We present Essence, a personal embedding cluster framework for
> recommendation that operates exclusively within a single user's engagement
> history, without consulting cross-user interaction data. We evaluate
> Essence on three datasets spanning distinct domains — Last.fm-1K (music),
> Amazon Books (e-commerce), and MovieLens-25M (film) — against a real
> collaborative-filtering baseline, a non-personalized popularity baseline, a
> content-based average-embedding baseline, three recency-based baselines
> (last-item, uniform-average, and exponentially-weighted retrieval), and
> two hand-implemented multi-interest neural baselines (MIND, ComiRec).
>
> Essence's performance relative to these baselines is **domain-dependent**,
> and the dependence has a specific, testable shape: Essence's rank among the
> ten evaluated systems tracks inversely with the strength of
> collaborative/popularity signal available in the domain. On Amazon Books,
> where collaborative and popularity baselines are structurally weak
> (Recall@10 of 0.0031 and 0.0021 respectively), Essence achieves the highest
> Recall@10 among personalized methods (0.0221) and beats collaborative
> filtering, popularity, content-averaging, and both neural multi-interest
> baselines with paired-bootstrap-validated, FDR-corrected significance. On
> MovieLens-25M, where collaborative signal is strong (item-based CF reaches
> 0.0850 Recall@10, driven almost entirely by the platform's densely-rated
> popular titles), Essence ranks 8th of 10 systems, losing to collaborative
> filtering, popularity, and both neural baselines by the largest effect
> sizes observed in this study (Cohen's d up to −0.56). On Last.fm-1K, simple
> recency-based retrieval (no clustering) outperforms Essence outright,
> raising a question this paper does not fully resolve: how much of Essence's
> advantage, where it exists, comes from clustering specifically versus
> recency-weighted retrieval alone.
>
> We report every comparison with paired bootstrap significance testing
> (10,000 resamples), Benjamini-Hochberg correction for multiple comparisons,
> standardized effect sizes, and BCa robustness cross-checks — replacing an
> earlier confidence-interval-overlap methodology that does not test the
> quantity of interest. We also report a popularity-decile breakdown showing
> that Essence's long-tail advantage, where present, is concentrated in
> moderately-rare-but-observed items rather than true cold-start items, and
> an artist/author-identity shortcut analysis showing that a disproportionate
> share of Essence's hits come from same-artist/author candidates despite
> these being a minority of its recommendations. We frame Essence not as a
> universally superior candidate generator, but as a data point on when
> peer-isolated, content-only personalization is and isn't the right
> architectural choice.

## Proposed introduction framing

Replace the current single motivating question ("can peer-isolated retrieval
recover long-tail items collaborative methods structurally underrepresent?")
with two questions, since the second is now the paper's actual empirical
center of gravity:

1. Can a recommendation signal be constructed using only a single user's own
   history, and does it recover items collaborative methods
   structurally underrepresent? *(original question, still addressed)*
2. **Under what domain conditions does this approach help versus hurt, and
   is the effect predictable from properties of the domain rather than
   discovered post hoc per dataset?** *(new — this is what 3 datasets
   actually let you claim that 1–2 datasets don't)*

Proposed contributions list (replacing the current 4-item list):

1. The Essence architecture (unchanged).
2. A recency-based active-cluster-selection mechanism, evaluated against a
   static mean-embedding alternative *and* against three recency-only
   baselines that don't cluster at all — the honest comparison the original
   ablation didn't include (`results/MASTER_FINDINGS.md` §1–3, Recall@10
   tables).
3. **A domain-dependence finding across three datasets** spanning music,
   e-commerce, and film: Essence's relative standing correlates inversely
   with the strength of collaborative/popularity signal in the domain,
   evidenced by effect sizes ranging from Essence's strongest win (Amazon,
   beating Random with d=+0.26) to its strongest loss (MovieLens, losing to
   CF-ItemKNN with d=−0.56).
4. Two hand-implemented, from-scratch multi-interest baselines (MIND,
   ComiRec) — RecBole doesn't ship either — trained and compared with the
   same statistical rigor as every other baseline, including a documented
   architecture bug found via a synthetic conformance test and fixed before
   any reported number (`experiments/mind_comirec/test_mind_routing.py`).
5. A statistical-methodology correction: paired bootstrap + FDR + effect
   sizes + BCa replacing confidence-interval overlap, applied uniformly
   across all three datasets and sub-analyses.

## Proposed results-section structure

Current structure is two flat per-dataset sections (§Last.fm-1K,
§Amazon Books) each ending in "Essence wins." Propose restructuring around
the domain-dependence claim directly:

1. **§5.1 Setup common to all three datasets** — same chronological split
   logic, same embedding model, same 10-system comparison, same statistical
   pipeline. State once, don't repeat three times.
2. **§5.2 Headline results, all three datasets, one table** — the
   `results/MASTER_FINDINGS.md` §1–3 headline tables, side by side, with
   Essence's rank (not just its number) visible in each column.
3. **§5.3 The domain-dependence pattern** — the inverse-correlation claim,
   with the three datasets' CF/Popularity baseline strength plotted or
   tabulated against Essence's rank. This is the section that makes the
   paper's contribution legible; currently this pattern is not stated
   anywhere because MovieLens didn't exist yet.
4. **§5.4 Statistical validation** — paired bootstrap, FDR, effect sizes,
   BCa cross-checks, presented once as methodology, then referenced per
   dataset rather than re-explained.
5. **§5.5 Where the long-tail advantage actually comes from** — the
   popularity-decile / cold-start-vs-singleton distinction
   (`results/longtail_definition_reconciliation.csv`,
   `README.md`'s "Cold-start vs. singleton" section) — this is a genuine
   scope-narrowing finding and reads as rigor, not weakness, if presented
   directly.
6. **§5.6 Limitations, stated as findings, not caveats** — the artist/author
   shortcut result (real, disclosed, quantified — not swept into a single
   "limitations" bullet), the Last.fm sample-size fragility (quantified via
   seed variance, not just asserted), and the unresolved
   clustering-vs-recency question from the abstract.

## What this reframing does *not* claim

- It does not claim Essence is a bad idea or that the paper shouldn't be
  published — the Amazon result is real, significant, and survives every
  robustness check applied to it (FDR, BCa, effect size).
- It does not claim the domain-dependence mechanism is fully explained —
  "collaborative signal strength" is measured post hoc via the baselines'
  own performance, not from an independent domain property. A reviewer could
  reasonably ask for a more principled domain characterization; this draft
  doesn't attempt to invent one.
- It does not include the cold-start invariant claim from the original
  brief, for the reason stated at the top of this document.
