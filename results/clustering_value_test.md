# Clustering Value Test — does clustering help conditional on taste multi-modality?

Essence (clustering) vs. Recency-Weighted (no clustering, same embeddings, same recency emphasis),
stratified by per-user silhouette score under each user's own K=3 fit (tertiles, computed per dataset).
FDR family size: 18 (3 datasets x up to 3 strata x 2 metrics).


## Last.fm-1K

| Stratum | Metric | n | Essence | Recency-W | Diff | Cohen's d | p | Sig. (FDR) |
|---|---|---|---|---|---|---|---|---|
| high | LT-Recall@10 | 29 | 0.0187 | 0.0402 | -0.0215 | -0.173 | 0.4746 | No |
| low | LT-Recall@10 | 32 | 0.0000 | 0.0000 | +0.0000 | +nan | 0.0000 | Yes |
| medium | LT-Recall@10 | 31 | 0.0000 | 0.0073 | -0.0073 | -0.318 | 0.0000 | Yes |
| high | Recall@10 | 33 | 0.0309 | 0.0660 | -0.0351 | -0.315 | 0.0002 | Yes |
| low | Recall@10 | 33 | 0.0007 | 0.0045 | -0.0038 | -0.241 | 0.1106 | No |
| medium | Recall@10 | 33 | 0.0041 | 0.0207 | -0.0165 | -0.402 | 0.0068 | Yes |

## Amazon Books

| Stratum | Metric | n | Essence | Recency-W | Diff | Cohen's d | p | Sig. (FDR) |
|---|---|---|---|---|---|---|---|---|
| high | LT-Recall@10 | 371 | 0.0469 | 0.0470 | -0.0001 | -0.001 | 0.9822 | No |
| low | LT-Recall@10 | 483 | 0.0053 | 0.0038 | +0.0015 | +0.030 | 0.6230 | No |
| medium | LT-Recall@10 | 438 | 0.0126 | 0.0088 | +0.0037 | +0.037 | 0.4436 | No |
| high | Recall@10 | 667 | 0.0534 | 0.0617 | -0.0084 | -0.080 | 0.0368 | No |
| low | Recall@10 | 667 | 0.0043 | 0.0057 | -0.0014 | -0.066 | 0.0706 | No |
| medium | Recall@10 | 666 | 0.0087 | 0.0106 | -0.0019 | -0.044 | 0.2490 | No |

## MovieLens-25M

| Stratum | Metric | n | Essence | Recency-W | Diff | Cohen's d | p | Sig. (FDR) |
|---|---|---|---|---|---|---|---|---|
| high | LT-Recall@10 | 106 | 0.0000 | 0.0047 | -0.0047 | -0.097 | 0.0000 | Yes |
| low | LT-Recall@10 | 189 | 0.0000 | 0.0066 | -0.0066 | -0.088 | 0.0000 | Yes |
| medium | LT-Recall@10 | 195 | 0.0051 | 0.0026 | +0.0026 | +0.032 | 0.9130 | No |
| high | Recall@10 | 667 | 0.0086 | 0.0133 | -0.0046 | -0.096 | 0.0094 | Yes |
| low | Recall@10 | 667 | 0.0025 | 0.0041 | -0.0015 | -0.064 | 0.0936 | No |
| medium | Recall@10 | 666 | 0.0032 | 0.0055 | -0.0022 | -0.083 | 0.0260 | No |