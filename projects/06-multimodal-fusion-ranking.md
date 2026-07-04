# Project 06 — Multimodal Similarity Fusion and Retrieval Ranking

## Problem

Restaurant text and recipe images are retrieved from separate models and vector
spaces. Their raw distances cannot be compared directly. Lab 06 creates a
transparent mixed ranking by normalizing each candidate pool independently and
applying configurable modality weights.

## Fusion algorithm

1. Retrieve the top restaurant articles with a normalized 384-D text query.
2. Retrieve the top recipe images with a normalized 512-D CLIP text query.
3. Convert cosine distance to similarity with `similarity = 1 - distance`.
4. Min-max normalize similarities within each modality.
5. Multiply normalized text scores by `w_text`.
6. Multiply normalized image scores by `w_image`.
7. Combine and sort all candidates by fused score.

Weights are normalized to sum to one. Negative weights and an all-zero weight
configuration are rejected.

## Edge cases

If one filtered pool contains a single candidate, or every candidate has the
same similarity, ordinary min-max scaling has a zero denominator. Lab 06 assigns
those valid candidates a normalized score of 1.0. Empty pools remain empty.

Strict metadata filters may return fewer than `k` candidates. Fusion proceeds
with the records that actually matched instead of fabricating or padding rows.

## Three demonstrations

- **Demo 1:** unfiltered restaurant and image fusion.
- **Demo 2:** location-filtered articles plus source-filtered images.
- **Demo 3:** article-heavy, balanced, and image-heavy weight settings.

Every row exposes its text score, image score, and final weighted score so the
reranking behavior can be audited.

## Interpretation

This lab mixes two candidate types; it does not claim that a restaurant article
and recipe image represent the same entity. Entity-level fusion would require a
shared identifier or learned cross-domain relationship. Here, fusion is best
understood as weighted interleaving for a recommendation feed.

## Run

Build the Lab 04 index, then open
`examples/06_multimodal_similarity_fusion.py` as notebook cells.
