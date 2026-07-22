# Learned Coordinate Space for CLUE

A design + training spec for replacing raw detector coordinates with a learned
low-dimensional space in which a **single** CLUE parameter set clusters both
overlapping EM showers and large irregular hadronic showers.

---

## 1. Problem & core idea

**Pipeline today:** `coords/raw.py` passes raw `(x, y, z)` + energy weights into
CLUEstering, tuned by `scripts/tune_clue.py`, scored by `metrics/physics.py`.

**Why raw space is fundamentally limited:** CLUE is a density-based clusterer
with *one* set of parameters `(dc, rhoc, dm, do)`. In raw detector space no single
density scale can simultaneously:

- keep two **overlapping EM showers** apart (needs a tight scale), and
- hold a **large, irregular hadronic shower** together (needs a loose scale).

These are opposite requirements. This is not a tuning failure — it is intrinsic
to clustering objects of heterogeneous size/shape/density at one scale.

**The idea:** train a network `f: hits → ℝ³` that **renormalizes every true object
— whatever its native size, shape, or density — into a standardized blob**: the
same characteristic radius, the same inter-blob gap, one density peak each. In
that homogenized space, one CLUE threshold fits every object type.

**The network is a drop-in `coords/` producer.** Everything downstream (CLUE,
metrics, tuning) is reused unchanged. Energy is **not** a coordinate — it passes
straight through as the CLUE weight.

**Objective (falsifiable):** learned space beats `raw.py` on `metrics/physics.py`,
with the gain concentrated on **overlapping** and **hadronic** objects, at a
single shared parameter set. If learned *loses* to raw on isolated,
non-overlapping objects, something is broken (raw is already fine there).

**Why a transformer (not raw CLUE, not a local GNN):** recognizing that a
sprawling, gappy set of hits is *one* hadronic shower is long-range,
content-based reasoning. Raw density cannot do it (that is the baseline's
failure); a local kNN GNN (k≈20, few layers) has too small a receptive field and
oversmooths before it spans the object. Attention is the mechanism for long-range
grouping.

---

## 2. Inputs

Available per node (layer cluster) today: `x, y, z, E, eta, phi, n_hits`.

Feature vector fed to the network (~9 dims), all derivable now:

```
[ x/100, y/100, z/400,          # position; z encodes longitudinal depth = EM/hadronic handle
  log1p(E),                     # energy
  eta, sin(phi), cos(phi),      # direction; sin/cos removes the φ wrap discontinuity
  log1p(n_hits),                # cluster size
  log(E / n_hits + eps) ]       # energy density per rechit: compact-EM vs diffuse-hadronic discriminator
```

`E` is **also** kept aside unchanged as the CLUE `weight`. No new preprocessing
needed. Future features (tracker tracks, hit time, L1 seeds) append to this
vector without structural change; time is the sharp pile-up handle.

---

## 3. Model — regional geometric transformer

```
Input MLP:      Linear(9 → 128)
Geometry bias:  MLP( pairwise Δ[x, y, z, eta, Δφ_wrapped] → per-head scalar ) added to attention logits
Encoder:        6 × pre-RMSNorm block:
                  MHSA(d=128, 8 heads, QK-norm, + geometry bias)
                  SwiGLU FFN (128 → 256 → 128)
                  DropPath 0.1, residual
Output head:    RMSNorm → Linear(128 → 3)      # raw Euclidean ℝ³, NO L2-normalization
```

**Design rationale**

- **Output dim = 3.** CLUE is density-based; high-D density is meaningless (curse
  of dimensionality). 3 is enough to pack a few standardized blobs. Revisit for
  high-multiplicity real data.
- **No output normalization.** CLUE needs absolute Euclidean scale; the fixed loss
  margins (§4) set that scale.
- **Geometry bias** is the single highest-value inductive bias (Point-Transformer
  / ParT relative-position attention, taken from the CS source).
- **Modern block** (RMSNorm, SwiGLU, QK-norm, pre-norm) — free, well-replicated
  wins over the 2017 encoder; QK-norm controls attention-logit blow-up and reduces
  warmup sensitivity.
- **Attention scope:** toy case = **global** attention over ~700 nodes (trivial
  cost). Real data (~500k nodes) = **L1-seeded regional** attention, same block,
  restricted scope. Deferred; the design has the slot.

---

## 4. Loss — energy- and purity-weighted discriminative embedding

Per event with truth objects `c = 1..C`, embeddings `z_i`, node weights `w_i`
(defined in §5):

```
μ_c   = Σ_{i∈c} w_i z_i / Σ_{i∈c} w_i                         # weighted center
L_var = (1/C) Σ_c  [ Σ_{i∈c} w_i · relu(‖z_i − μ_c‖ − δ_v)² / Σ_{i∈c} w_i ]
L_dist= (1/(C(C−1))) Σ_{c≠c'} relu(2·δ_d − ‖μ_c − μ_c'‖)²
L_reg = (1/C) Σ_c ‖μ_c‖
L     = L_var + L_dist + 1e-3 · L_reg
```

**How each term maps onto CLUE's requirements**

- **`L_var` with fixed radius `δ_v` = standardization + unimodality.** It forces
  *every* object — tiny EM, huge hadronic — into a blob of the **same** radius
  around **one** center. This is what stops CLUE from (a) merging overlapping EM
  (now `≥ 2·δ_d` apart) and (b) fragmenting hadronic showers (contracted to one
  tight unimodal blob). No separate unimodality term needed — the hinged
  single-center pull *is* it.
- **Weighting by `w_i`** makes `μ_c` the energy-weighted center and pulls
  high-energy hits hardest, so the latent **density peak coincides with the energy
  core** — exactly where CLUE's energy-weighted seed lands.
- **`L_dist` margin `δ_d`** guarantees a density valley between objects, so CLUE's
  δ-decision fires and seeds separate.

Hinged terms: `L_var` only penalizes points beyond `δ_v` (blobs of radius `δ_v`
are free); `L_dist` only penalizes centers closer than `2·δ_d`.

This is still contrastive in spirit (attract same-object hits, repel different
centers), in the **Euclidean-margin** dialect — chosen because it (a) homogenizes
EM/hadronic scale, (b) gives CLUE an absolute density scale, and (c) extrapolates
to arbitrary multiplicity. A cosine-softmax (SupCon) loss has no absolute scale,
so it is not the backbone.

---

## 5. Node weights — handling fractional (shared-hit) truth

Truth gives each node energy fractions across showers (e.g. shower 0 @ 0.7,
shower 1 @ 0.3). A shared hit *is* the overlap physics. Handle it as follows.

**Assignment: hard (argmax).** CLUE produces a hard partition and needs clean
density valleys between showers. Placing shared hits *between* blobs (soft
placement) fills the valley and **merges** the showers — the exact failure we are
preventing. So boundary hits must be **committed** to one side, not left floating.

**Weight: energy × purity.** Let `p_i` = the node's max energy fraction (your
existing `argmax_fraction`; if top-2 fractions are stored later, use the margin
`f₁ − f₂` for a sharper signal). Define

```
w_i = E_i · φ(p_i),     φ(p) = relu( (p − p0) / (1 − p0) ),   p0 ≈ 0.5–0.6
```

Apply it as:

- **Centroids (`μ_c`) and the push term** are strongly purity-weighted → cluster
  centers and margins are defined by clean, energetic hits, immune to boundary
  noise.
- **A shared hit's own pull** keeps a *small but nonzero* weight to its argmax
  center — enough to commit it to one side (out of the valley), not enough for a
  50/50 coin flip to dominate the gradient. Do **not** zero it out, or the hit
  floats into the gap and re-merges the blobs.

**Why not hard argmax with equal weights:** a 0.51/0.49 node assigned by a coin
toss then gets a large, confident pull/push built on noise, distorts `μ_c`, and
inflates `L_var` — worst exactly in the overlap regime we care about.

This mirrors energy-weighted physics scoring (a 50/50 hit contributes half its
energy to each shower, so mis-assigning it is cheap). Start today with
`argmax_fraction`; no reprocessing required.

**Reuse the fractions beyond the weight:**

- **Curriculum / sampling:** events with many hits at `p_i ≈ 0.5` are the hard
  overlap cases — schedule later, oversample once stable.
- **Evaluation honesty:** stratify metrics by hit purity; do not over-penalize
  mis-assignment of genuinely ambiguous low-purity hits.
- **Optional future head:** predict `p_i` (a "shared-ness" score) to flag hits
  whose energy should be split in a post-CLUE energy-sharing step. Not now.

---

## 6. Margin ↔ CLUE parameter coupling

The loss geometry and the clusterer are configured as **one system**. Work in
CLUE units with `dc = 1`.

| Quantity | Symbol | Start value | Reason |
|---|---|---|---|
| Blob radius | `δ_v` | `0.5` | whole object within ~one density radius → dense, unimodal |
| Half min center gap | `δ_d` | `1.5` | centers ≥ `3·dc`; empty surface gap ≥ `2·dc` → clear valley |
| CLUE density radius | `dc` | `1.0` | the unit; blob dense within it |
| CLUE seeding distance | `dm` | `~2.0` | between gap and center distance → one seed per blob |
| CLUE outlier distance | `do` | `~1.0` | ≈ `dc` |
| CLUE min density | `rhoc` | tune to weight/energy scale | seed vs outlier threshold |

These are a **coupled starting point**. After training, run
`scripts/tune_clue.py` on the learned space to lock the final
`(dc, rhoc, dm, do)`. Metric: `euclidean` (output is plain ℝ³; energy is the
weight channel).

---

## 7. Training procedure

**Optimizer / schedule**

- AdamW, lr `3e-4`, betas `(0.9, 0.95)`, weight_decay `0.05` (weights only, not
  norms/bias).
- 5% linear warmup → cosine decay.
- grad-clip `1.0`.
- **bf16** autocast (drop the fp16 `GradScaler`).
- Weight EMA for evaluation.
- **DropPath 0.1** replaces feature dropout.

**Batching:** PyG, ~16–32 events/batch, loss computed per event then averaged
(as the existing loss does).

**Curriculum** (overlap = min inter-CP centroid distance in raw space; hardness
also from `p_i ≈ 0.5` density):

Epochs ~100 (tune to convergence of the CLUE-metric selection below).

**Checkpoint selection — on the REAL objective, not the surrogate loss.**
The margin loss going down does **not** guarantee CLUE improves. Every `K`
epochs:

1. embed the validation set,
2. run CLUE (coupled params from §6, small grid) on the embeddings,
3. compute `metrics/physics.py`,
4. keep the checkpoint maximizing a combined score, e.g.
   `efficiency × purity − fake_rate`.

---

## 8. Evaluation — the milestone

Reuse `clustering/clue.py` + `metrics/physics.py` + `scripts/tune_clue.py`
unchanged. Compare:

- **Baseline:** `raw.py` → best-tuned CLUE.
- **Learned:** `embed.py` → best-tuned CLUE.

Report metrics **stratified by**:

- object type (EM vs hadronic),
- overlap (isolated vs overlapping),
- multiplicity (`n_cp`),
- hit purity (see §5).

**Two claims to prove**

1. Learned beats raw overall on efficiency / purity / fake rate / energy
   response.
2. **A single learned-space parameter set handles EM and hadronic
   simultaneously** — which raw space provably cannot — with the largest gains on
   overlapping + hadronic objects.

**Sanity check:** on isolated, non-overlapping objects, learned should at least
match raw. Losing there means the embedding is destroying easy structure.

---

## 9. Repo integration

- **New** `caloembed/coords/embed.py`:
  `transform(event) -> (model(features(event)), event.weights)` — loads a trained
  checkpoint; identical signature to `raw.py`, so `run_pipeline.py`, `clue.py`,
  `physics.py`, `tune_clue.py` are reused untouched.
- **In `training/`:** register the model (`@register_model("geo_transformer")`)
  and loss (`@register_loss("discriminative")`); add the geometry-bias attention,
  bf16, DropPath, EMA, curriculum, and the CLUE-metric checkpoint-selection hook.
  Changes to `data.py` / `loop.py` go behind config flags (surgical).

---

## 10. Forward-compatibility hooks (noted, not built)

- **Pile-up:** unlabeled low-energy hits → treat as an *ignore/noise* class in the
  loss (no attract/repel); CLUE's energy-weight + outlier logic demotes them.
  Later, train with overlaid pile-up for domain match.
- **500k nodes:** L1-seeded regional attention (window ≈ largest object); embed
  only regions of interest.
- **Extra CMSSW features:** tracker tracks, hit time (pile-up handle) append to
  the §2 vector.
- **Learned weight:** optionally replace the raw-E CLUE weight with a learned
  "objectness" scalar (object-condensation-style β). Future.

---

## Summary

A low-dimensional geometric transformer trained with an energy- and
purity-weighted, fixed-margin discriminative loss whose margins are coupled to
CLUE's four parameters, so that EM and hadronic showers are renormalized into
standardized blobs that one CLUE threshold can cluster — committing shared hits to
one side to keep density valleys clean, and **selected and tuned on real CLUE
physics metrics, not on the loss**.
