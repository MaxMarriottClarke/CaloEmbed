# Systematic raw-CLUE baseline

How the raw (x, y, z + energy) CLUE baseline is tuned, and why it is set up this
way. The learned-coordinate path is in [learned-coords-for-clue.md](learned-coords-for-clue.md).

## What this replaces

The first raw tune (`results/param_tuning/raw_noPU_d5_10k`, `configs/tune.yaml`)
produced the working point still baked into `configs/raw.yaml`:
`dc=1.1784, rhoc=2.2580, do=3.9951, dm=2.5715, z_scale=0.1`. Four problems with
how it was obtained:

1. **Biased event set.** `n_events: 5000` reads the first 50 files in filename
   order, and each file holds ~100 events at one fixed `n_cp`. That gave
   `{3:13, 4:10, 5:9, 6:8, 7:6, 2:4}` files — tilted toward the easy,
   low-multiplicity end.
2. **Half the front was degenerate.** 30 of 61 Pareto points were
   over-fragmentation (`pen_ratio` 100–143, `min_eff` ~0.12): formally
   non-dominated, physically useless, and consuming swarm budget.
3. **Bounds pressed at the edge.** `z_scale` sat at its 0.1 floor in 22/61
   points (15/31 even among feasible ones) and `dc` at its 0.01 floor in 38/61.
   The chosen working point is itself bound-limited in `z_scale`.
4. **No held-out set.** The front was scored on the same 5000 events the swarm
   optimised over.

It also had not converged — best `pen_ratio` was still falling at iteration 55.

## The setup now

`configs/tune_raw_d5_strat.yaml`, run through `caloembed-tune`.

**Objectives are deliberately unchanged** — mean per-event `max` reco-to-sim
score (min), mean per-event `min` CP efficiency (max), mean penalised
`n_reco/n_sim` (min). They are per-event extrema and therefore noisy, and they
are not what the physics plots report. Both true; changing them would make this
front incomparable to `transformer_clue_d5` and `gnn_clue_d5`, which is the
whole point of producing it. The noise is attacked through the event set, the
feasibility cap and the denser search instead, and `caloembed-eval-front`
reports mean purity/efficiency/ratio as diagnostics alongside.

**Stratified selection.** 4600 events / 46 files, weighted hard toward high
multiplicity: `{3: 100, 5: 800, 6: 1500, 7: 2200}`. `data.select_seed` is pinned
to 42 and never varies, so a multi-seed run varies only the swarm.

**Feasibility cap** (`objectives.ratio_cap: 10.0`). A point whose mean penalised
ratio exceeds the cap is reported as worst-possible purity *and* efficiency, so
any feasible point dominates it and it cannot reach the front — while the true
ratio stays in the third slot, leaving the swarm a gradient back into
feasibility rather than a flat plateau. With `under_cluster_penalty: 1000` this
constrains both failure modes: an under-clustered event contributes ~144, so
roughly 6% of events merging CPs also trips the cap. That is intended.

**Re-centred bounds + log sampling.** Bounds now bracket the old front's
feasible region with headroom; the box is ~80× smaller at the same 40 particles.
`dc`, `rhoc`, `do`, `dm` are searched in log10 space (`log: true`) because they
span three decades and MOPSO samples its box uniformly — under linear bounds
almost nothing below `dc=0.5` was ever tried, yet there is a real feasible mode
at `dc→0` (density collapses to the LC's own energy, so `rhoc` seeds on energy
alone and `dm` does the linking) that produced the old front's best
`min_efficiency` of 0.75. `z_scale` stays linear on [0.1, 0.5]: 0.1 is a
physical floor and the feasible front never exceeded 0.31.

Bounds and defaults in the config are always **linear**; only the swarm's
coordinates are logarithmic. `pareto_front.parquet` is linear. The raw patatune
history CSVs use `log10_`-prefixed column names so they cannot be misread.

## Running it

```bash
# H100/H200 pool (preferred: ~7h20m/seed, uncontended)
./make_hx3_bundle.sh                                  # re-run after ANY code/config/data change
condor_submit condor/tune_hx3.sub seeds="42 43 44"

# legacy lxb/lxc GPU pool (~13h/seed)
condor_submit condor/tune.sub seeds="42 43 44"
```

Then, on the merged non-dominated union of the three fronts:

```bash
caloembed-eval-front results/param_tuning/raw_d5_strat_hx3_s42 --seed 137 \
    --extra-point "dc=1.1784,rhoc=2.2580,do=3.9951,dm=2.5715,z_scale=0.1"
```

The `--extra-point` re-scores the *old* working point on the same held-out
events. Without it the comparison is unfair in the other direction: the new
validation set is deliberately harder than what the old point was tuned on, so
its absolute numbers will look worse even if it is genuinely better.

Pick the working point on the `val_*` columns, put it in `configs/raw.yaml`,
then `caloembed-run` over `d5/100k_mix_sep` and `d10/100k_mix_sep`.

## Success criteria

1. No front point with `pen_ratio > 10`.
2. No parameter pinned at a bound in >20% of front points, `z_scale` at 0.1
   excepted. If one is, widen and re-run — otherwise the bound-limiting problem
   has just been moved, not fixed.
3. Validation objectives within ~10% of tuning objectives at the chosen point.
4. Chosen point beats the old one on ≥2 of 3 objectives, **both scored on
   validation**.
5. New d5 + d10 dataframes and physics plots.

## The H100/H200 pool (hep-hx3-batch)

These nodes are **containers with no shared filesystem** — no `/vols`, no shared
`$HOME`, no CVMFS. `initialdir`, micromamba and the HDF5 input are all invisible
there, so `condor/tune.sub` cannot be pointed at them by adding `+HX3 = true`.
They do have: 252GB local scratch, network egress, AlmaLinux 9.8 / glibc 2.34
(same as the submit node), 16 CPUs, 128GB RAM, one H200 (sm_90, 47GB).

`make_hx3_bundle.sh` + `condor/tune_hx3.sub` ship everything in instead
(~505MB/job): the 1.1GB `caloembed-tune-slim` env, the repo, the selected HDF5
files, and `libboost_atomic.so.1.75.0`. That last one is the *only* shared
library the CLUE CUDA binary needs that the image lacks — its CUDA runtime is
statically linked, so no `libcudart` is required.

`stream_output`/`stream_error` are on: without them a file-transfer job's stdout
is held on the execute node until it exits, so a 7h tune shows nothing at all.

**`caloembed-eval-front` does not run here** — it needs the full 1000-file
directory to draw a validation set disjoint from the tuning files. Run it on the
legacy pool or interactively; it only takes ~45 min.

## Rebuilding CLUEstering

```bash
condor_submit condor/build_clustering.sub
```

Not on a login node: those cap the user slice at 16 GiB
(`/sys/fs/cgroup/user.slice/user-*.slice/memory.max`) and nvcc's `cicc` needs
more than that for the alpaka CUDA translation unit at four architectures. It
fits for `ARCHS="60;70;86"` and gets SIGKILLed the moment sm_90 is added, with a
misleading `'cicc' died due to signal 9` and no mention of memory.

The legacy pool also rations memory by core (`RequestMemory <= 12085 *
RequestCpus`) and caps `MaxRuntime` at 10800s, which is why that submit file
asks for 8 CPUs / 64GB / 3h.

Current binary: `CLUE_GPU_CUDA.cpython-311-*.so` embeds sm_60, sm_70, sm_86,
sm_90. The **cpython-312** binary (used by the `caloembed-py312` training env)
is still at three archs — rebuild it there too before running any py312 job on
an H200.
