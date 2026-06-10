"""3D event display:

Produces three datasets:
  - mixed/    : 10 events from the mixed (multi-PDG) dataset
  - electron/ : 10 events from a single-electron file (PDG 11)
  - pion/     : 10 events from a single-pion file (PDG -211)

Each event is a 3-panel figure. Coloring:
  - Truth: LCs coloured by argmax CP index; CP barycenters marked.
  - CLUE3D / CLUEstering: matched LCs coloured by CP; unmatched gray.
    Cluster barycenters shown as stars — same colour as LCs.
    Two same-colour stars = the algorithm split that CP.
  - Reco objects with < 3 LCs are shown in gray.

Usage:
  caloembed-display                                   # defaults: configs/raw.yaml
  caloembed-display --config configs/tune.yaml
  caloembed-display --dc 3.0 --rhoc 2.5 --n-events 5
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import uproot
import yaml

from caloembed.clustering.clue import run_clue, probe_backend

CP_COLORS = [matplotlib.colors.to_rgba(c) for c in plt.cm.tab10.colors]
GRAY  = matplotlib.colors.to_rgba('lightgray')
BLACK = matplotlib.colors.to_rgba((0.4, 0.4, 0.4, 1.0))



def score_and_color(cluster_ids, weights, lc_cp_idx, n_cp, min_lc, score_threshold):
    """Per-LC color index: >=0 matched CP, -1 gray (no match / too small), -2 black (outlier)."""
    lc_color = np.full(len(weights), -2, dtype=int)
    energy_sq = weights.astype(np.float64) ** 2

    for reco_id in np.unique(cluster_ids):
        if reco_id < 0:
            continue
        mask = cluster_ids == reco_id
        if mask.sum() < min_lc:
            lc_color[mask] = -1
            continue
        e_sq = energy_sq[mask]
        overlap = np.bincount(lc_cp_idx[mask], weights=e_sq, minlength=n_cp)
        cp_idx = int(np.argmax(overlap))
        score = 1.0 - overlap[cp_idx] / e_sq.sum()
        lc_color[mask] = cp_idx if score < score_threshold else -1

    return lc_color


def _idx_to_rgba(idx, n_cp):
    if idx >= 0:
        return CP_COLORS[idx % len(CP_COLORS)]
    return GRAY if idx == -1 else BLACK


def color_array(lc_color_idx, n_cp):
    return np.array([_idx_to_rgba(i, n_cp) for i in lc_color_idx])



def reco_centroids(cluster_ids, x, y, z, e, color_idx, min_lc):
    """Energy-weighted barycentre for each valid reco cluster."""
    out = []
    for cid in np.unique(cluster_ids):
        if cid < 0:
            continue
        mask = cluster_ids == cid
        if mask.sum() < min_lc:
            continue
        w = e[mask]
        out.append((
            np.average(z[mask], weights=w),
            np.average(x[mask], weights=w),
            np.average(y[mask], weights=w),
            int(color_idx[mask][0]),
        ))
    return out


def truth_centroids(truth_cp, x, y, z, e, n_cp):
    """Energy-weighted barycentre for each truth CP."""
    out = []
    for cp_idx in range(n_cp):
        mask = truth_cp == cp_idx
        if not mask.any():
            continue
        w = e[mask]
        out.append((
            np.average(z[mask], weights=w),
            np.average(x[mask], weights=w),
            np.average(y[mask], weights=w),
            cp_idx,
        ))
    return out



def load_event(h5_path, root_path, ev_idx, backend, clue_params, min_lc, score_threshold):
    with h5py.File(h5_path, 'r') as h:
        lc_off = h['lc/offsets'][:]
        cp_off = h['event/cp_offsets'][:]
        ls, le = int(lc_off[ev_idx]), int(lc_off[ev_idx + 1])
        cs, ce = int(cp_off[ev_idx]), int(cp_off[ev_idx + 1])
        x        = h['lc/x'][ls:le]
        y        = h['lc/y'][ls:le]
        z        = h['lc/z'][ls:le]
        e        = h['lc/energy'][ls:le]
        truth_cp = h['truth/argmax_cp_idx'][ls:le]
        pdg_ids  = h['event/cp_pdg_id'][cs:ce]
        n_cp     = ce - cs

    with uproot.open(root_path) as rf:
        vi = rf['ticlDumper/hltTiclTrackstersCLUE3DHigh']['vertices_indexes'].array(
            entry_start=ev_idx, entry_stop=ev_idx + 1
        )[0]

    lc_trackster = np.full(len(x), -2, dtype=int)
    for ts_idx, lc_list in enumerate(vi):
        for lc_i in lc_list:
            lc_trackster[int(lc_i)] = ts_idx

    clue3d_color = score_and_color(lc_trackster, e, truth_cp, n_cp, min_lc, score_threshold)

    coords = np.stack([x, y, z], axis=1).astype(np.float32)
    result = run_clue(coords=coords, weights=e.astype(np.float32),
                      backend=backend, **clue_params)
    clue_color = score_and_color(result.cluster_ids, e, truth_cp, n_cp, min_lc, score_threshold)

    return x, y, z, e, n_cp, truth_cp, clue3d_color, clue_color, pdg_ids, lc_trackster, result.cluster_ids



def _scatter3d(ax, z, x, y, color_idx, e, n_cp, title, centroids):
    colors = color_array(color_idx, n_cp)
    sizes  = np.clip(e / np.percentile(e, 95) * 35 + 2, 2, 55)

    bg = color_idx < 0
    if bg.any():
        ax.scatter(z[bg], x[bg], y[bg],
                   c=colors[bg], s=sizes[bg] * 0.6, alpha=0.40,
                   linewidths=0, rasterized=True, zorder=1)

    fg = ~bg
    if fg.any():
        ax.scatter(z[fg], x[fg], y[fg],
                   c=colors[fg], s=sizes[fg], alpha=0.85,
                   linewidths=0, rasterized=True, zorder=2)
    for cz, cx, cy, col in centroids:
        face = CP_COLORS[col % len(CP_COLORS)] if col >= 0 else GRAY
        ax.scatter([cz], [cx], [cy],
                   c=[face], s=280, marker='*',
                   edgecolors='black', linewidths=0.8,
                   zorder=10, alpha=1.0)

    ax.set_xlabel('z [cm]', fontsize=8, labelpad=0)
    ax.set_ylabel('x [cm]', fontsize=8, labelpad=0)
    ax.set_zlabel('y [cm]', fontsize=8, labelpad=0)
    ax.set_title(title, fontsize=11, pad=5)
    ax.tick_params(labelsize=6, pad=0)
    ax.view_init(elev=25, azim=-55)


def plot_event(h5_path, root_path, ev_idx, backend, clue_params, min_lc, score_threshold, out_path):
    x, y, z, e, n_cp, truth_cp, c3d_col, clue_col, pdg_ids, lc_ts, clue_ids = load_event(
        h5_path, root_path, ev_idx, backend, clue_params, min_lc, score_threshold
    )

    c3d_centres  = reco_centroids(lc_ts,    x, y, z, e, c3d_col,  min_lc)
    clue_centres = reco_centroids(clue_ids, x, y, z, e, clue_col, min_lc)
    truth_cens   = truth_centroids(truth_cp, x, y, z, e, n_cp)

    fig = plt.figure(figsize=(22, 7))

    panels = [
        (truth_cp, 'Truth',       truth_cens),
        (c3d_col,  'CLUE3D',      c3d_centres),
        (clue_col, 'CLUEstering', clue_centres),
    ]

    axes = []
    for col_i, (color_idx, label, centres) in enumerate(panels):
        ax = fig.add_subplot(1, 3, col_i + 1, projection='3d')
        _scatter3d(ax, z, x, y, color_idx, e, n_cp, label, centres)
        axes.append(ax)

    pad = 8
    zlim = (z.min() - pad, z.max() + pad)
    xlim = (x.min() - pad, x.max() + pad)
    ylim = (y.min() - pad, y.max() + pad)

    z_span = zlim[1] - zlim[0]
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    max_span = max(z_span, x_span, y_span)

    for ax in axes:
        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)
        ax.set_zlim(*ylim)
        ax.set_box_aspect([z_span / max_span, x_span / max_span, y_span / max_span])

    stem = Path(h5_path).stem
    pdg_str = ', '.join(map(str, np.unique(pdg_ids)))
    p = clue_params
    param_str = (f"dc={p['dc']}  rhoc={p['rhoc']}  "
                 f"do={p.get('do', '—')}  dm={p.get('dm', '—')}  "
                 f"metric={p['metric']}")
    fig.suptitle(
        f'{stem}  event {ev_idx}  |  n_CP={n_cp}  |  PDG={pdg_str}\n{param_str}',
        fontsize=9, y=1.0,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')



def find_file_by_pdg(hdf5_dir, target_pdg):
    """Return the first HDF5 file where all CPs have the given PDG ID."""
    for h5_file in sorted(hdf5_dir.glob('*.h5')):
        with h5py.File(h5_file, 'r') as h:
            pdgids = np.unique(h['event/cp_pdg_id'][:])
            if len(pdgids) == 1 and int(pdgids[0]) == target_pdg:
                return h5_file
    raise FileNotFoundError(f'No single-PDG file for PDG={target_pdg} in {hdf5_dir}')



_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG      = _REPO_ROOT / 'configs' / 'raw.yaml'
_DEFAULT_MIXED_HDF5  = Path('/vols/cms/mm1221/cms/Data/100k_mix/hdf5')
_DEFAULT_MIXED_ROOT  = Path('/vols/cms/mm1221/cms/Data/100k_mix/root')
_DEFAULT_SEP_HDF5    = Path('/vols/cms/mm1221/cms/Data/100k_mix_sep/hdf5')
_DEFAULT_SEP_ROOT    = Path('/vols/cms/mm1221/cms/Data/100k_mix_sep/root')
_DEFAULT_OUTPUT      = _REPO_ROOT / 'plots' / '3d_events'


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='3D event display: Truth | CLUE3D | CLUEstering')
    p.add_argument('--config', default=str(_DEFAULT_CONFIG),
                   help='YAML config file for clustering params (default: configs/raw.yaml)')
    p.add_argument('--mixed-data', default=str(_DEFAULT_MIXED_HDF5),
                   help='HDF5 dir for mixed events')
    p.add_argument('--single-data', default=str(_DEFAULT_SEP_HDF5),
                   help='HDF5 dir for single-particle events (electron + pion files)')
    p.add_argument('--output', default=str(_DEFAULT_OUTPUT),
                   help='Output directory for plots')
    p.add_argument('--n-events', type=int, default=10,
                   help='Events to plot per dataset (default: 10)')
    p.add_argument('--dc',      type=float, help='Override dc')
    p.add_argument('--rhoc',    type=float, help='Override rhoc')
    p.add_argument('--do',      type=float, help='Override do')
    p.add_argument('--dm',      type=float, help='Override dm')
    p.add_argument('--backend', default='auto', help='CLUEstering backend')
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clue_cfg = cfg['clustering']
    clue_params = dict(
        dc            = args.dc   if args.dc   is not None else clue_cfg['dc'],
        rhoc          = args.rhoc if args.rhoc is not None else clue_cfg['rhoc'],
        do            = args.do   if args.do   is not None else clue_cfg.get('do'),
        dm            = args.dm   if args.dm   is not None else clue_cfg.get('dm'),
        ppbin         = clue_cfg.get('ppbin', 128),
        metric        = clue_cfg.get('metric', 'euclidean'),
        metric_params = clue_cfg.get('metric_params'),
    )

    metrics_cfg     = cfg.get('metrics', {})
    min_lc          = metrics_cfg.get('min_lc', 3)
    score_threshold = metrics_cfg.get('purity_threshold', 0.2)

    mixed_hdf5 = Path(args.mixed_data)
    mixed_root = Path(str(mixed_hdf5).replace('/hdf5', '/root'))
    sep_hdf5   = Path(args.single_data)
    sep_root   = Path(str(sep_hdf5).replace('/hdf5', '/root'))
    out_root   = Path(args.output)

    backend = probe_backend(args.backend)
    print(f'Config:  {args.config}')
    print(f'Backend: {backend}')
    print(f'CLUE params: dc={clue_params["dc"]}  rhoc={clue_params["rhoc"]}  '
          f'do={clue_params["do"]}  dm={clue_params["dm"]}  '
          f'metric={clue_params["metric"]}\n')

    # Mixed events
    (out_root / 'mixed').mkdir(parents=True, exist_ok=True)
    print('=== mixed ===')
    h5_file   = sorted(mixed_hdf5.glob('*.h5'))[0]
    root_file = mixed_root / (h5_file.stem + '.root')
    for ev_idx in range(args.n_events):
        plot_event(str(h5_file), str(root_file), ev_idx, backend,
                   clue_params, min_lc, score_threshold,
                   str(out_root / 'mixed' / f'{h5_file.stem}_ev{ev_idx:03d}.png'))
    print()

    # Single-particle: electron and pion
    for target_pdg, label in [(11, 'electron'), (-211, 'pion')]:
        (out_root / label).mkdir(parents=True, exist_ok=True)
        print(f'=== {label} (PDG={target_pdg}) ===')
        h5_file   = find_file_by_pdg(sep_hdf5, target_pdg)
        root_file = sep_root / (h5_file.stem + '.root')
        print(f'  File: {h5_file.name}')
        for ev_idx in range(args.n_events):
            plot_event(str(h5_file), str(root_file), ev_idx, backend,
                       clue_params, min_lc, score_threshold,
                       str(out_root / label / f'{h5_file.stem}_ev{ev_idx:03d}.png'))
        print()


if __name__ == '__main__':
    main()
