"""
Compare an original event-transition probability matrix to a generated one.

Both matrices are expected to have been produced with the same event-type
ordering (rows/columns), e.g. the original by ``create_transition_matrix.py``
and the generated one by ``create_transition_matrix_generated.py``.

Outputs (under ``--output_dir``):
    comparison_per_row.csv         -- per source-event metrics
    comparison_overall.json        -- aggregate metrics + max abs/rel diffs
    top_transitions_comparison.csv -- union of top-10 cells from each matrix
    heatmap_original.png
    heatmap_generated.png
    heatmap_diff.png               -- signed P_gen - P_orig

Console: max absolute diff, max relative diff (two variants), top-10 transitions
of original, top-10 transitions of generated.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from diffusion_transformer.evaluation.create_transition_matrix import (  # noqa: E402
    load_transition_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _kl_divergence(p, q, eps=1e-12):
    """KL(p || q) per row -- p, q are 2-D arrays of shape (n, n)."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=1)


def _js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence per row (symmetric, bounded in [0, log 2])."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, eps) + 0.5 * _kl_divergence(q, m, eps)


def _topk_rank_overlap(p, q, k):
    """Per-row Jaccard overlap of the top-k targets (averaged over rows)."""
    overlaps = []
    for i in range(p.shape[0]):
        top_p = set(np.argsort(p[i])[::-1][:k])
        top_q = set(np.argsort(q[i])[::-1][:k])
        union = top_p | top_q
        if not union:
            overlaps.append(1.0)
        else:
            overlaps.append(len(top_p & top_q) / len(union))
    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(original_dir, generated_dir, output_dir, top_k=10):
    original_dir = Path(original_dir)
    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading original matrix from   {original_dir}")
    prob_orig_df, count_orig_df, mapping_orig = load_transition_matrix(
        cache_dir=str(original_dir), as_dataframe=True
    )

    logger.info(f"Loading generated matrix from  {generated_dir}")
    prob_gen_df, count_gen_df, mapping_gen = load_transition_matrix(
        cache_dir=str(generated_dir), as_dataframe=True
    )

    # ---- alignment check ---------------------------------------------------
    if list(prob_orig_df.index) != list(prob_gen_df.index) or \
       list(prob_orig_df.columns) != list(prob_gen_df.columns):
        raise ValueError(
            "Original and generated matrices have different row/column orderings. "
            "Both must be produced with the same event-type set."
        )

    event_names = list(prob_orig_df.index)
    P_orig = prob_orig_df.values.astype(np.float64)
    P_gen = prob_gen_df.values.astype(np.float64)
    n = P_orig.shape[0]

    diff = P_gen - P_orig
    abs_diff = np.abs(diff)

    # ---- per-row metrics --------------------------------------------------
    l1_per_row = abs_diff.sum(axis=1)               # = 2 * total variation
    tv_per_row = 0.5 * l1_per_row
    jsd_per_row = _js_divergence(P_orig, P_gen)
    kl_orig_gen = _kl_divergence(P_orig, P_gen)
    kl_gen_orig = _kl_divergence(P_gen, P_orig)

    per_row_df = pd.DataFrame({
        'event_type': event_names,
        'l1_distance': l1_per_row,
        'total_variation': tv_per_row,
        'js_divergence': jsd_per_row,
        'kl_orig_gen': kl_orig_gen,
        'kl_gen_orig': kl_gen_orig,
    })
    per_row_df.to_csv(output_dir / 'comparison_per_row.csv', index=False)
    logger.info(f"Saved per-row metrics to {output_dir / 'comparison_per_row.csv'}")

    # ---- max diffs (absolute and relative) --------------------------------
    flat_idx_abs = int(np.argmax(abs_diff))
    i_abs, j_abs = flat_idx_abs // n, flat_idx_abs % n
    max_abs_info = {
        'source': event_names[i_abs],
        'target': event_names[j_abs],
        'p_orig': float(P_orig[i_abs, j_abs]),
        'p_gen': float(P_gen[i_abs, j_abs]),
        'abs_diff': float(abs_diff[i_abs, j_abs]),
    }

    eps_rel = 1e-9
    rel_diff_all = abs_diff / np.maximum(P_orig, eps_rel)
    flat_idx_rel = int(np.argmax(rel_diff_all))
    i_rel, j_rel = flat_idx_rel // n, flat_idx_rel % n
    max_rel_all_info = {
        'source': event_names[i_rel],
        'target': event_names[j_rel],
        'p_orig': float(P_orig[i_rel, j_rel]),
        'p_gen': float(P_gen[i_rel, j_rel]),
        'abs_diff': float(abs_diff[i_rel, j_rel]),
        'rel_diff': float(rel_diff_all[i_rel, j_rel]),
        'eps': eps_rel,
    }

    # Variant restricted to non-tiny original probabilities
    p_threshold = 1e-4
    mask = P_orig > p_threshold
    if mask.any():
        rel_diff_masked = np.where(mask, abs_diff / np.maximum(P_orig, eps_rel), -np.inf)
        flat_idx_rel_m = int(np.argmax(rel_diff_masked))
        i_rm, j_rm = flat_idx_rel_m // n, flat_idx_rel_m % n
        max_rel_masked_info = {
            'source': event_names[i_rm],
            'target': event_names[j_rm],
            'p_orig': float(P_orig[i_rm, j_rm]),
            'p_gen': float(P_gen[i_rm, j_rm]),
            'abs_diff': float(abs_diff[i_rm, j_rm]),
            'rel_diff': float(rel_diff_masked[i_rm, j_rm]),
            'p_orig_threshold': p_threshold,
        }
    else:
        max_rel_masked_info = None

    # ---- top-k transitions in each matrix ---------------------------------
    def _flat_topk(prob_matrix, k):
        flat_idx = np.argsort(prob_matrix.ravel())[::-1][:k]
        return [(int(idx // n), int(idx % n)) for idx in flat_idx]

    top_orig_ij = _flat_topk(P_orig, top_k)
    top_gen_ij = _flat_topk(P_gen, top_k)

    set_orig = set(top_orig_ij)
    set_gen = set(top_gen_ij)
    union = list(set_orig | set_gen)

    top_rows = []
    # Render original list first (in original ranking order), then generated-only entries.
    seen = set()
    for rank, (i, j) in enumerate(top_orig_ij, 1):
        seen.add((i, j))
        top_rows.append({
            'source': event_names[i],
            'target': event_names[j],
            'p_orig': float(P_orig[i, j]),
            'p_gen': float(P_gen[i, j]),
            'abs_diff': float(abs_diff[i, j]),
            'in_top10_orig': True,
            'orig_rank': rank,
            'in_top10_gen': (i, j) in set_gen,
            'gen_rank': (top_gen_ij.index((i, j)) + 1) if (i, j) in set_gen else None,
        })
    for rank, (i, j) in enumerate(top_gen_ij, 1):
        if (i, j) in seen:
            continue
        top_rows.append({
            'source': event_names[i],
            'target': event_names[j],
            'p_orig': float(P_orig[i, j]),
            'p_gen': float(P_gen[i, j]),
            'abs_diff': float(abs_diff[i, j]),
            'in_top10_orig': False,
            'orig_rank': None,
            'in_top10_gen': True,
            'gen_rank': rank,
        })

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(output_dir / 'top_transitions_comparison.csv', index=False)
    logger.info(f"Saved top-{top_k} transition comparison to "
                f"{output_dir / 'top_transitions_comparison.csv'}")

    # ---- overall metrics --------------------------------------------------
    total_count_orig = int(count_orig_df.values.sum())
    total_count_gen = int(count_gen_df.values.sum())

    overall = {
        'matrix_shape': list(P_orig.shape),
        'num_event_types': n,
        'total_transitions_original': total_count_orig,
        'total_transitions_generated': total_count_gen,
        'mean_l1_per_row': float(np.mean(l1_per_row)),
        'mean_total_variation': float(np.mean(tv_per_row)),
        'mean_js_divergence': float(np.mean(jsd_per_row)),
        'frobenius_norm_diff': float(np.linalg.norm(diff, 'fro')),
        'max_abs_cell_diff': max_abs_info,
        'max_rel_cell_diff_all': max_rel_all_info,
        'max_rel_cell_diff_filtered': max_rel_masked_info,
        'topk_overlap': {
            'k=1': _topk_rank_overlap(P_orig, P_gen, 1),
            'k=3': _topk_rank_overlap(P_orig, P_gen, 3),
            'k=5': _topk_rank_overlap(P_orig, P_gen, 5),
        },
        'top_transitions_original': [
            {'rank': r + 1,
             'source': event_names[i],
             'target': event_names[j],
             'p_orig': float(P_orig[i, j]),
             'p_gen': float(P_gen[i, j])}
            for r, (i, j) in enumerate(top_orig_ij)
        ],
        'top_transitions_generated': [
            {'rank': r + 1,
             'source': event_names[i],
             'target': event_names[j],
             'p_gen': float(P_gen[i, j]),
             'p_orig': float(P_orig[i, j])}
            for r, (i, j) in enumerate(top_gen_ij)
        ],
    }
    with open(output_dir / 'comparison_overall.json', 'w') as f:
        json.dump(overall, f, indent=2)
    logger.info(f"Saved overall metrics to {output_dir / 'comparison_overall.json'}")

    # ---- heatmaps ---------------------------------------------------------
    _plot_heatmap(prob_orig_df, output_dir / 'heatmap_original.png',
                  title='Original transition probabilities', signed=False)
    _plot_heatmap(prob_gen_df, output_dir / 'heatmap_generated.png',
                  title='Generated transition probabilities', signed=False)
    diff_df = pd.DataFrame(diff, index=event_names, columns=event_names)
    _plot_heatmap(diff_df, output_dir / 'heatmap_diff.png',
                  title='P_gen - P_orig (signed)', signed=True)

    # ---- console summary --------------------------------------------------
    _print_console_summary(overall, top_orig_ij, top_gen_ij,
                           event_names, P_orig, P_gen, top_k)

    return overall


def _plot_heatmap(df, path, title, signed=False):
    n = df.shape[0]
    fig_w = max(10, 0.45 * n)
    fig_h = max(8, 0.45 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if signed:
        vmax = float(np.max(np.abs(df.values)))
        cmap = 'RdBu_r'
        sns.heatmap(df, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
                    square=True, linewidths=0.2, cbar_kws={'shrink': 0.7})
    else:
        sns.heatmap(df, ax=ax, cmap='viridis', vmin=0.0, vmax=1.0,
                    square=True, linewidths=0.2, cbar_kws={'shrink': 0.7})

    ax.set_title(title)
    ax.set_xlabel('Next event')
    ax.set_ylabel('Current event')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap: {path}")


def _print_console_summary(overall, top_orig_ij, top_gen_ij,
                           event_names, P_orig, P_gen, top_k):
    print()
    print('=' * 78)
    print('TRANSITION MATRIX COMPARISON')
    print('=' * 78)
    print(f"Matrix shape           : {overall['matrix_shape']}")
    print(f"Total transitions orig : {overall['total_transitions_original']:,}")
    print(f"Total transitions gen  : {overall['total_transitions_generated']:,}")
    print(f"Mean L1 per row        : {overall['mean_l1_per_row']:.6f}")
    print(f"Mean total variation   : {overall['mean_total_variation']:.6f}")
    print(f"Mean JS divergence     : {overall['mean_js_divergence']:.6f}")
    print(f"Frobenius norm of diff : {overall['frobenius_norm_diff']:.6f}")
    print(f"Top-1/3/5 row overlap  : "
          f"{overall['topk_overlap']['k=1']:.3f} / "
          f"{overall['topk_overlap']['k=3']:.3f} / "
          f"{overall['topk_overlap']['k=5']:.3f}")

    m = overall['max_abs_cell_diff']
    print()
    print(f"Max ABSOLUTE cell diff:")
    print(f"  {m['source']:>22s} -> {m['target']:<22s}  "
          f"P_orig={m['p_orig']:.6f}  P_gen={m['p_gen']:.6f}  "
          f"|diff|={m['abs_diff']:.6f}")

    m = overall['max_rel_cell_diff_all']
    print(f"Max RELATIVE cell diff (all cells, eps={m['eps']:.0e}):")
    print(f"  {m['source']:>22s} -> {m['target']:<22s}  "
          f"P_orig={m['p_orig']:.6e}  P_gen={m['p_gen']:.6e}  "
          f"rel_diff={m['rel_diff']:.6f}")

    m = overall['max_rel_cell_diff_filtered']
    if m is not None:
        print(f"Max RELATIVE cell diff (P_orig > {m['p_orig_threshold']:.0e}):")
        print(f"  {m['source']:>22s} -> {m['target']:<22s}  "
              f"P_orig={m['p_orig']:.6f}  P_gen={m['p_gen']:.6f}  "
              f"rel_diff={m['rel_diff']:.6f}")

    print()
    print(f"Top-{top_k} ORIGINAL transitions  (source -> target : P_orig  (P_gen))")
    print('-' * 78)
    for rank, (i, j) in enumerate(top_orig_ij, 1):
        print(f"  {rank:2d}. {event_names[i]:>22s} -> {event_names[j]:<22s} : "
              f"{P_orig[i, j]:.6f}  ({P_gen[i, j]:.6f})")

    print()
    print(f"Top-{top_k} GENERATED transitions (source -> target : P_gen   (P_orig))")
    print('-' * 78)
    for rank, (i, j) in enumerate(top_gen_ij, 1):
        print(f"  {rank:2d}. {event_names[i]:>22s} -> {event_names[j]:<22s} : "
              f"{P_gen[i, j]:.6f}  ({P_orig[i, j]:.6f})")
    print('=' * 78)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original vs generated transition probability matrices."
    )
    parser.add_argument(
        '--original_dir', type=str, default='cache/transition_matrix',
        help='Directory of cached original transition matrix files.'
    )
    parser.add_argument(
        '--generated_dir', type=str, default='cache/transition_matrix_generated',
        help='Directory of generated transition matrix files.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs/transition_comparison',
        help='Where to write CSVs, JSON, and heatmaps.'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top transitions to report per matrix (default: 10).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    def _resolve(path_str):
        p = Path(path_str)
        return p if p.is_absolute() else project_root / p

    compare(
        original_dir=_resolve(args.original_dir),
        generated_dir=_resolve(args.generated_dir),
        output_dir=_resolve(args.output_dir),
        top_k=args.top_k,
    )


if __name__ == '__main__':
    main()
