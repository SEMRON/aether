#!/usr/bin/env python3
"""
Visualize evaluation metrics from JSON files or JSONL/CSV.

Usage:
    # From a directory of JSON files (from evaluate.py):
    python scripts/visualize_eval_sweep.py --eval-dir logs/gptneo/eval_results/

    # From a single metrics file:
    python scripts/visualize_eval_sweep.py --metrics-path eval_results/metrics.jsonl
"""

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


def _read_json_files(directory: Path) -> List[Dict[str, Any]]:
    """Read all JSON files from a directory, excluding 'latest' files."""
    rows: List[Dict[str, Any]] = []
    for json_file in sorted(directory.glob("*.json")):
        # Skip 'latest' symlink/file
        if "latest" in json_file.name:
            continue
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                # Add filename for reference
                data["_source_file"] = json_file.name
                rows.append(data)
        except Exception as e:
            print(f"Warning: Failed to read {json_file}: {e}")
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _coerce_numeric(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """csv.DictReader returns strings; try to coerce obvious numeric columns."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr: Dict[str, Any] = {}
        for k, v in r.items():
            if v is None:
                rr[k] = None
                continue
            if isinstance(v, (int, float)):
                rr[k] = v
                continue
            s = str(v).strip()
            if s == "":
                rr[k] = None
                continue
            try:
                rr[k] = int(s)
                continue
            except Exception:
                pass
            try:
                rr[k] = float(s)
                continue
            except Exception:
                rr[k] = s
        out.append(rr)
    return out


def _parse_ts(s: Any) -> Optional[datetime]:
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return None


_CKPT_RE = re.compile(r"^checkpoint_(?P<ts>.+)\.pt$")


def _parse_checkpoint_timestamp_from_path(checkpoint: Any) -> Optional[datetime]:
    """
    Best-effort parse of our checkpoint saver filename:
      checkpoint_2026-01-05_04:23:50.355696.pt
    """
    if not checkpoint:
        return None
    try:
        p = Path(str(checkpoint))
    except Exception:
        return None
    m = _CKPT_RE.match(p.name)
    if not m:
        return None
    ts = m.group("ts")
    # the saver uses datetime.now().isoformat(sep="_")
    try:
        return datetime.fromisoformat(ts.replace("_", "T", 1))
    except Exception:
        return None


def _best_effort_timestamp(r: Dict[str, Any]) -> Optional[datetime]:
    # Prefer checkpoint sweeps which store this explicitly.
    ts = _parse_ts(r.get("checkpoint_timestamp"))
    if ts is not None:
        return ts
    # Standard eval outputs.
    ts = _parse_ts(r.get("timestamp"))
    if ts is not None:
        return ts
    # Fallback: infer from checkpoint filename if present.
    return _parse_checkpoint_timestamp_from_path(r.get("checkpoint"))


def _sort_by_timestamp(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort rows by timestamp field."""
    def get_ts(r: Dict[str, Any]) -> datetime:
        ts = _parse_ts(r.get("timestamp"))
        if ts is None:
            # Fallback: try to parse from filename
            src = r.get("_source_file", "")
            # Format: eval_hostname_YYYYMMDD_HHMMSS.json
            try:
                parts = src.replace(".json", "").split("_")
                if len(parts) >= 2:
                    date_str = parts[-2]
                    time_str = parts[-1]
                    ts = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except Exception:
                pass
        return ts or datetime.min

    return sorted(rows, key=get_ts)


def _build_x(rows: List[Dict[str, Any]], *, x_axis: str) -> Tuple[List[float], Optional[List[str]], str]:
    """
    Returns:
      x: numeric axis values
      xlabels: optional (used when x is an index / categorical-ish)
      xlabel: axis label
    """
    xlabels: Optional[List[str]] = None

    if x_axis == "updates":
        x: List[float] = []
        for r in rows:
            v = r.get("updates")
            try:
                x.append(float(v) if v is not None else float("nan"))
            except Exception:
                x.append(float("nan"))
        return x, None, "updates"

    if x_axis == "samples":
        x = []
        for r in rows:
            v = r.get("examples_processed", r.get("samples_processed"))
            try:
                x.append(float(v) if v is not None else float("nan"))
            except Exception:
                x.append(float("nan"))
        return x, None, "samples processed"

    if x_axis == "time":
        tss = [_best_effort_timestamp(r) for r in rows]
        avail = [t for t in tss if t is not None]
        if not avail:
            # No timestamps: fall back to index.
            x = list(range(len(rows)))
            labs = [str(i) for i in x]
            return [float(i) for i in x], labs, "evaluation"

        t0 = min(avail)
        x = []
        for ts in tss:
            if ts is None:
                x.append(float("nan"))
            else:
                # Relative time: hours since first available timestamp
                x.append((ts - t0).total_seconds() / 3600.0)
        return x, None, "time (hours since first checkpoint)"

    # Default: index / checkpoint order with human-readable labels.
    x = list(range(len(rows)))
    labs: List[str] = []
    for r in rows:
        ts = _best_effort_timestamp(r)
        if ts is not None:
            labs.append(ts.strftime("%m-%d %H:%M"))
        elif r.get("checkpoint"):
            labs.append(Path(str(r["checkpoint"])).name)
        else:
            labs.append(str(len(labs)))
    return [float(i) for i in x], labs, "evaluation"


def _sorted_rows(rows: List[Dict[str, Any]], *, x_axis: str) -> List[Dict[str, Any]]:
    if x_axis == "time":
        def key(r: Dict[str, Any]) -> datetime:
            return _best_effort_timestamp(r) or datetime.min

        return sorted(rows, key=key)

    if x_axis in {"updates", "samples"}:
        key_name = "updates" if x_axis == "updates" else "examples_processed"

        def key(r: Dict[str, Any]):
            v = r.get(key_name, r.get("samples_processed") if x_axis == "samples" else None)
            try:
                return (0, float(v))
            except Exception:
                return (1, float("inf"))

        return sorted(rows, key=key)

    return rows


def _plot_series(
    *,
    out_path: Path,
    x: List[float],
    xlabels: Optional[List[str]],
    y: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    show_values: bool = False,
    yscale: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    line, = ax.plot(x, y, marker="o", linewidth=1.5, markersize=4)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)

    # Add value annotations if requested and not too many points
    if show_values and len(y) <= 20:
        for xi, yi in zip(x, y):
            ax.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    if xlabels is not None and len(xlabels) == len(x):
        if len(xlabels) <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
        else:
            # Show subset of labels for readability
            step = max(1, len(x) // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(xlabels[::step], rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _print_summary(rows: List[Dict[str, Any]], metric_name: str) -> None:
    """Print summary statistics for a metric."""
    values = [float(r[metric_name]) for r in rows if metric_name in r and r[metric_name] is not None]
    if not values:
        return
    
    print(f"\n{metric_name.upper()} Summary:")
    print(f"  Count:   {len(values)}")
    print(f"  Min:     {min(values):.4f}")
    print(f"  Max:     {max(values):.4f}")
    print(f"  Mean:    {sum(values) / len(values):.4f}")
    print(f"  Latest:  {values[-1]:.4f}")
    
    # Show trend (first vs last 3)
    if len(values) >= 6:
        first_avg = sum(values[:3]) / 3
        last_avg = sum(values[-3:]) / 3
        change = ((last_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0
        trend = "↓" if change < 0 else "↑" if change > 0 else "→"
        print(f"  Trend:   {trend} {abs(change):.1f}% (first 3 avg: {first_avg:.4f}, last 3 avg: {last_avg:.4f})")


def plot_eval_sweep(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    x_axis: str = "time",
    show_values: bool = False,
) -> None:
    """
    Shared plotting entrypoint for:
    - `scripts/visualize_eval_sweep.py` CLI
    - `scripts/evaluate_checkpoint_sweep.py` checkpoint sweeps
    """
    if not rows:
        raise RuntimeError("No data found.")

    valid_rows = [r for r in rows if "error" not in r]
    if not valid_rows:
        raise RuntimeError("No valid evaluation results found (all rows have errors).")

    valid_rows = _sorted_rows(valid_rows, x_axis=x_axis)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, xlabels, xlabel = _build_x(valid_rows, x_axis=x_axis)

    def _get_float_series(key: str) -> Optional[List[float]]:
        if not any((key in r) and (r.get(key) is not None) for r in valid_rows):
            return None
        ys: List[float] = []
        for r in valid_rows:
            v = r.get(key)
            try:
                ys.append(float(v) if v is not None else float("nan"))
            except Exception:
                ys.append(float("nan"))
        return ys

    # Loss (generic)
    losses = _get_float_series("loss")
    if losses is not None:
        _plot_series(
            out_path=out_dir / "loss.png",
            x=x,
            xlabels=xlabels,
            y=losses,
            title="Evaluation Loss Over Sweep",
            xlabel=xlabel,
            ylabel="loss",
            show_values=show_values,
        )
        _print_summary(valid_rows, "loss")

    # Perplexity (LLM)
    ppl = _get_float_series("perplexity")
    if ppl is not None and not all(math.isinf(p) for p in ppl if not math.isnan(p)):
        finite = [p for p in ppl if (not math.isnan(p)) and (not math.isinf(p))]
        if finite:
            _plot_series(
                out_path=out_dir / "perplexity.png",
                x=x,
                xlabels=xlabels,
                y=ppl,
                title="Perplexity Over Sweep",
                xlabel=xlabel,
                ylabel="perplexity",
                show_values=show_values,
                yscale="log",
            )
        _print_summary(valid_rows, "perplexity")

    # Accuracy (CV)
    acc = _get_float_series("accuracy_top1")
    if acc is not None:
        _plot_series(
            out_path=out_dir / "accuracy_top1.png",
            x=x,
            xlabels=xlabels,
            y=acc,
            title="Top-1 Accuracy Over Sweep",
            xlabel=xlabel,
            ylabel="accuracy_top1",
            show_values=show_values,
        )
        _print_summary(valid_rows, "accuracy_top1")

    # Image generation metrics (BigGAN)
    fids = _get_float_series("FID")
    if fids is not None:
        _plot_series(
            out_path=out_dir / "fid.png",
            x=x,
            xlabels=xlabels,
            y=fids,
            title="FID Over Sweep",
            xlabel=xlabel,
            ylabel="FID (lower is better)",
            show_values=show_values,
        )

    is_means = _get_float_series("IS_mean")
    is_stds = _get_float_series("IS_std")
    if is_means is not None:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.errorbar(x, is_means, yerr=is_stds, fmt="o-", linewidth=1.2, capsize=2)
        ax.set_title("Inception Score Over Sweep", fontsize=14)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("IS (higher is better)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if xlabels is not None and len(xlabels) == len(x) and len(xlabels) <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "is.png", dpi=200)
        plt.close(fig)

    if (fids is not None) and (is_means is not None):
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(fids, is_means, s=30)
        ax.set_title("IS vs FID (sweep)")
        ax.set_xlabel("FID (lower is better)")
        ax.set_ylabel("IS (higher is better)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "is_vs_fid.png", dpi=200)
        plt.close(fig)


@click.command()
@click.option(
    "--eval-dir",
    type=str,
    default=None,
    help="Directory containing JSON eval files (e.g., logs/gptneo/eval_results/)",
)
@click.option(
    "--metrics-path",
    type=str,
    default=None,
    help="Path to metrics.jsonl or metrics.csv (alternative to --eval-dir)",
)
@click.option(
    "--out-dir",
    type=str,
    default=None,
    help="Output directory for plots. Default: <input>/plots/",
)
@click.option(
    "--show-values",
    is_flag=True,
    help="Annotate data points with their values",
)
@click.option(
    "--x-axis",
    type=click.Choice(["time", "samples", "updates", "index"]),
    default="time",
    show_default=True,
    help="X axis for plots. 'time' is relative (hours since first).",
)
def main(
    eval_dir: Optional[str],
    metrics_path: Optional[str],
    out_dir: Optional[str],
    show_values: bool,
    x_axis: str,
):
    """Visualize evaluation metrics from JSON files or JSONL/CSV."""
    
    # Determine input source
    if eval_dir is not None:
        input_path = Path(eval_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_path}")
        if not input_path.is_dir():
            raise ValueError(f"--eval-dir must be a directory: {input_path}")
        
        rows = _read_json_files(input_path)
        rows = _sort_by_timestamp(rows)
        default_out = input_path / "plots"
        
    elif metrics_path is not None:
        mp = Path(metrics_path)
        if not mp.exists():
            raise FileNotFoundError(f"Metrics file not found: {mp}")

        if mp.suffix.lower() == ".jsonl":
            rows = _read_jsonl(mp)
        elif mp.suffix.lower() == ".csv":
            rows = _coerce_numeric(_read_csv(mp))
        else:
            raise ValueError("metrics-path must end with .jsonl or .csv")
        default_out = mp.parent / "plots"
        
    else:
        raise click.UsageError("Must specify either --eval-dir or --metrics-path")

    if not rows:
        raise RuntimeError("No data found.")

    out = Path(out_dir) if out_dir is not None else default_out
    plot_eval_sweep(rows=rows, out_dir=out, x_axis=x_axis, show_values=show_values)
    print(f"\nPlots saved to: {out}")


if __name__ == "__main__":
    main()
