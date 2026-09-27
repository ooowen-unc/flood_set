import argparse
import json
from pathlib import Path

from config import ARTIFACTS, MEASURES


def plot_curves(curves_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    payload = json.loads(Path(curves_path).read_text(encoding="utf-8"))
    if not payload.get("models"):
        raise ValueError("No curve data; regenerate metrics with the updated train.py first")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind in ("pr", "roc"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, measure in zip(axes.flat, MEASURES):
            prevalence = None
            defined = False
            for name, curves in payload["models"].items():
                curve = curves[measure]
                if not curve["defined"]:
                    continue
                defined = True
                prevalence = curve["prevalence"]
                points = curve[kind]
                xname, yname = ("recall", "precision") if kind == "pr" else ("false_positive_rate", "true_positive_rate")
                metric = curve["average_precision"] if kind == "pr" else curve["roc_auc"]
                label = f"{name}: {'AP' if kind == 'pr' else 'AUC'}={metric:.3f}"
                if kind == "pr":
                    ax.step([p[xname] for p in points], [p[yname] for p in points], where="post", label=label)
                else:
                    ax.plot([p[xname] for p in points], [p[yname] for p in points], label=label)
            if kind == "pr" and prevalence is not None:
                ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1, label="Prevalence")
            elif kind == "roc":
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random ranking")
            ax.set(title=measure.replace("_", " ").title(), xlim=(0, 1), ylim=(0, 1.02),
                   xlabel="Recall" if kind == "pr" else "False positive rate",
                   ylabel="Precision" if kind == "pr" else "True positive rate")
            if defined:
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(.5, .5, "Undefined: only one outcome class", ha="center", va="center")
            ax.grid(alpha=.2)
        fig.suptitle(f"TEST {'Precision-Recall' if kind == 'pr' else 'ROC'}: continuous scores, all candidates")
        path = output_dir / ("measure_test_" + kind + ".png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=ARTIFACTS / "measure_test_curves.json")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "plots")
    args = parser.parse_args()
    print(json.dumps({"plots": plot_curves(args.curves, args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
