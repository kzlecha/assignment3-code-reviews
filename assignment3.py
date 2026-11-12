"""CSE 242 Assignment 3, Question 1: hard-margin linear SVM (30 points).
"""

import argparse
from pathlib import Path


SVM_X = ((1, 1), (1, 2), (2, 1), (0, 0), (1, 0), (0, 1))
SVM_Y = (1, 1, 1, -1, -1, -1)


def plot_svm_data(output_path):
    """Generate an annotated scatter plot to help inspect Q1(a) by eye.

    Point IDs are 1-based and match the source table order. This helper
    does not identify support vectors or draw the separating boundary.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib could not be imported. Use the README's clean virtual "
            "environment and install requirements.txt."
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 6))
    for label, color, marker in ((1, "tab:blue", "o"), (-1, "tab:orange", "s")):
        points = [x for x, y in zip(SVM_X, SVM_Y) if y == label]
        ax.scatter(
            [x[0] for x in points], [x[1] for x in points],
            color=color, marker=marker, s=90, label=f"y = {label:+d}",
        )
    for point_id, (x1, x2) in enumerate(SVM_X, start=1):
        ax.annotate(str(point_id), (x1, x2), xytext=(7, 7),
                    textcoords="offset points")
    ax.set(xlabel="x1", ylabel="x2", title="Assignment 3: SVM training data",
           xlim=(-0.5, 2.5), ylim=(-0.5, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def identify_support_vectors():
    """Q1(a), 10 points together with the plot: identify support vectors by eye.

    Return their 1-based point IDs and explain your choice in the report.
    """
    # the separator is a line going though (x1,x2) points: (0,1.5) (0.5,1) (1,0.5) (1.5, 0)
    # i.e. the line x1 + x2 = 1.5. The points closest to that line (one on
    # each side) are the support vectors:
    #   point 1 (1,1), y=+1 -> distance |1+1-1.5|/sqrt(2) = 0.354
    #   point 5 (1,0), y=-1 -> distance |1+0-1.5|/sqrt(2) = 0.354
    #   point 6 (0,1), y=-1 -> distance |0+1-1.5|/sqrt(2) = 0.354
    # points 2 (1,2), 3 (2,1), 4 (0,0) are all 1.5/sqrt(2) = 1.06 away, so
    # moving them would not change the boundary -> not support vectors.
    return (1, 5, 6)


def derive_svm_separator()->tuple:
    """Q1(b), 10 points for the boundary and 10 for the geometric margin.

    Use a linear, hard-margin SVM. The decision boundary is w dot x + b = 0;
    retain the intercept b. Use the support vectors to derive it.

    Return (weights, bias, geometric_margin), with weights a pair of numbers.
    Explain the derivation and your margin convention in the written report.
    """
    import numpy as np

    # optimize min(1/2 * wlen**2) subject to yn(w*xn+b)>=1 for all n
    # the constraint is tight (= 1) exactly at the support vectors, so
    # y_n (w . x_n + b) = 1  <=>  w . x_n + b = y_n   (since y_n = +/-1)
    # point 1 (1,1): w1 + w2 + b = +1
    # point 5 (1,0): w1      + b = -1
    # point 6 (0,1):      w2 + b = -1
    # -> w1 = w2 = 2, b = -3, i.e. 2*x1 + 2*x2 - 3 = 0  (same as x1 + x2 = 1.5)
    sv_ids = identify_support_vectors()
    sv_x = np.array([SVM_X[i - 1] for i in sv_ids], dtype=float)
    sv_y = np.array([SVM_Y[i - 1] for i in sv_ids], dtype=float)
    # rows are [x1, x2, 1] so that A @ [w1, w2, b] = y
    A = np.column_stack([sv_x, np.ones(len(sv_ids))])
    w1, w2, b = np.linalg.solve(A, sv_y)

    # margin = (1/w_len) on each side of w
    # this is the perpendicular distance from the boundary to the closest
    # points on each side: 1/||w|| = 1/sqrt(2^2 + 2^2) = 1/(2*sqrt(2)) ~= 0.354
    # (NOT 0.5 -- 0.5 is the horizontal/vertical gap on the grid, not the
    # perpendicular distance). Full width between w.x+b=+1 and w.x+b=-1 is
    # 2/||w|| = 1/sqrt(2) ~= 0.707.
    w_len = np.sqrt(w1 ** 2 + w2 ** 2)
    geometric_margin = 1 / w_len
    return (float(w1), float(w2)), float(b), float(geometric_margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("plot", "svm"),
                        default="plot", help="Default: generate the SVM data plot.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "svm_data.png",
                        help="Output image path for --task plot.")
    args = parser.parse_args()
    if args.task == "plot":
        print(f"Saved plot to {plot_svm_data(args.output)}")
    else:
        print("Support vector point IDs:", identify_support_vectors())
        weights, bias, margin = derive_svm_separator()
        print(f"Weights: {weights}; bias: {bias}; geometric margin: {margin}")


if __name__ == "__main__":
    main()
