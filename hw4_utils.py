import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sign(z: float) -> int:
    """Return +1 if z >= 0 else -1."""
    return 1 if z >= 0 else -1


def make_stream(n: int=200, margin: int=0.05):
    """Yield a stream of (x, y) pairs where:
    - x is a np.array with shape (2,)
    - y is in {-1, +1}"""
    rng = np.random.default_rng(0)

    true_w = rng.normal(size=2)
    true_w /= (np.linalg.norm(true_w) + 1e-12)
    true_b = rng.uniform(-0.2, 0.2)

    count = 0
    while count < n:
        x = rng.uniform(-1.0, 1.0, size=(2,))
        s = true_w @ x + true_b
        if abs(s) < margin:
            continue
        y = sign(s)
        yield x, y
        count += 1


def collect_stream_data(n: int = 300, margin: float = 0.05) -> list:
    """Collect n labelled examples from a fresh stream into a plain list.

    The visualisation helpers need a reusable list, not a one-shot generator,
    so use this wrapper instead of calling make_stream() directly.
    """
    return list(make_stream(n=n, margin=margin))


def _build_score_grids(w: np.ndarray, b: float, resolution: int = 60):
    """Return (X1, X2, Z_score, Z_pred) meshgrids over the [-1, 1]^2 domain."""
    lin = np.linspace(-1.0, 1.0, resolution)
    X1, X2 = np.meshgrid(lin, lin)
    Z_score = w[0] * X1 + w[1] * X2 + b          # raw linear score
    Z_pred  = np.where(Z_score >= 0, 1.0, -1.0)  # sign(score)
    return X1, X2, Z_score, Z_pred


def visualize_perceptron_3d(
    w: np.ndarray,
    b: float,
    data: list,
    title: str = "Perceptron — 3-D Score Surface & Step Function",
) -> None:
    """Two-panel 3-D plot that illuminates what the perceptron has learned.

    Parameters
    ----------
    w    : weight vector, shape (2,)
    b    : bias scalar
    data : list of (x, y) pairs  (use collect_stream_data() to make one)
    """
    xs = np.array([pt[0] for pt in data])   # shape (N, 2)
    ys = np.array([pt[1] for pt in data])   # shape (N,)
    pos, neg = ys == 1, ys == -1

    X1, X2, Z_score, Z_pred = _build_score_grids(w, b)

    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Left panel: raw score surface ─────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    surf = ax1.plot_surface(
        X1, X2, Z_score,
        cmap="RdBu_r", alpha=0.50,
        vmin=-2.5, vmax=2.5,
        linewidth=0, antialiased=True,
    )
    fig.colorbar(surf, ax=ax1, shrink=0.45, pad=0.08, label="score = w·x + b")

    # Semi-transparent green plane at z = 0 (the decision boundary)
    ax1.plot_surface(X1, X2, np.zeros_like(Z_score),
                     color="limegreen", alpha=0.18, linewidth=0)

    # Green contour where the score surface crosses zero
    if not np.allclose(w, 0):
        ax1.contour(X1, X2, Z_score, levels=[0],
                    colors="green", linewidths=2.0, zdir="z", offset=0.0)

    # Data points at their true label height
    ax1.scatter(xs[pos, 0], xs[pos, 1], np.ones(pos.sum()),
                c="steelblue", marker="^", s=20, alpha=0.75,
                depthshade=True, label="y = +1")
    ax1.scatter(xs[neg, 0], xs[neg, 1], -np.ones(neg.sum()),
                c="firebrick", marker="o", s=20, alpha=0.75,
                depthshade=True, label="y = -1")

    ax1.set_xlabel("x1", labelpad=8)
    ax1.set_ylabel("x2", labelpad=8)
    ax1.set_zlabel("w·x + b", labelpad=8)
    ax1.set_title(
        "Linear Score Surface\n"
        "Green plane & line = decision boundary (score = 0)",
        fontsize=9,
    )
    ax1.legend(loc="upper left", fontsize=8, markerscale=1.4)

    # ── Right panel: step / sign function ────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax2.plot_surface(X1, X2, Z_pred,
                     cmap="RdBu_r", alpha=0.55,
                     vmin=-1.6, vmax=1.6,
                     linewidth=0, antialiased=True)

    # Green contour projected onto the lower shelf to mark the boundary edge
    if not np.allclose(w, 0):
        ax2.contour(X1, X2, Z_score, levels=[0],
                    colors="limegreen", linewidths=2.0, zdir="z", offset=-1.0)

    ax2.scatter(xs[pos, 0], xs[pos, 1], np.ones(pos.sum()),
                c="steelblue", marker="^", s=20, alpha=0.75,
                depthshade=True, label="y = +1")
    ax2.scatter(xs[neg, 0], xs[neg, 1], -np.ones(neg.sum()),
                c="firebrick", marker="o", s=20, alpha=0.75,
                depthshade=True, label="y = -1")

    ax2.set_xlabel("x1", labelpad=8)
    ax2.set_ylabel("x2", labelpad=8)
    ax2.set_zlabel("sign(w·x + b)", labelpad=8)
    ax2.set_title(
        "Step (Sign) Function Output\n"
        "y-hat snaps to ±1 — sharp jump at the boundary",
        fontsize=9,
    )
    ax2.legend(loc="upper left", fontsize=8, markerscale=1.4)

    plt.tight_layout()
    plt.show()


def visualize_training_snapshots(
    snapshots: list,
    data: list,
    figsize_per_panel: tuple = (5, 4.5),
) -> None:
    """Show the decision boundary at several moments during training.
    Parameters
    ----------
    snapshots          : list of (step, w, b) — from run_online_with_snapshots
    data               : the same list of (x, y) pairs used during training
    figsize_per_panel  : (width, height) for each sub-panel in inches
    """
    n_panels = len(snapshots)
    ncols    = min(n_panels, 3)
    nrows    = (n_panels + ncols - 1) // ncols

    xs = np.array([pt[0] for pt in data])
    ys = np.array([pt[1] for pt in data])
    pos, neg = ys == 1, ys == -1

    lin = np.linspace(-1.0, 1.0, 50)
    X1, X2 = np.meshgrid(lin, lin)

    fig = plt.figure(
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    )
    fig.suptitle(
        "How the Decision Boundary Evolves Throughout Training\n"
        "Blue = region predicted +1   |   Red = region predicted -1\n"
        "Green line = current decision boundary",
        fontsize=11, fontweight="bold",
    )

    # Subsample data points so small panels stay readable
    every = max(1, len(xs) // 80)
    xs_pos_sub = xs[pos][::every] if pos.any() else np.empty((0, 2))
    xs_neg_sub = xs[neg][::every] if neg.any() else np.empty((0, 2))

    for idx, (step, w, b) in enumerate(snapshots):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")

        Z_score = w[0] * X1 + w[1] * X2 + b
        Z_pred  = np.where(Z_score >= 0, 1.0, -1.0)

        ax.plot_surface(X1, X2, Z_pred,
                        cmap="RdBu_r", alpha=0.50,
                        vmin=-1.6, vmax=1.6,
                        linewidth=0, antialiased=True)

        # Draw the boundary contour only when weights are non-zero
        if not np.allclose(w, 0):
            ax.contour(X1, X2, Z_score, levels=[0],
                       colors="limegreen", linewidths=2.0,
                       zdir="z", offset=-1.0)

        if len(xs_pos_sub) > 0:
            ax.scatter(xs_pos_sub[:, 0], xs_pos_sub[:, 1],
                       np.ones(len(xs_pos_sub)),
                       c="steelblue", marker="^", s=10,
                       alpha=0.65, depthshade=True)
        if len(xs_neg_sub) > 0:
            ax.scatter(xs_neg_sub[:, 0], xs_neg_sub[:, 1],
                       -np.ones(len(xs_neg_sub)),
                       c="firebrick", marker="o", s=10,
                       alpha=0.65, depthshade=True)

        step_label = "init" if step == 0 else f"t = {step}"
        ax.set_title(f"After {step_label}", fontsize=9)
        ax.set_xlabel("x1", labelpad=3, fontsize=7)
        ax.set_ylabel("x2", labelpad=3, fontsize=7)
        ax.set_zlabel("y-hat", labelpad=3, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_zlim(-1.3, 1.3)

    plt.tight_layout()
    plt.show()
