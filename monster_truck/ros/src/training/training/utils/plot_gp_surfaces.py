#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from gp_dynamics import GPManager   # your GPManager with .load() and .dataset()


def plot_gp_surfaces_with_uncertainty(
    gp,
    a_values,
    a_uncert=1.0,
    action_tolerance=0.25,
    title_zlabel="d(rate)/dt",
):
    """
    One figure with 4 plots:
      - Top row: 3 mean surfaces for actions in `a_values`
      - Bottom row: mean surface for `a_uncert`, colored by predictive std

    gp              : GPManager instance (already loaded)
    a_values        : list of 3 actions for top row, e.g. [-1.0, 0.27, 1.0]
    a_uncert        : action value used in the uncertainty plot (bottom)
    action_tolerance: how close training samples' actions must be to overlay
    title_zlabel    : label for Z axis (e.g. "d(rate)/dt")
    """

    # ---------- 1) Training data ----------
    X_train, Y_train = gp.dataset()
    flip = X_train[:, 0]   # state 0
    rate = X_train[:, 1]   # state 1
    act  = X_train[:, 2]   # action
    dY   = Y_train         # GP target (e.g. d(rate)/dt)

    # Grid in (flip, rate)
    p_min, p_max = flip.min(), flip.max()
    r_min, r_max = rate.min(), rate.max()

    p_grid = np.linspace(p_min, p_max, 80)
    r_grid = np.linspace(r_min, r_max, 80)
    P, R = np.meshgrid(p_grid, r_grid)

    # ---------- 2) Figure layout with GridSpec ----------
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.06],   # last column is for colorbars
        height_ratios=[1, 1],
        wspace=0.3,
        hspace=0.25,
    )

    ax_top = [
        fig.add_subplot(gs[0, i], projection="3d") for i in range(3)
    ]
    cax_mean = fig.add_subplot(gs[0, 3])    # colorbar for mean (top row)

    ax_uncert = fig.add_subplot(gs[1, 0:3], projection="3d")
    cax_std   = fig.add_subplot(gs[1, 3])   # colorbar for std (bottom)

    # Font sizes
    label_fs = 18
    title_fs = 18
    tick_fs  = 14

    # ---------- 3) Top row: mean surfaces for each action ----------
    z_min, z_max = np.inf, -np.inf
    mean_surfaces = []

    # First pass: compute means and global z limits
    for a_fixed in a_values:
        X_grid = np.column_stack([
            P.ravel(),
            R.ravel(),
            np.full(P.size, a_fixed, np.float32),
        ])
        Mean_t, _ = gp.predict_torch(X_grid)
        Mean_np = Mean_t.detach().cpu().numpy().reshape(P.shape)

        mean_surfaces.append((a_fixed, Mean_np))
        z_min = min(z_min, Mean_np.min())
        z_max = max(z_max, Mean_np.max())

    # Second pass: plot on each of the top axes
    surfaces = []
    for ax, (a_fixed, Mean_np) in zip(ax_top, mean_surfaces):
        print(f"[TOP] Action a = {a_fixed}")

        surf = ax.plot_surface(
            P, R, Mean_np,
            cmap="coolwarm",
            linewidth=0,
            antialiased=True,
            alpha=0.9,
            vmin=z_min,
            vmax=z_max,
        )
        surfaces.append(surf)

        # overlay training samples near this action
        mask = np.abs(act - a_fixed) < action_tolerance
        print(f"  training samples near a={a_fixed}: n = {np.sum(mask)}")

        ax.scatter(
            flip[mask], rate[mask], dY[mask],
            color="k", s=15, alpha=0.7, label="data"
        )

        ax.set_xlabel("flip (state 0)", fontsize=label_fs)
        ax.set_ylabel("rate (state 1)", fontsize=label_fs)
        ax.set_zlabel(title_zlabel, fontsize=label_fs)
        ax.set_title(f"a = {a_fixed:.2f}", fontsize=title_fs)
        ax.set_zlim(z_min, z_max)

        ax.legend(fontsize=10)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.tick_params(axis="z", labelsize=tick_fs)

    # Shared colorbar for mean (top row)
    from matplotlib.cm import ScalarMappable
    sm_mean = ScalarMappable(
        cmap=surfaces[0].cmap,
        norm=surfaces[0].norm,
    )
    sm_mean.set_array([])
    cb_mean = fig.colorbar(sm_mean, cax=cax_mean)
    cb_mean.set_label("GP mean", fontsize=label_fs)
    cb_mean.ax.tick_params(labelsize=tick_fs)

    # ---------- 4) Bottom: mean height + std color for a_uncert ----------
    print(f"[BOTTOM] Uncertainty plot for a_uncert = {a_uncert}")
    X_grid_unc = np.column_stack([
        P.ravel(),
        R.ravel(),
        np.full(P.size, a_uncert, np.float32),
    ])

    Mean_t, Var_t = gp.predict_torch(X_grid_unc)
    Mean_unc = Mean_t.detach().cpu().numpy().reshape(P.shape)
    Var_unc  = Var_t.detach().cpu().numpy().reshape(P.shape)
    Std_unc  = np.sqrt(Var_unc)

    # Color by std
    norm_std = plt.Normalize(vmin=Std_unc.min(), vmax=Std_unc.max())
    colors_std = plt.cm.viridis(norm_std(Std_unc))

    surf_unc = ax_uncert.plot_surface(
        P, R, Mean_unc,
        facecolors=colors_std,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    # Overlay data near that action
    mask_unc = np.abs(act - a_uncert) < action_tolerance
    print(f"  training samples near a={a_uncert}: n = {np.sum(mask_unc)}")

    ax_uncert.scatter(
        flip[mask_unc], rate[mask_unc], dY[mask_unc],
        color="k", s=15, alpha=0.6,
        label=f"data (a≈{a_uncert})"
    )

    ax_uncert.set_xlabel("flip (state 0)", fontsize=label_fs)
    ax_uncert.set_ylabel("rate (state 1)", fontsize=label_fs)
    ax_uncert.set_zlabel(title_zlabel, fontsize=label_fs)
    ax_uncert.set_title(
        f"a = {a_uncert:.2f} — Mean (height), Std (color)",
        fontsize=title_fs,
    )
    ax_uncert.tick_params(axis="both", labelsize=tick_fs)
    ax_uncert.tick_params(axis="z", labelsize=tick_fs)
    ax_uncert.legend(fontsize=10)

    # Colorbar for std
    sm_std = ScalarMappable(cmap="viridis", norm=norm_std)
    sm_std.set_array([])
    cb_std = fig.colorbar(sm_std, cax=cax_std)
    cb_std.set_label("GP predictive std", fontsize=label_fs)
    cb_std.ax.tick_params(labelsize=tick_fs)

    plt.show()


def main():
    # Load your already-trained model (no retrain)
    gp_path = "models/gp_dynamics_1.pt"    # adapt to your path
    gp = GPManager.load(gp_path)

    # Top row actions
    a_values = [-1.0, 0.27, 1.0]
    # Bottom plot action (can be one of those, e.g. 1.0)
    a_uncert = 1.0

    plot_gp_surfaces_with_uncertainty(
        gp,
        a_values=a_values,
        a_uncert=a_uncert,
        action_tolerance=0.5,
        title_zlabel="d(rate)/dt",
    )


if __name__ == "__main__":
    main()
