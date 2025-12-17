import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------- 
def set_style():
    # portability
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
    })

def oed_loss_model(N_raw, H_free, nu, p=2.0):
    N_eff_0 = N_raw ** nu
    N_eff_actual = (N_eff_0 * (p * H_free)) / (N_eff_0 + p * H_free)
    L = H_free / N_eff_actual
    return L, N_eff_0, N_eff_actual
# -------------------------------------------------------

def plot_figures(outdir="."):
    set_style()

    # ---------------------------
    # Figure 1: Regime Diagram
    # ---------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    N_raw = np.logspace(1, 7, 200)
    p = 2.0

    configs = [
        (1e4, 0.4, r"Standard ($\nu=0.4,\, H=10^4$)"),
        (1e4, 0.8, r"Efficient ($\nu=0.8,\, H=10^4$)"),
        (1e2, 0.4, r"Data Limited ($\nu=0.4,\, H=10^2$)"),
    ]

    for H, nu, label in configs:
        L, _, _ = oed_loss_model(N_raw, H, nu, p)
        ax1.loglog(N_raw, L, label=label)

    # bending marker: N_eff^(0) = pH
    H_dl, nu_dl = 1e2, 0.4
    bend_x = (p * H_dl) ** (1.0 / nu_dl)
    bend_y = oed_loss_model(bend_x, H_dl, nu_dl, p)[0]
    ax1.scatter([bend_x], [bend_y], color="black", zorder=6)

    ax1.annotate(
        r"Bending" + "\n" + r"$N_{\mathrm{eff}}^{(0)}\approx pH$",
        xy=(bend_x, bend_y),
        xytext=(bend_x * 2.0, bend_y * 6.0),
        arrowprops=dict(arrowstyle="simple", facecolor="black", edgecolor="black"),
        ha="left", va="bottom",
    )

    ax1.set_xlabel(r"Nominal Parameters ($N_{\mathrm{raw}}$)")
    ax1.set_ylabel(r"Test Loss ($L$)")
    ax1.set_title("Regime Diagram of Extended Scaling")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, which="both", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(f"{outdir}/Fig1.pdf")
    fig1.savefig(f"{outdir}/Fig1.png", dpi=300)

    # ---------------------------
    # Figure 2: Schematic Data Collapse (toy)
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    N_samples = np.logspace(1, 6, 70)

    # toy master curve: 1 + phi_plot
    phi_theory = np.logspace(-4, 3, 300)
    ax2.loglog(phi_theory, 1.0 + phi_theory, "--", color="black",
               linewidth=2.5,
               label=r"Toy master curve: $1+\phi_{\mathrm{plot}}$")

    # enumerate
    for i, (H, nu, label) in enumerate(configs):
        L, N_eff_0, _ = oed_loss_model(N_samples, H, nu, p)

        phi_plot = N_eff_0 / (p * H)
        Ltilde_plot = L * N_eff_0 / H

        if i == 1:
            y_offset = 0.85
        else:
            y_offset = 1.15

        ax2.loglog(phi_plot, Ltilde_plot * y_offset,
                   linestyle="None", marker="o", markersize=6, alpha=0.9,
                   label=label)

    ax2.set_xlabel(r"Schematic filling ratio ($\phi_{\mathrm{plot}} \equiv N_{\mathrm{eff}}^{(0)}/(pH_{\mathrm{free}})$)")
    ax2.set_ylabel(r"Schematic normalized loss ($\tilde{L}_{\mathrm{plot}}\equiv L\cdot N_{\mathrm{eff}}^{(0)}/H_{\mathrm{free}}$)")
    ax2.set_title("Schematic Data Collapse (Toy Visualization)")
    ax2.set_xlim(1e-4, 2e3)
    ax2.set_ylim(1e-1, 2e3)
    ax2.grid(True, which="both", alpha=0.25)

    ax2.text(2e-3, 3.0, "Scaling Regime\n(Compression)", ha="center", va="center")
    ax2.text(6e0, 1.2e2, "Saturation Regime\n(Expansion)", ha="center", va="center")

    ax2.legend(loc="upper left", frameon=True)
    fig2.tight_layout()
    fig2.savefig(f"{outdir}/Fig2.pdf")
    fig2.savefig(f"{outdir}/Fig2.png", dpi=300)

if __name__ == "__main__":
    plot_figures(".")
