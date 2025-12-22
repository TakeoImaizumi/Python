import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# -----------------------------------------------------------------------------
# 1. Plotting Style Settings
# -----------------------------------------------------------------------------
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "lines.linewidth": 2.5,
        "figure.dpi": 150,
        "legend.fontsize": 11,
        "mathtext.fontset": "cm",
    })

# -----------------------------------------------------------------------------
# 2. Theoretical Models
# -----------------------------------------------------------------------------
def oed_cost(N, H, p=2.0):
    return (1.0/p) * np.log(N) + H / N

def scaling_model(N_raw, H_free, nu, p=2.0):
    N_eff_0 = N_raw ** nu
    N_eff_actual = (N_eff_0 * (p * H_free)) / (N_eff_0 + p * H_free)
    L = H_free / N_eff_actual
    return L, N_eff_0

# -----------------------------------------------------------------------------
# 3. Figure builders
# -----------------------------------------------------------------------------
def make_fig1(p=2.0):
    fig, ax = plt.subplots(figsize=(8, 6))

    N_range = np.logspace(0, 4.5, 500)
    payloads = [20, 100, 500]
    colors_A = ['#2ca02c', '#1f77b4', '#ff7f0e']

    for H, col in zip(payloads, colors_A):
        ax.semilogx(N_range, oed_cost(N_range, H, p), color=col, label=rf"Total Cost ($H={H}$)")
        min_N = p * H
        min_C = oed_cost(min_N, H, p)
        ax.scatter([min_N], [min_C], color=col, s=60, zorder=5, edgecolors='white')

    # Decomposition lines: H=20
    H_demo = 20
    ax.semilogx(N_range, H_demo / N_range, linestyle='--', color='gray', alpha=0.6,
                linewidth=1.5, label=rf'Compression ($H/N$), $H={H_demo}$')
    ax.semilogx(N_range, (1.0/p) * np.log(N_range), linestyle=':', color='gray', alpha=0.6,
                linewidth=1.5, label=r'Expansion ($p^{-1}\ln N$)')

    ax.set_xlabel(r'Dimensionality $N$ (log scale)')
    ax.set_ylabel(r'OED Cost $\mathcal{C}_p$')
    ax.set_title(r'Geometric Equilibrium Shift (Conceptual)')
    ax.set_ylim(0, 8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    return fig

def make_fig2(p=2.0):
    fig, ax = plt.subplots(figsize=(8, 4.6))

    # Data Limited: H=1e5
    configs = [
        (1e7, 0.45, "Standard"),
        (1e7, 0.90, "High"),
        (1e5, 0.45, "Data Limited"),
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    N_raw_ext = np.logspace(4, 20, 700)

    for (H, nu, name), col in zip(configs, colors):
        label = rf"{name} ($\nu={nu:.2f}$, $H=10^{{{int(np.log10(H))}}}$)"
        L, _ = scaling_model(N_raw_ext, H, nu, p)
        ax.loglog(N_raw_ext, L, label=label, color=col)

        # mark bending (phi~1)
        sat_N = (p * H) ** (1.0 / nu)
        if 1e4 < sat_N < 1e20:
            sat_L = scaling_model(np.array([sat_N]), H, nu, p)[0][0]
            ax.scatter([sat_N], [sat_L], s=50, marker='o', color=col,
                       edgecolors='white', zorder=4)

    ax.set_xlabel(r"Nominal Parameters ($N_{raw}$)")
    ax.set_ylabel(r"Test Loss ($L$)")

    # Title without "Astronomical Scale"
    ax.set_title("Extended Scaling")

    ax.set_xlim(1e4, 1e20)
    ax.set_ylim(0.4, 100)
    ax.grid(True, which="both", alpha=0.2)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))

    # L = 10^0 dotted line in LIGHT GRAY
    ax.axhline(1.0, linestyle=':', color='gray', linewidth=1.2, alpha=0.35, zorder=0)

    ax.legend(loc="upper right", framealpha=0.9)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.15)
    return fig

def make_fig3(p=2.0):
    # Data Limited: H=1e5
    configs = [
        (1e7, 0.45, "Standard"),
        (1e7, 0.90, "High"),
        (1e5, 0.45, "Data Limited"),
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    z_orders = [4, 2, 3]

    N_raw_race = np.logspace(1, 11, 400)
    phi_axis = np.logspace(-6, 6, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8))

    ax1.loglog(phi_axis, 1.0 + phi_axis, "--", color="gray", alpha=0.6, linewidth=2,
               label=r"Master: $\tilde{L}=1+\phi$")
    ax2.loglog(phi_axis, phi_axis, "--", color="gray", alpha=0.6, linewidth=2,
               label=r"Master: $\tilde{L}-1=\phi$")

    ax1.axvline(x=1.0, color='black', linestyle=':', linewidth=1)
    ax2.axvline(x=1.0, color='black', linestyle=':', linewidth=1)

    for i, ((H, nu, name), col) in enumerate(zip(configs, colors)):
        L, N_eff_0 = scaling_model(N_raw_race, H, nu, p)
        phi = N_eff_0 / (p * H)
        L_tilde = L * N_eff_0 / H

        label = rf"{name} ($\nu={nu:.2f}$, $H=10^{{{int(np.log10(H))}}}$)"
        z = z_orders[i]

        ax1.loglog(phi, L_tilde, color=col, linewidth=4, label=label, zorder=z)
        ax2.loglog(phi, L_tilde - 1.0, color=col, linewidth=4, label=label, zorder=z)

        x_end = phi[-1]
        y1_end = L_tilde[-1]
        y2_end = (L_tilde - 1.0)[-1]

        mk = {'Standard':'s','High':'*','Data Limited':'X'}[name]
        ax1.scatter([x_end], [y1_end], s=220, color=col, marker=mk,
                    edgecolors='black', zorder=z+10)
        ax2.scatter([x_end], [y2_end], s=220, color=col, marker=mk,
                    edgecolors='black', zorder=z+10)

        # Left panel labels
        if name in ("High", "Data Limited"):
            ax1.annotate(r"$N=10^{11}$", xy=(x_end, y1_end),
                         xytext=(14, 6), textcoords='offset points',
                         ha='left', va='center',
                         color=col, fontsize=10, fontweight='bold', zorder=z+20)
        else:
            ax1.annotate(r"$N=10^{11}$", xy=(x_end, y1_end),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', va='bottom',
                         color=col, fontsize=10, fontweight='bold', zorder=z+20)

        # Right panel labels (High/Data moved a bit more right)
        if name in ("High", "Data Limited"):
            ax2.annotate(r"$N=10^{11}$", xy=(x_end, y2_end),
                         xytext=(20, 12), textcoords='offset points',
                         ha='left', va='center',
                         color=col, fontsize=10, fontweight='bold', zorder=z+20)
        else:
            ax2.annotate(r"$N=10^{11}$", xy=(x_end, y2_end),
                         xytext=(0, 16), textcoords='offset points',
                         ha='center', va='bottom',
                         color=col, fontsize=10, fontweight='bold', zorder=z+20)

    ax1.set_xlabel(r"Filling Ratio $\phi = N_{eff}^{(0)}/(pH)$")
    ax1.set_ylabel(r"Normalized Loss $\tilde{L}$")
    ax1.set_title(r"$\tilde{L}$ vs $\phi$")
    ax1.grid(True, which="both", alpha=0.2)
    ax1.set_xlim(1e-6, 1e6)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_xlabel(r"Filling Ratio $\phi = N_{eff}^{(0)}/(pH)$")
    ax2.set_ylabel(r"Shifted: $\tilde{L}-1$")
    ax2.set_title(r"$\tilde{L}-1$ vs $\phi$")
    ax2.grid(True, which="both", alpha=0.2)
    ax2.set_xlim(1e-6, 1e6)
    ax2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 4. Export
# -----------------------------------------------------------------------------
def export_all(show=False):
    set_style()
    p = 2.0

    fig1 = make_fig1(p)
    fig2 = make_fig2(p)
    fig3 = make_fig3(p)

    fig1.savefig("Fig1.pdf", bbox_inches="tight")
    fig2.savefig("Fig2.pdf", bbox_inches="tight")
    fig3.savefig("Fig3.pdf", bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

if __name__ == "__main__":
    export_all(show=False)
