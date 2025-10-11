#This program is a Python script that reproduces Figures 2 through 10 from the paper "A Dynamic Model of Subjectivity: Toward an Integrative Computational Architecture for the Mind - Simulation Code".
#
#Author: Takeo Imaizumi
#DATE: Oct 1, 2025
#License: MIT License
#Contact: takeoimaizumi@song.ocn.ne.jp
#
#
# --- Importing libraries ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns

# --- Global Matplotlib settings ---
plt.rcParams.update({
    "figure.dpi": 600,
    "font.family": "serif",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.format": "pdf",
    "font.size": 10,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,   
})
plt.style.use("classic")
plt.rcParams.update({  
    "font.size": 10,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

# --- Simulation parameter settings ---
alpha = 0.1
sigma_epsilon = 0.05
T_plot = 100
T_satori_plot = 100
T_long = 10000
transient = 1000
seed_value = 42
G_gt1, G_lt1, G_neg, G_satori = 1.5, 0.3, -0.5, 1.0
mu_pos, mu_neg, mu_zero = 0.5, -0.5, 0.0
G_min, G_max, G_num = -3.0, 3.0, 400
mu_min, mu_max, mu_num = -1.5, 1.5, 300

# --- Function definitions ---
def update_ms_biased(ms_t, G, alpha, sigma_epsilon, mu, Mo=0):
    epsilon = np.random.normal(0, sigma_epsilon)
    return (1 - G) * ms_t - alpha * ms_t**3 + G * mu + G * epsilon

def generate_trajectory(G, mu, T, seed):
    np.random.seed(seed)
    ms = np.zeros(T)
    ms[0] = 0.0
    for t in range(T - 1):
        ms[t+1] = update_ms_biased(ms[t], G, alpha, sigma_epsilon, mu)
    return ms

def find_stable_points(G, alpha, mu):
    coeffs = [alpha, 0, G, -G * mu]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    return sorted([r for r in real_roots if abs(1 - G - 3 * alpha * r**2) < 1.0])

def sample_distribution_fixed_seed(stable_pts, mu, T_long, transient, seed):
    collected = []
    for i, start in enumerate(stable_pts):
        np.random.seed(seed + i)
        ms = np.zeros(T_long)
        ms[0] = start + np.random.normal(0, 0.1)
        for t in range(T_long - 1):
            ms[t+1] = update_ms_biased(ms[t], G_neg, alpha, sigma_epsilon, mu)
        collected.append(ms[transient:])
    return np.concatenate(collected)

def count_stable_equilibria(G, mu, alpha):
    coeffs = [alpha, 0, G, -G * mu]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    stable_count = 0
    for x_star in real_roots:
        F_prime = 1 - G - 3 * alpha * x_star**2
        if abs(F_prime) < 1.0:
            stable_count += 1
    return stable_count

# --- Data generation for each figure (common to all) -----------------
x_display          = np.arange(T_plot)
x_display_satori   = np.arange(T_satori_plot)

# Trajectories for later figures
ms_pos_fig4 = generate_trajectory(G_gt1, mu_pos, T_plot, seed_value)
ms_neg_fig4 = generate_trajectory(G_gt1, mu_neg, T_plot, seed_value)
ms_pos_fig5 = generate_trajectory(G_lt1, mu_pos, T_plot, seed_value)
ms_neg_fig5 = generate_trajectory(G_lt1, mu_neg, T_plot, seed_value)
ms_pos_fig6 = generate_trajectory(G_neg, mu_pos, T_plot, seed_value)
ms_neg_fig6 = generate_trajectory(G_neg, mu_neg, T_plot, seed_value)

# Stable points & distributions (Fig.6–7)
stable_B1 = find_stable_points(G_neg, alpha, mu_pos)
stable_B2 = find_stable_points(G_neg, alpha, mu_neg)
dist_B1   = sample_distribution_fixed_seed(stable_B1, mu_pos, T_long, transient, seed_value)
dist_B2   = sample_distribution_fixed_seed(stable_B2, mu_neg, T_long, transient, seed_value)

# Satori trajectory (Fig.8)
np.random.seed(seed_value)
ms_satori = np.zeros(T_satori_plot)
ms_satori[0] = 2.0
for t in range(T_satori_plot - 1):
    ms_satori[t+1] = update_ms_biased(ms_satori[t], G_satori, alpha, sigma_epsilon, mu_zero)

# Stability map for Fig.3
G_vals = np.linspace(G_min, G_max, G_num)
mu_vals = np.linspace(mu_min, mu_max, mu_num)
G_grid, mu_grid = np.meshgrid(G_vals, mu_vals)
stability_map = np.zeros_like(G_grid, dtype=int)
for i in range(mu_num):
    for j in range(G_num):
        stability_map[i, j] = count_stable_equilibria(G_grid[i, j], mu_grid[i, j], alpha)

# ---------------------------------------------------------------------
# --- Plotting and saving Fig.2 (Φ_H vs GΔt : Discretization comparison) ---
# ---------------------------------------------------------------------
def plot_and_save_fig2():
    """
    Generate Fig2.pdf: Comparison of linear self-coefficients Φ_H
    for several discretization methods vs GΔt.

    Requirements:
      - monochrome lines, distinct linestyles
      - no title
      - legend label 'ZOH' (no parentheses)
      - x=1 vertical dashed gray line (no text)
      - Φ=0 solid thin black line
      - larger fonts (axis labels 18pt, ticks 16pt, legend 14pt)
      - all spines (top/right/bottom/left) visible and solid
    """
    import matplotlib as mpl
    with mpl.rc_context({
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8
    }):
        # Data
        x = np.linspace(0, 2, 200)
        y_euler  = 1 - x
        y_zoh    = np.exp(-x)
        y_tustin = (1 - x/2) / (1 + x/2)
        y_taylor = 1 - x + (x**2)/2

        # Figure
        fig, ax = plt.subplots(figsize=(5.0, 2.6))

        # Curves (monochrome)
        ax.plot(x, y_taylor, label='2nd-order Taylor', color='black', linestyle=':',  linewidth=2.5)
        ax.plot(x, y_zoh,    label='ZOH',              color='black', linestyle='--', linewidth=1.5)
        ax.plot(x, y_tustin, label='Tustin',           color='black', linestyle='-', linewidth=1.0)
        ax.plot(x, y_euler,  label='Forward Euler',    color='black', linestyle='-',  linewidth=2.2)

        # Reference lines
        ax.axvline(1.0, color='gray',  linestyle='--', linewidth=1.0)  # GΔt=1 (no label)
        ax.axhline(0.0, color='black', linestyle='-',  linewidth=1.0)  # Φ=0 (solid thin black)

        # Labels / limits / grid
        ax.set_xlabel(r'$G\Delta t$')
        ax.set_ylabel(r'$\Phi_H$')
        ax.set_xlim(0, 2)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True)

        # Frame: show all spines as solid
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.0)
            ax.spines[side].set_linestyle('solid')

        ax.legend(loc='lower left')

        plt.tight_layout()
        fig.savefig("fig2.pdf", format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)
# --- Plotting and saving Fig.2 ---
plot_and_save_fig2()

# ---------------------------------------------------------------------
# --- Plotting and saving Fig.3 (Phase Diagram) ---
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.0, 4.0))

# Background colouring: number of stable equilibria
cmap = mcolors.ListedColormap(["white", "darkgray"])
norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5], cmap.N)
ax.contourf(G_grid, mu_grid, stability_map,
            levels=[0.5, 1.5, 2.5], cmap=cmap, norm=norm, zorder=1)
ax.contourf(G_grid, mu_grid, stability_map,
            levels=[-0.5, 0.5], colors="none", hatches=["///"], zorder=2)

# Figure markers (Fig.4–5,8 sampling points)
points = {
    "Fig. 4": [(1.5,  0.5), (1.5, -0.5)],
    "Fig. 5": [(0.3,  0.5), (0.3, -0.5)],
    "Fig. 6": [(-0.5, 0.5), (-0.5, -0.5)],
    "Fig. 8": [(1.0,  0.0)],
}
markers = {"Fig. 4": "s", "Fig. 5": "^", "Fig. 6": "o", "Fig. 8": "D"}
for key, coords in points.items():
    face = "white" if key == "Fig. 6" else "none"
    for gx, mx in coords:
        ax.scatter(gx, mx, marker=markers[key], s=60, edgecolor="black",
                   facecolor=face, linewidth=1.0, zorder=3)
        dx, dy = (0.08, 0.11) if mx >= 0 else (0.08, -0.18)
        if key == "Fig. 6": dx, dy = (-0.26, 0.09 if mx > 0 else -0.21)
        if key == "Fig. 4": dx = -0.1
        if key == "Fig. 8": dx, dy = (0.08, 0.08)
        ax.text(gx + dx, mx + dy, key, fontsize=8,
                ha="left", va="bottom", color="black", zorder=3)

# ----- Theory / bifurcation lines ------------------------------------
mu_bif   = np.linspace(mu_min, mu_max, 300)
g_cusp   = -27 * alpha / 4 * mu_bif**2
valid_sn = g_cusp < 0
ax.plot(g_cusp[valid_sn], mu_bif[valid_sn], "k--", linewidth=1.5)        # Saddle–Node

# Pitchfork (solid) & Reference Gain (dash-dot)
ax.plot([0, 0], [mu_min, mu_max],  "k-",  linewidth=1.5)                # Pitchfork


# Baseline (G axis)
ax.plot([G_min, G_max], [0, 0],    "k-",  linewidth=0.8)

# ----- Flip: dotted ---------------------------------------------------
# Flip curve (|λ|=1 → λ=−1)
x_vals = np.linspace(-4.0, 4.0, 800)
denom  = 2 - 3 * alpha * x_vals**2
G_flip_curve   = denom        # = 2 - 3αx²
mu_flip_curve  = np.where(np.abs(denom) > 1e-8,
                          2 * x_vals * (1 - alpha * x_vals**2) / denom,
                          np.nan)

# To prevent clipping artifacts at the plot boundaries, 
# replace out-of-bounds points with NaN so matplotlib creates a line break.
G_plot = G_flip_curve.copy()
mu_plot = mu_flip_curve.copy()
out_of_bounds = (
    (G_plot < G_min) | (G_plot > G_max) |
    (mu_plot < mu_min) | (mu_plot > mu_max)
)
G_plot[out_of_bounds] = np.nan
mu_plot[out_of_bounds] = np.nan

# Plot the cleaned curve
ax.plot(G_plot, mu_plot, linestyle=":", color="black", linewidth=1.5, zorder=4)

# ----- Example trajectory (unchanged) --------------------------------
cG = [-3.73898572881733, 0.2997416015624208,
      -40.215645791488505, 60.640193168639506,
      -17.642026071507033]
cM = [-2.33594094718375, -1.1986348351836051,
      7.300724487374742, 13.350866903128976,
      -12.358719377793546]
def _poly(cs, t): return sum(c * t**k for k, c in enumerate(cs))
G_traj  = lambda t: (1 - t) * 1.5 + t * 1.0 + t * (1 - t) * _poly(cG, t)
mu_traj = lambda t: (1 - t) * (-0.5) + t * 0.0 + t * (1 - t) * _poly(cM, t)
t_vals = np.linspace(0, 1, 400)
ax.plot(G_traj(t_vals), mu_traj(t_vals),
        linewidth=1.5, color="black", zorder=4)

# ----- Legend ---------------------------------------------------------
patch0 = mpatches.Patch(facecolor="white", edgecolor="black",
                        hatch="///", label="0 (Unstable)", linewidth=0.8)
patch1 = mpatches.Patch(facecolor="white", edgecolor="black",
                        label="1 (Single Stable)")
patch2 = mpatches.Patch(color="darkgray", label="2 (Bistable)")

legend_lines = [
    mpl.lines.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                     label="Saddle-Node Bif."),
    mpl.lines.Line2D([0], [0], color="black", linestyle="-",  linewidth=1.5,
                     label="Pitchfork Line"),
    mpl.lines.Line2D([0], [0], color="black", linestyle=":",  linewidth=1.5,
                     label="Flip"),
]

ax.legend(handles=[patch0, patch1, patch2] + legend_lines,
          loc="upper left",
          bbox_to_anchor=(0.0, 0.82, 0.44, 0.18),   
          frameon=True, handlelength=2.6,
          fontsize=8)

# ----- Axes / grid ----------------------------------------------------
ax.set_xticks(np.arange(G_min, G_max + 0.1, 1.0))
ax.set_yticks(np.arange(mu_min, mu_max + 0.1, 0.5))
ax.grid(True, linestyle="-", linewidth=0.5)
ax.set_xlabel("Affective Gain ($G$)")
ax.set_ylabel("Cognitive Bias ($\\mu$)")
ax.set_xlim(G_min, G_max)
ax.set_ylim(mu_min, mu_max)

plt.tight_layout()
fig.savefig("fig3.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.close()
# ---------------------------------------------------------------------
# --- Plotting and saving Fig.4 - Fig.9
# ---------------------------------------------------------------------

subplot_figsize = (5.0, 1.7)

# --- Plotting and saving Fig.4 (Trajectories for G > 1) ---
fig, axes = plt.subplots(1, 2, figsize=subplot_figsize, sharey=True)
axes[0].plot(x_display, ms_pos_fig4, color="black")
axes[0].axhline(0, color="black", linestyle="-", linewidth=1.0)
axes[0].set_xlim(-5, 100)
axes[0].set_title(r"Positive Bias ($\mu=0.5$)", fontsize=8)
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel(r"$M_s(t)$")
axes[0].tick_params(axis='both', which='major')
axes[0].grid(True, linestyle="-", linewidth=0.5)
axes[1].plot(x_display, ms_neg_fig4, color="black")
axes[1].axhline(mu_neg, color="black", linestyle="--")
axes[1].axhline(0, color="black", linestyle="-", linewidth=1.0)
axes[1].set_xlim(-5, 100)
axes[1].set_title(r"Negative Bias ($\mu=-0.5$)", fontsize=8)
axes[1].set_xlabel("Time Step")
axes[1].tick_params(axis='both', which='major')
axes[1].grid(True, linestyle="-", linewidth=0.5)
plt.tight_layout()
fig.savefig('fig4.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# --- Plotting and saving Fig.5 (Trajectories for 0 < G < 1) ---
fig, axes = plt.subplots(1, 2, figsize=subplot_figsize, sharey=True)
axes[0].plot(x_display, ms_pos_fig5, color="black")
#axes[0].axhline(mu_pos, color="black", linestyle="-")
axes[0].axhline(0, color="black", linestyle="-", linewidth=1.5)
axes[0].set_xlim(-5, 100)
axes[0].set_title(r"Positive Bias ($\mu=0.5$)", fontsize=8)
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel(r"$M_s(t)$")
axes[0].tick_params(axis='both', which='major')
axes[0].grid(True, linestyle="-", linewidth=0.5)
axes[1].plot(x_display, ms_neg_fig5, color="black")
#axes[1].axhline(mu_neg, color="black", linestyle="-")
axes[1].axhline(0, color="black", linestyle="-", linewidth=1.5)
axes[1].set_xlim(-5, 100)
axes[1].set_title(r"Negative Bias ($\mu=-0.5$)", fontsize=8)
axes[1].set_xlabel("Time Step")
axes[1].tick_params(axis='both', which='major')
axes[1].grid(True, linestyle="-", linewidth=0.5)
plt.tight_layout()
fig.savefig('fig5.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# --- Plotting and saving Fig.6 (Trajectories for G < 0 - Bistable) ---
fig, axes = plt.subplots(1, 2, figsize=subplot_figsize, sharey=True)
axes[0].plot(x_display, ms_pos_fig6, color="black")
axes[0].axhline(0, color="black", linestyle="-", linewidth=1.5)
for pt in stable_B1: axes[0].axhline(pt, color="black", linestyle="--")
axes[0].set_xlim(-5, 100)
axes[0].set_title(r"Positive Bias ($\mu=0.5$)", fontsize=8)
axes[0].set_xlabel("Time Step")
axes[0].set_ylabel(r"$M_s(t)$")
axes[0].tick_params(axis='both', which='major')
axes[0].grid(True, linestyle="-", linewidth=0.5)
axes[1].plot(x_display, ms_neg_fig6, color="black")
axes[1].axhline(0, color="black", linestyle="-", linewidth=1.5)
for pt in stable_B2: axes[1].axhline(pt, color="black", linestyle="--")
axes[1].set_xlim(-5, 100)
axes[1].set_title(r"Negative Bias ($\mu=-0.5$)", fontsize=8)
axes[1].set_xlabel("Time Step")
axes[1].tick_params(axis='both', which='major')
axes[1].grid(True, linestyle="-", linewidth=0.5)
plt.tight_layout()
fig.savefig('fig6.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# --- Plotting and saving Fig.7 (Distributions for G < 0 - Bistable) ---
fig, axes = plt.subplots(1, 2, figsize=subplot_figsize, sharey=True)
kde_kws = {'linewidth': 1.0}
axes[0].hist(dist_B1, bins=50, density=True, alpha=0.5, color="darkgray")
sns.kdeplot(dist_B1, ax=axes[0], color="black", **kde_kws)
axes[0].axvline(0, color="black", linestyle="-", linewidth=1.5)
for pt in stable_B1: axes[0].axvline(pt, color="black", linestyle="--")
axes[0].set_title(r"Positive Bias ($\mu=0.5$)", fontsize=8)
axes[0].set_xlabel(r"$M_s$")
axes[0].set_ylabel("Density")
axes[0].tick_params(axis='both', which='major')
axes[0].grid(True, linestyle="-", linewidth=0.5)
axes[1].hist(dist_B2, bins=50, density=True, alpha=0.5, color="darkgray")
sns.kdeplot(dist_B2, ax=axes[1], color="black", **kde_kws)
axes[1].axvline(0, color="black", linestyle="-", linewidth=1.5)
for pt in stable_B2: axes[1].axvline(pt, color="black", linestyle="--")
axes[1].set_title(r"Negative Bias ($\mu=-0.5$)", fontsize=8)
axes[1].set_xlabel(r"$M_s$")
axes[1].tick_params(axis='both', which='major')
axes[1].grid(True, linestyle="-", linewidth=0.5)
plt.tight_layout()
fig.savefig('fig7.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# --- Plotting and saving Fig.8 (Satori Trajectory, G=1, mu=0) ---
fig, ax = plt.subplots(figsize=(4.5, 1.7))
ax.plot(x_display_satori, ms_satori, color="black")
ax.axhline(0, color="black", linestyle="-")
ax.set_xlim(-5, 100)
ax.set_xlabel("Time Step")
ax.set_ylabel(r"$M_s(t)$")
ax.tick_params(axis='both', which='major')
ax.grid(True, linestyle="-", linewidth=0.5)
plt.tight_layout()
fig.savefig('fig8.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# --- Plotting and saving Fig.9 (Trajectory with time-varying G and mu) ---
fig, ax1 = plt.subplots(figsize=(5.0, 2.6))
T_total = 1000
t_vals_sim = np.linspace(0, 1, T_total)
G_vals_sim = G_traj(t_vals_sim)
mu_vals_sim = mu_traj(t_vals_sim)
np.random.seed(seed_value)
ms_vals_sim = np.zeros(T_total)
ms_vals_sim[0] = -0.5
for t in range(T_total - 1):
    ms_vals_sim[t+1] = update_ms_biased(ms_vals_sim[t], G_vals_sim[t], alpha, sigma_epsilon, mu_vals_sim[t])
ax1.plot(ms_vals_sim, label=r"$M_s(t)$", color="black", linewidth=1.0)
ax1.set_xlabel("Time Step")
ax1.set_ylabel(r"$M_s(t)$", color="black")
ax1.tick_params(axis='y', labelcolor="black")
ax1.tick_params(axis='x', which='major')
ax1.set_xlim(0, T_total)
ax1.set_xticks(np.arange(0, T_total + 1, 200))
ax1.grid(axis='x', linestyle="-", linewidth=0.5)
ax1.grid(axis='y', linestyle="-", linewidth=0.5)
ax1.axhline (y=0.0, color="black",linestyle="-", zorder=1)
axtwin = ax1.twinx()
axtwin.plot(G_vals_sim, label=r"$G(t)$", linestyle=":", color="black", linewidth=2.0)
axtwin.plot(mu_vals_sim, label=r"$\mu(t)$", linestyle="--", color="black", linewidth=1.0)
axtwin.set_ylabel(r"$G(t)$ / $\mu(t)$", color="black")
axtwin.tick_params(axis='y', labelcolor="black", length=0, labelsize=plt.rcParams["ytick.labelsize"])
axtwin.grid(False)
bistable_start = 400
bistable_end = 700
ax1.axvspan(bistable_start, bistable_end, color="darkgray", alpha=1.0, zorder=0)
Ms_lower = -1.1
Ms_upper = 2.7
ax1.set_ylim(Ms_lower, Ms_upper)
axtwin.set_ylim(Ms_lower, Ms_upper)
right_ticks = np.array([2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0])
axtwin.set_yticks(right_ticks)
axtwin.set_yticklabels([f"{tick:.1f}" for tick in right_ticks])
text_fontsize_fig9 = 8
ymax_ax1 = ax1.get_ylim()[1]
offset_fig9 = 0.08 * (ymax_ax1 - Ms_lower)
text_ypos_fig9 = ymax_ax1 + offset_fig9
ax1.text(bistable_start / 2, text_ypos_fig9, "Single Stable", ha='center', va='bottom', fontsize=text_fontsize_fig9)
ax1.text((bistable_start + bistable_end) / 2, text_ypos_fig9, "Bistable", ha='center', va='bottom', fontsize=text_fontsize_fig9)
ax1.text((bistable_end + T_total) / 2, text_ypos_fig9, "Single Stable", ha='center', va='bottom', fontsize=text_fontsize_fig9)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = axtwin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper right", bbox_to_anchor=(0.96, 0.99),
           frameon=True, handlelength=2.8, fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("fig9.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------
# --- Plotting and saving Fig.10 (SGBD Regimes: 2x2 in one figure) ---
# ---------------------------------------------------------------------
def plot_and_save_fig10(
    suptitle_text="Fig. 10  SGBD Framework: The Four Psychodynamic Regimes",
    suptitle_size=18,
    panel_title_size=14, 
    label_size=13,
    tick_size=11, 
    figsize=(11, 11),
):
    # Grid & common params
    m_range = np.linspace(-5.0, 5.0, 40)
    m1, m2 = np.meshgrid(m_range, m_range)
    M = np.array([m1, m2])

    def draw_panel(ax, title, G, W_Fe):
        dM_dt = -G * np.tensordot(W_Fe, M, axes=1) - alpha * M**3
        ax.streamplot(m1, m2, dM_dt[0], dM_dt[1], density=0.8,
                      color="black", linewidth=0.8, arrowsize=1.5)
        ax.set_title(title, fontsize=panel_title_size)
        ax.set_xlabel("m1", fontsize=label_size)
        ax.set_ylabel("m2", fontsize=label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(linestyle=":", color="gray", linewidth=1.5)
        ax.set_xlim([-5, 5]); ax.set_ylim([-5, 5])
        ax.set_aspect("equal", adjustable="box")

    # Single figure with 2x2 subplots
    fig10, axes10 = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    if suptitle_text:
        fig10.suptitle(suptitle_text, fontsize=suptitle_size, y=0.98)
        rect = [0, 0, 1, 0.95]
    else:
        rect = [0, 0, 1, 1]


    # (A) Healthy Conflict (G>0, S-dominant)
    G_q1 = 0.8
    W_Fe_q1 = np.array([[-1.0, 0.0], [0.0, 1.0]])
    draw_panel(axes10[0, 0], "(A) Healthy Conflict (G>0, S-dominant)", G_q1, W_Fe_q1)

    # (B) Severe Conflict (G<0, S-dominant)
    G_q2 = -0.8
    W_Fe_q2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    draw_panel(axes10[0, 1], "(B) Severe Conflict (G<0, S-dominant)", G_q2, W_Fe_q2)

    # (C) Stable Rumination (G>0, R-dominant)
    G_q3 = 0.4
    W_Fe_q3 = np.array([[1.0, 2.5], [-2.5, 1.0]])
    draw_panel(axes10[1, 0], "(C) Stable Rumination (G>0, R-dominant)", G_q3, W_Fe_q3)

    # (D) Catastrophic Rumination (G<0, R-dominant)
    G_q4 = -0.4
    W_Fe_q4 = W_Fe_q3
    draw_panel(axes10[1, 1], "(D) Catastrophic Rumination (G<0, R-dominant)", G_q4, W_Fe_q4)

    plt.tight_layout(rect=rect)
    for ax in axes10[0, :]:
         ax.tick_params(axis='x', labelbottom=True)
    for ax in axes10[:, 1]:
         ax.tick_params(axis='y', labelleft=True, left=True,
                                                labelright=False, right=False)
         ax.yaxis.set_label_position("left")
         ax.yaxis.tick_left()
    fig10.subplots_adjust (hspace=0.18)
    fig10.subplots_adjust (wspace=0.1)
    fig10.savefig("fig10.pdf", format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig10)

plot_and_save_fig10(suptitle_text=None, panel_title_size=16, label_size=14, tick_size=12)
