# --- Library Import ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns

# --- Global Settings ---
plt.rcParams.update({
    "figure.dpi": 600,
    "font.size": 14, # Base font size set to 14pt
    "axes.labelsize": 14, # Default size for axis labels
    "axes.titlesize": 16, # Default size for titles
    "xtick.labelsize": 12, # Default size for x-axis tick labels
    "ytick.labelsize": 12, # Default size for y-axis tick labels
    "legend.fontsize": 12, # Default size for legends (Fig.3 overrides this)
    "font.family": "serif",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
})
