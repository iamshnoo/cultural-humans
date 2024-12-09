"""
# Python 3.8.6
pip install numpy pandas matplotlib seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.4)
sns.set_theme(style="whitegrid")
sns.set_context(
    "paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20}
)

# Load the data
file_path = "../results/summary_table.csv"
data = pd.read_csv(file_path)

# Filter rows where 'tgt_country' is "Average"
average_rows = data[data["tgt_country"] == "Average"]

# Filter out the "Overall Average" row where 'src_country' is "Overall Average" and 'tgt_country' is blank
average_rows = average_rows[average_rows["src_country"] != "Overall Average"]

# Reset index for easier handling
average_rows.reset_index(drop=True, inplace=True)

# Prepare the data for the radar chart
categories = average_rows["src_country"]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Close the circle


# Radar chart plotting function
def create_subplot_radar_chart(ax, metric_values, title, color, marker):
    values = metric_values.tolist()
    values += values[:1]  # Close the circle
    ax.plot(
        angles,
        values,
        color=color,
        linestyle="solid",
        linewidth=2,
        label=title,
        marker=marker,
    )
    ax.fill(angles, values, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        categories, fontsize=8, rotation=90, ha="center", position=(-0.08, 0.2)
    )
    ax.set_ylim(0, 1)  # SSIM is between 0 and 1
    ax.grid(True, color="#d3d3d3", linestyle="--")

    # Title inside gray box
    bbox_props = dict(
        facecolor="lightgrey", edgecolor="grey", alpha=0.7, boxstyle="round", pad=0.3
    )
    ax.set_title(
        title,
        fontsize=10,
        weight="bold",
        pad=20,
        bbox=bbox_props,
        ha="center",
        va="top",
    )


# Create subplots for SSIM, M1, and M2
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

# Plot radar charts for each metric
create_subplot_radar_chart(
    axs[0], average_rows["SSIM"], "SSIM", color="blue", marker="o"
)
create_subplot_radar_chart(
    axs[1],
    average_rows["M1"] / 100,
    "M1: (delta1<0) and (delta2>0) ",
    color="orange",
    marker="s",
)
create_subplot_radar_chart(
    axs[2], average_rows["M2"] / 100, "M2: (delta2-delta1>0)", color="green", marker="d"
)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("../results/figs/radar_chart.png", dpi=600, bbox_inches="tight")
