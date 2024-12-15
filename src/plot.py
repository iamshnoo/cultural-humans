"""
# Python 3.8.6
pip install numpy pandas matplotlib seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Seaborn and Matplotlib settings
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

# Filter out the "Overall Average" row
average_rows = average_rows[average_rows["src_country"] != "Overall Average"]

# Reset index
average_rows.reset_index(drop=True, inplace=True)

# Convert metrics to numeric
for metric in ["SSIM", "M1", "M2"]:
    average_rows[metric] = pd.to_numeric(average_rows[metric], errors="coerce")

# Radar chart setup
categories = average_rows["src_country"]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]


def create_subplot_radar_chart(ax, metric_values, title, color, marker):
    values = metric_values.tolist()
    values += values[:1]
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
    ax.set_ylim(0, 1)
    ax.grid(True, color="#d3d3d3", linestyle="--")

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
create_subplot_radar_chart(
    axs[0], average_rows["SSIM"], "SSIM", color="blue", marker="o"
)
create_subplot_radar_chart(
    axs[1],
    average_rows["M1"] / 100,
    "M1: (delta1<0) and (delta2>0)",
    color="orange",
    marker="s",
)
create_subplot_radar_chart(
    axs[2], average_rows["M2"] / 100, "M2: (delta2-delta1>0)", color="green", marker="d"
)
plt.tight_layout()
plt.savefig("../results/figs/radar_chart.png", dpi=600, bbox_inches="tight")

# Weight-Based Composite Scoring
weight_versions = [
    {"SSIM": 0.7, "M2": 0.3},
    {"SSIM": 0.4, "M2": 0.6},
    {"SSIM": 0.5, "M2": 0.5},
]

for i, weights in enumerate(weight_versions, start=1):
    average_rows[f"Composite_Score_V{i}"] = (
        average_rows["SSIM"] * weights["SSIM"] + average_rows["M2"] * weights["M2"]
    ) / 2

    composite_sorted = average_rows.sort_values(
        by=f"Composite_Score_V{i}", ascending=False
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=f"Composite_Score_V{i}",
        y="src_country",
        data=composite_sorted,
        palette="coolwarm",
        edgecolor="black",
    )
    plt.title(f"Composite Score by Country (Version {i})", fontsize=16, weight="bold")
    plt.xlabel("Composite Score", fontsize=14)
    plt.ylabel("Source Country", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"../results/figs/composite_score_v{i}_bar_chart.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()

# Country-Continent Mapping
continent_mapping = {
    "Algeria": "Africa",
    "Argentina": "South America",
    "Brazil": "South America",
    "Canada": "North America",
    "China": "Asia",
    "Egypt": "Africa",
    "France": "Europe",
    "India": "Asia",
    "Italy": "Europe",
    "Japan": "Asia",
    "Malaysia": "Asia",
    "Mexico": "North America",
    "Morocco": "Africa",
    "South Africa": "Africa",
    "Spain": "Europe",
    "Thailand": "Asia",
    "Tunisia": "Africa",
    "Turkey": "Asia",
    "United Kingdom": "Europe",
    "United States": "North America",
}


def sort_countries_by_continent(countries):
    return sorted(countries, key=lambda x: (continent_mapping.get(x, ""), x))


def country_pair_interaction_analysis(data, metric):
    if metric in ["M1", "M2"]:
        data[metric] = data[metric] / 100.0

    pivot_table = data.pivot_table(
        index="src_country", columns="tgt_country", values=metric, aggfunc="mean"
    )
    sorted_countries = sort_countries_by_continent(pivot_table.index)
    pivot_table = pivot_table.reindex(index=sorted_countries, columns=sorted_countries)

    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        pivot_table,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        square=True,
        annot_kws={"size": 8},
    )
    plt.title(f"{metric}", fontsize=16, weight="bold")
    plt.xlabel("Target Country", fontsize=14)
    plt.ylabel("Source Country", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"../results/figs/country_pair_{metric}.png", dpi=600, bbox_inches="tight"
    )
    plt.show()


metrics_to_analyze = ["SSIM", "M1", "M2"]
for metric in metrics_to_analyze:
    country_pair_interaction_analysis(data, metric)

# Activity Performance Analysis
metrics_csv = "../results/metrics.csv"
metrics = pd.read_csv(metrics_csv)

grouped = metrics.groupby("activity")
activity_performance = []
for activity, group in grouped:
    ssim = group["similarity"].mean()
    m1 = ((group["delta1"] < 0) & (group["delta2"] > 0)).mean() * 100
    m2 = ((group["delta2"] - group["delta1"]) > 0).mean() * 100
    activity_performance.append(
        {"activity": activity, "SSIM": ssim, "M1": m1, "M2": m2}
    )

activity_df = pd.DataFrame(activity_performance)
for metric in ["SSIM", "M1", "M2"]:
    activity_df[metric] /= activity_df[metric].max()

activity_df.set_index("activity").plot(
    kind="bar",
    figsize=(14, 8),
    color=["#FFB3BA", "#FFDFBA", "#BAFFC9", "#BAE1FF", "#FFFFBA"],
    edgecolor="black",
    width=0.8,
)
plt.title("Performance Across Activities", fontsize=16, weight="bold")
plt.xlabel("Activities", fontsize=14)
plt.ylabel("Normalized Metric Values", fontsize=14)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Metrics", fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(
    "../results/figs/performance_across_activities.png", dpi=600, bbox_inches="tight"
)
plt.show()
