"""
# Python 3.8.6
pip install numpy pandas
"""
import os
import pandas as pd


def combine_metrics(
    base_dir="../results/metrics", output_file="../results/metrics.csv"
):
    combined_data = []

    # Iterate over setting directories
    for setting in ["setting1", "setting2"]:
        setting_path = os.path.join(base_dir, setting)
        if not os.path.exists(setting_path):
            print(f"Setting path {setting_path} does not exist. Skipping.")
            continue

        # Iterate over metric files in each setting
        for metric_file in os.listdir(setting_path):
            if metric_file.endswith("_metrics.csv"):
                file_path = os.path.join(setting_path, metric_file)
                print(f"Processing {file_path}...")

                # Read the CSV and append to the combined list
                try:
                    data = pd.read_csv(file_path)
                    data["setting"] = setting  # Add a column for the setting
                    combined_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Concatenate all data and save to a single CSV
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined metrics saved to {output_file}")
    else:
        print("No data found to combine.")


def generate_summary_table(
    metrics_csv="../results/metrics.csv", output_csv="../results/summary_table.csv"
):
    # Load the combined metrics
    metrics = pd.read_csv(metrics_csv)

    # Calculate SSIM (Similarity Average) for each src-target pair
    grouped = metrics.groupby(["src_country", "tgt_country"])
    summary_data = []

    for (src, tgt), group in grouped:
        ssim = group["similarity"].mean()  # Average SSIM

        # Metric M1: % of cases where (delta1 < 0 and delta2 > 0)
        m1 = ((group["delta1"] < 0) & (group["delta2"] > 0)).mean() * 100

        # Metric M2: % of cases where (delta2 - delta1 > 0)
        m2 = ((group["delta2"] - group["delta1"]) > 0).mean() * 100

        summary_data.append(
            {
                "src_country": src,
                "tgt_country": tgt,
                "SSIM": round(ssim, 2),
                "M1": round(m1, 2),
                "M2": round(m2, 2),
            }
        )

    # Create DataFrame for the summary
    summary_df = pd.DataFrame(summary_data)

    # Add averages for each source country
    averages = []
    for src, group in summary_df.groupby("src_country"):
        avg_ssim = group["SSIM"].mean()
        avg_m1 = group["M1"].mean()
        avg_m2 = group["M2"].mean()
        averages.append(
            {
                "src_country": src,
                "tgt_country": "Average",
                "SSIM": round(avg_ssim, 2),
                "M1": round(avg_m1, 2),
                "M2": round(avg_m2, 2),
            }
        )

    # Add overall averages
    overall = {
        "src_country": "Overall Average",
        "tgt_country": "",
        "SSIM": round(summary_df["SSIM"].mean(), 2),
        "M1": round(summary_df["M1"].mean(), 2),
        "M2": round(summary_df["M2"].mean(), 2),
    }

    # Append averages to the summary DataFrame
    summary_df = pd.concat([summary_df, pd.DataFrame(averages)], ignore_index=True)
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    # Save to a CSV file
    summary_df.to_csv(output_csv, index=False)
    print(f"Summary table saved to {output_csv}")


if __name__ == "__main__":
    if not os.path.exists("../results/metrics.csv"):
        combine_metrics()
    if not os.path.exists("../results/summary_table.csv"):
        generate_summary_table()
