# CS:GO Competitive Economy Analysis (PCA & K-Means)

Analysis of 51,618 early-round CS:GO snapshots evaluating how economic and utility advantages impact Counter-Terrorist (CT) round win rates.

## Key Findings
* **PCA:** PC1 (aggregate firepower and financial dominance) accounts for **60.58%** of total variance; PC2 (utility density vs. raw hardware) accounts for **14.29%**.
* **Clustering:** K-Means ($k=4$) separates rounds into distinct economic states, ranging from heavy T-side control (**17% CT win rate** in Cluster 2) to dominant CT full-buys with heavy utility (**85% CT win rate** in Cluster 1).

## Setup
1. Download `csgo_round_snapshots.csv` (Kaggle CS:GO Round Winner Classification dataset) and place it in the root directory.
2. Install dependencies: `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`
3. Run `csgo_analysis.ipynb`.
