# Quantifying the Value of Grenades in Competitive CS:GO
**An Unsupervised and Predictive Machine Learning Analysis on Early-Round Economy**  
*By: Liam Hibbs*

## Overview
Counter-Strike: Global Offensive (CS:GO) is a tactical, economy-driven 5v5 shooter where teams manage cash at the start of every round to purchase primary weapons, armor, and tactical utility (flashbangs, smoke grenades, HE grenades, and incendiaries). This project analyzes **51,618 early-round match snapshots** (filtered to the first 15 seconds of active rounds before bomb plants, `time_left >= 100`) to quantify how early-round buy decisions drive Counter-Terrorist (CT) win probability—and calculates the exact cash-equivalent value of a single tactical grenade.

---

## Methodology & Feature Engineering
* **Dataset:** CS:GO Round Winner Classification (Kaggle), spanning 97 real-time match state features across tier 1 and 2 competitive matches on 8 maps (51.38% baseline CT win rate).
* **Head-to-Head Differential Matrix ($51,618 \times 7$):** Subtracted Terrorist (T) resources from Counter-Terrorist (CT) resources (`CT - T`) to capture relative round advantage across `delta_money`, `delta_armor`, `delta_rifles`, `delta_awps`, `delta_utility`, `delta_smokes`, and `delta_flashes`.
* **Dimensionality Reduction (PCA):** Standardized features and compressed the 7 equipment differentials into 3 principal components capturing **85.58%** of total variance.
* **Unsupervised Clustering (K-Means, $k=4$):** Segmented early-round buy states without round-winner labels to uncover distinct economic archetypes.
* **Predictive Modeling (Logistic Regression):** Trained on an 80/20 train-test split (10,324 test rounds) using PCA components and cluster assignments to predict round outcomes, alongside a secondary model on standardized gear weights to derive the utility-to-cash exchange rate.

---

## Key Findings

### 1. Principal Component Analysis (PCA)
* **PC1 — General Firepower & Economy (60.58% variance):** Captures aggregate buy strength where armor, rifles, and grenades scale together during full-buy rounds.
* **PC2 — Bank Reserves vs. Hardware (14.29% variance):** Isolates unspent cash (`delta_money` loading = `0.876`) from active rifle investments (`delta_rifles` loading = `-0.307`).
* **PC3 — The AWP Bet (10.71% variance):** Separates heavy sniper investments (`delta_awps` loading = `0.832`) from standard rifle spreads (`delta_rifles` loading = `-0.334`).

### 2. K-Means Economic Archetypes ($k=4$)
| Cluster | Archetype | Mean $\Delta$ Money | Mean $\Delta$ Rifles | Mean $\Delta$ AWPs | Mean $\Delta$ Utility | CT Win Rate | Rounds |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Evenly Matched / Slight T Edge | -$6,007.75 | -0.45 | -0.02 | +0.49 | **44%** | 25,191 |
| **1** | Fully Equipped + Heavy Utility | +$1,459.12 | +1.64 | +0.77 | +12.09 | **85%** | 7,881 |
| **2** | Eco / Completely Broke | -$5,453.93 | -2.99 | -0.33 | -7.40 | **17%** | 7,968 |
| **3** | Rich Snipers (Cash Hoarding) | +$9,773.43 | -0.86 | +0.95 | +5.61 | **71%** | 10,578 |

* **Takeaway:** Comparing **Cluster 1** to **Cluster 3**, teams in Cluster 1 held **$8,314 less unspent cash** on average but invested heavily in rifles and tactical utility (+12.09 utility advantage)—resulting in a **+14% higher CT win rate (85% vs. 71%)**.

### 3. Predictive Performance & Grenade Valuation
* **Early-Round Win Prediction:** Logistic Regression predicts the round winner before combat begins with **71.14% accuracy** (F1-score `0.71`) and a **0.7763 ROC-AUC** score across 10,324 test rounds (`[[3736, 1284], [1696, 3608]]`).
* **Utility-to-Cash Exchange Rate:** Comparing standardized feature weights shows that **1 tactical grenade delivers the same win-rate boost as ~$925.74 in raw team equipment/cash**—more than $3\times$ its $200–$300 in-game purchase cost. A full 4-grenade utility set offsets roughly **$3,703** in raw gun/cash deficits.

---

## Repository Structure
* `csgo_analysis.ipynb` — Complete Python notebook containing data cleaning, feature engineering, PCA scree/loading plots, K-Means profiling, and Logistic Regression evaluation.
* `README.md` — Project methodology, statistical tables, and key findings.
* `.gitignore` — Excludes raw CSV data and Jupyter checkpoints.

## Setup & Reproduction
1. Download `csgo_round_snapshots.csv` from the [Kaggle CS:GO Round Winner Classification](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification) dataset and place it in the project root folder.
2. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
