# 📊 Clustering Algorithms Analysis

This project compares **K-Means** and **Hierarchical Agglomerative Clustering** algorithms using the **silhouette coefficient** for evaluation. All algorithms are implemented from scratch. It includes **outlier detection and removal**, and allows customization on a **synthetic 3D dataset** to determine the superior clustering algorithm based on clustering quality.

---

## 📑 Table of Contents
- 📌 [Overview](#-overview)
- ✨ [Features](#-features)
- 🧪 [Dataset](#-dataset)
- 🔎 [How It Works](#-how-it-works)
- ⚙️ [Usage](#-usage)
- 🧰 [Requirements](#-requirements)
- 📄 [License](#-license)

---

## 📌 Overview

This project evaluates the performance of two popular clustering algorithms—**K-Means** and **Hierarchical Agglomerative Clustering**—by using the **silhouette coefficient**, a metric that reflects how well each point fits within its assigned cluster.

🏆 **Goal:** Identify which algorithm performs better on a synthetic 3D dataset based on clustering quality.

---

## ✨ Features

- 🧼 **Outlier Detection and Removal**  
  Detects and removes outliers based on a user-defined threshold.

- 🔹 **K-Means Algorithm**  
  Standard implementation to partition data into K clusters.

- 🔗 **Hierarchical Agglomerative Clustering**  
  Supports multiple linkage criteria: Min, Max, Average, and Center Distance.

- 📈 **Silhouette Coefficient Calculation**  
  Computes silhouette scores for each method to evaluate cluster quality.

- 🛠️ **Customizable K Value**  
  Users can set the desired number of clusters.

---

## 🧪 Dataset

- 🎯 **Synthetic 3D dataset** of **500 points**  
- Points are randomly distributed in 3D space  
- Easily modifiable to test different scenarios  
- Built-in outlier removal based on user-specified threshold  

---

## 🔎 How It Works

1. **🧼 Outlier Detection:**  
   Removes outliers using point distance and a user-defined threshold.

2. **🔹 K-Means Clustering:**  
   Executes K-Means and computes the silhouette score.

3. **🔗 Hierarchical Clustering:**  
   Performs clustering with:
   - 🔽 Min Distance
   - 🔼 Max Distance
   - ➗ Average Distance
   - 🎯 Center Distance

4. **📊 Comparison:**  
   All silhouette scores are compared, and the highest-scoring method is declared the best.

---

## ⚙️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/clustering-algorithms-analysis.git
   cd clustering-algorithms-analysis
2. Run the script:
    ```bash
    python clustering_analysis.py
3. Input the number of clusters (K) and other parameters when prompted.
4. View silhouette scores and algorithm comparison in the output.

Requirements:
  Python 3.x
  numpy
  random
  math


---

### ✅ Tips:
- Replace `https://github.com/your-username/clustering-algorithms-analysis.git` with your actual GitHub repo URL.
- Include a `clustering_analysis.py` script and optionally a `LICENSE` file in your repo.

Would you like help generating visuals (e.g. a 3D scatter plot or cluster comparison chart) to embed in your README as well?


   
