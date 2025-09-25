# Retail Customer Segmentation using K-Means Clustering

A simple Machine Learning project demonstrating **K-Means clustering** on customer data with both a script and a basic GUI interface.

---

## What It Does

- Loads the **Mall Customers dataset**  
- Applies **K-Means clustering** to group customers based on features (like annual income and spending score)  
- Visualizes the resulting clusters  
- Provides a **GUI** for experimenting with different numbers of clusters interactively  

---

## How It Works

1. **Data Loading**  
   - The dataset (`Mall_Customers.csv`) is read using `pandas`.  

2. **Preprocessing**  
   - Selects relevant numerical features (e.g., `Annual Income`, `Spending Score`) for clustering.  

3. **Clustering**  
   - Uses `scikit-learn`’s `KMeans` to group customers into clusters.  
   - The user can change the number of clusters (`k`).  

4. **Visualization**  
   - Clusters are plotted using `matplotlib`.  
   - GUI (`gui.py`) provides controls to input cluster count and see results dynamically.  

---

## What It Explains

This project is meant to **teach/illustrate**:  
- How **unsupervised learning** (clustering) works in practice  
- How to apply **K-Means** to real-world datasets  
- The importance of choosing the right number of clusters (`k`)  
- How visualization helps interpret ML results  
- How to make a **simple GUI** around a data science model  

---

## Dataset

**Mall Customers Dataset (`Mall_Customers.csv`)**  
- Contains basic customer information.  
- Features include:  
  - `CustomerID`  
  - `Gender`  
  - `Age`  
  - `Annual Income (k$)`  
  - `Spending Score (1–100)`  

In this project, clustering is typically done on **Annual Income** vs **Spending Score**.

---

## Dependencies to Install

Make sure you have Python 3.7+ installed.  
Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Running clustering directly: 
  ```bash
    python kmeans_clustering.py
  ```
To run GUI:
  ```bash
    streamlit run gui.py
  ```
