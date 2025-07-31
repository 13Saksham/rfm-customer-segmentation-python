# Customer Segmentation with RFM Model and K-Means Clustering
## 🔍 Overview

This project demonstrates how to segment customers using the RFM (Recency, Frequency, Monetary) model combined with K-Means clustering. It helps identify patterns in customer behavior and groups them into actionable segments such as "Loyal Customers", "At Risk", and "Need Attention".

All customer data used in this project comes from OTA (Online Travel Agency) transactional data from kaggle. Customer IDs were randomized, it didn't exist in the data.

---

## 📁 Dataset

* Source: Proprietary OTA customer booking data
* Anonymized with randomized `CustomerID`
* Time Period: 1 year snapshot

Fields used:

* `Recency`: Days since last booking
* `Frequency`: Number of bookings
* `Monetary`: Total value of bookings

---

## 🧪 Step 1: Data Cleaning & Preprocessing

After loading the dataset, I:

* Removed duplicates and nulls
* Calculated RFM values
* Applied square root transformation to reduce skew


---

## 📊 Step 2: RFM Scoring

Each customer was scored on a scale of 1 to 4 based on R, F, and M quartiles using `pd.qcut`. These scores were summed into an `RFM_Score`.

```python
rfm_df['RFM_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
```

**Output**:

<img width="624" height="165" alt="image" src="https://github.com/user-attachments/assets/67f6c55a-a21d-494d-acd5-db70ff0f6e32" />


---

## 🔄 Step 3: Optimal Cluster Selection

To determine the best number of clusters for K-Means, I used:

* Elbow Method
* Silhouette Score

```python
from sklearn.metrics import silhouette_score

```

**Output**:

<img width="901" height="361" alt="image" src="https://github.com/user-attachments/assets/8a1f8816-7b71-4755-b2ca-fec90edff136" />


* Based on results, I chose **k = 4**.

---

## 📊 Step 4: Clustering with K-Means

I standardized the RFM sqrt values and applied KMeans clustering.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)
```

**Output**:

<img width="939" height="150" alt="image" src="https://github.com/user-attachments/assets/8f80a878-6247-42b4-83fe-35f015d125b9" />



---
## 📋 Step 5: Inspect distributions after preprocessing  
Before running the clustering algorithms I checked whether the square-root
transformation really reduced skewness and brought the three R ∙ F ∙ M features
onto comparable shapes.  

**Output**: 

<img width="715" height="716" alt="image" src="https://github.com/user-attachments/assets/ef09acf8-a634-4313-b0cf-59806188608a" />

## 📋 Step 6: Summarise the resulting clusters
Once the optimal `k` was chosen I calculated the mean **Recency,
Frequency,** and **Monetary** values for each cluster and counted how many
customers fell into every group. 

**Output**:

<img width="401" height="159" alt="image" src="https://github.com/user-attachments/assets/7f4fdc3f-190b-494d-aa53-a345748bd219" />

## 📋 Step 7: Segment Mapping

I assigned human-readable labels to each cluster based on their average RFM behavior:

```python
cluster_labels = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Loyal Customers',
    3: 'Need Attention'
}
rfm_df['Segment'] = rfm_df['Cluster'].map(cluster_labels)
```

**Output**:

<img width="272" height="158" alt="image" src="https://github.com/user-attachments/assets/17dc3fdc-b5fd-4ced-9cca-03faf8d3043e" />


---

## 📈 Step 8: Final Analysis & Insights

I aggregated cluster statistics and visualized them using bar plots.

```python
cluster_summary = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
```

**Output**:

<img width="900" height="546" alt="image" src="https://github.com/user-attachments/assets/e93b2379-ff1d-4cf9-96a9-827172f1880f" />

---

## 🔺 Segment Takeaways

| Segment         | Insight                                           |
| --------------- | ------------------------------------------------- |
| Champions       | Recently active, high value, and frequent bookers |
| Loyal Customers | Consistent bookers with solid spend               |
| At Risk         | Haven't booked in a while but used to spend a lot |
| Need Attention  | Low engagement and low spend                      |

---

## 🪧 Tools Used

* Python (Pandas, Scikit-learn, Seaborn, Matplotlib)
* JupyterLab (GitHub Codespaces)

---

## 📚 How to Use This Repo

```bash
git clone https://github.com/13Saksham/rfm-customer-segmentation-python.git
cd rfm-customer-segmentation-python

```

---

## 🙏 Acknowledgements

* [Kaggle Datasets](https://www.kaggle.com/)
* Concepts inspired by DataCamp, StackOverflow, and self-practice

