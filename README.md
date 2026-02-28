# DataAnalytics_ML_AI_DataMining_Works

# 01.  NLP Discovery: Classification, Clustering, and Sentiment Analysis

## 📊 Dataset
**Twitter US Airline Sentiment**
* **Source:** A collection of ~14,640 tweets regarding major U.S. airlines.
* **Features:** Includes tweet text, sentiment labels (positive, neutral, negative), and metadata like negative reasons (e.g., "Late Flight" or "Customer Service Issue").

---

## 🎯 Project Goal
The goal of this project is to demonstrate the practical application of both **Supervised** and **Unsupervised Learning** in Natural Language Processing (NLP). By using real-world social media data, the project explores how machines can interpret human emotion and automatically organize large volumes of unstructured text into meaningful categories.

---

## 🛠️ Tasks Performed
1.  **Text Classification (Supervised):** Training a model to automatically sort tweets into three predefined categories: *Positive*, *Neutral*, or *Negative*. This simulates real-world systems like automated customer support routing.
2.  **Text Clustering (Unsupervised):** Grouping tweets based on semantic similarity without using labels. This helps discover "hidden" patterns, such as identifying a sudden cluster of complaints related specifically to "baggage" or "delays."
3.  **Sentiment Analysis:** Applying specialized NLP techniques to measure the emotional tone of the text. This task focuses on understanding the intensity and nature of public opinion toward different airline brands.

---

## 🚀 Implementation
You can view the full code, data preprocessing, and model implementation in the Google Colab notebook below:

**[View Notebook on Google Colab](https://colab.research.google.com/drive/15iv3-VGmfoFE1GeBQ0cc18BnLaRKoHoq?usp=sharing#scrollTo=h45fCDeOg4Bc)**
# 02.  📊 Marketing Campaign Analysis & Customer Segmentation

This project performs an end-to-end data analysis and unsupervised machine learning pipeline to segment customers based on their demographics and purchasing behavior. By identifying distinct customer profiles, businesses can better tailor their marketing strategies and resource allocation.

---

## 🚀 Project Overview
The primary goal is to transform raw marketing data into actionable insights using clustering techniques. The workflow covers everything from initial data cleaning and feature engineering to dimensionality reduction and cluster profiling.



## 🛠️ Key Stages of Analysis

### 1. Data Inspection & Cleaning
* **Initial Audit:** Handled missing values in the `Income` column and corrected data types (e.g., converting `Dt_Customer` to datetime).
* **Outlier Management:** Filtered records to ensure realistic distributions for `Age` (18–90) and `Income` ($0–$200k).

### 2. Feature Engineering
Created new, high-impact variables to enrich the model:
* **`Age` & `Customer_For`:** Derived from birth years and enrollment dates.
* **`Spent`:** Aggregated total spending across all product categories.
* **`Family_Size` & `Is_Parent`:** Consolidating marital status and household composition.

### 3. Dimensionality Reduction (PCA)
To handle the "curse of dimensionality" and improve clustering performance, **Principal Component Analysis (PCA)** was applied to reduce the scaled numerical features into **3 principal components**. This also allows for 3D visualization of the customer segments.

### 4. Optimal Clustering & Segmentation
* **The Elbow Method:** Utilized `KElbowVisualizer` to mathematically determine the optimal number of clusters.
* **Agglomerative Clustering:** Implemented hierarchical clustering to group customers into **4 distinct segments**.



### 5. Cluster Profiling & Evaluation
The identified clusters were evaluated based on:
* **Spending Power:** Relationship between `Income` and `Spent`.
* **Campaign Engagement:** Analysis of accepted promotions and deals purchased.
* **Demographics:** Joint plots comparing `Age` and `Customer_For` across different segments.

---

## 💻 Implementation & Code
The full technical implementation, including data visualizations and model evaluation, can be found in the interactive notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1yKJVVp82g2YlO7I1kbeDnpSX0kbr1-0b#scrollTo=01bcf061)**

---
  
# 03.  Comparative Analysis: Deep Learning for Banana and Orange Classification & Detection

This project provides a comprehensive end-to-end computer vision workflow comparing various architectures for the classification and localization of fruit (Bananas and Oranges). It bridges the gap between traditional Machine Learning and state-of-the-art Deep Learning models.

---

## 🏗️ Project Architecture

### 1. Dataset Preparation
* **Classification:** Utilizes the **Fruits-360** dataset for high-quality, single-object fruit images.
* **Object Detection:** Custom multi-fruit dataset annotated in **YOLO format** for bounding box prediction.

### 2. Image Classification (The Three-Tier Approach)
We evaluate three distinct levels of complexity to understand the evolution of image recognition:
* **Traditional ML:** Color Histogram extraction + Support Vector Machine (SVM).
* **Custom CNN:** A deep learning model built from scratch to learn spatial hierarchies.
* **Transfer Learning:** Fine-tuning **MobileNetV2** for high accuracy with low computational overhead.



### 3. Object Detection (Localization & Identification)
Comparison of three industry-standard detectors based on the "Speed vs. Accuracy" trade-off:
* **YOLOv8n (Ultralytics):** Optimized for real-time performance and edge deployment.
* **SSDLite320:** A lightweight Single Shot Detector designed for mobile devices.
* **Faster R-CNN:** A high-precision, two-stage detector for maximum localization accuracy.



## 📊 Evaluation Metrics
Models are rigorously tested using the following benchmarks:
* **Classification:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
* **Detection:** mean Average Precision (**mAP@50** and **mAP@50-95**).
* **Efficiency:** Inference speed (ms/image) and Frames Per Second (FPS).


## 📜 Critical Comparative Analysis
The project concludes with a detailed discussion on:
* **Accuracy-Speed Trade-offs:** Which model wins for real-time video vs. static high-res analysis?
* **Training Difficulty:** Computational requirements and convergence rates.
* **Recommendation:** Evidence-based selection of the best detector for real-world agricultural or retail scenarios.


## 💻 Technical Implementation
Explore the full training logs, visualizations, and comparative charts in the interactive notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/10O7s4OwA3XKsYc4Dg2N-qHjYwXyLuZ4o?usp=sharing#scrollTo=eee23334)**