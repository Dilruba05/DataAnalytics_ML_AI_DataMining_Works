# DataAnalytics_ML_AI_DataMining_Works


# 🚀 Data Science & AI Portfolio
Welcome to my project repository! Below is a curated summary of 18 projects demonstrating my expertise in Machine Learning, NLP, Computer Vision, and Reinforcement Learning, followed by detailed technical breakdowns for each implementation.

---

## 📂 Portfolio Overview

| # | Category | Project Title | Key Tech Stack | Link |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **NLP** | Twitter Sentiment Analysis | TextBlob, VADER, Transformers | [Open Colab 🔗](https://colab.research.google.com/drive/1ZQX8Eam0EshocTujlA0hF396pcvyxred) |
| **02** | **NLP** | SMS Spam Detection: NLTK Pipeline | NLTK, Stopwords, Tokenization | [Open Colab 🔗](https://colab.research.google.com/drive/12cskgKD47gk3qHRq89CfMS7ZzYXRaUyn) |
| **03** | **NLP** | SMS Analytics & EDA | WordCloud, Frequency Analysis | [Open Colab 🔗](https://colab.research.google.com/drive/14Xskqbadn8rdTOBXY2Dt08QDXn4hF-c8) |
| **04** | **CV** | Diverse Object Detection | YOLOv8, skimage, OpenCV | [Open Colab 🔗](https://colab.research.google.com/drive/1Dk5U2vMJrpkssVuBYQjB_fNliaEfSE6v) |
| **05** | **RL** | Algorithm Comparison (Pendulum-v1) | PPO, A2C, TD3, Gymnasium | [Open Colab 🔗](https://colab.research.google.com/drive/1bGYF6QU3SlSHUfcUEzcoUwkoAj7dkNz6) |
| **06** | **ML** | Housing Value Prediction | DecisionTreeRegressor, MSE/R² | [Open Colab 🔗](https://colab.research.google.com/drive/1zcnDRer7v7pBzXATRqwJ2YcUTNL1aBZK) |
| **07** | **ML** | Customer Segmentation (K-Means) | Scikit-Learn, StandardScaler | [Open Colab 🔗](https://colab.research.google.com/drive/1DRy-0KrirAAelPnosIct3HUp3eT0hfin) |
| **08** | **ML** | Titanic Survival: EDA & Classif. | Logistic Regression, Encoding | [Open Colab 🔗](https://colab.research.google.com/drive/1unnS2NgqvGfl0rFFhSTZXaliYcmwDFHe) |
| **09** | **ML** | Iris Species Classification | SVM (Support Vector Machines) | [Open Colab 🔗](https://colab.research.google.com/drive/1KjWKt5zjekUk_0mwILv_ckWbAqPhs_pd) |
| **10** | **ML** | Heart Disease: Binary vs. Multiclass | Random Forest, ROC-AUC | [Open Colab 🔗](https://colab.research.google.com/drive/1l6ot3dpvMzf6a07UQ3LCSK-ORply0wVU) |
| **11** | **ML** | Credit Card Behavioral Clustering | K-Means, Marketing Insights | [Open Colab 🔗](https://colab.research.google.com/drive/1jnXbXoKwF7InGVHxG4quYH-N-ofAowZ6) |
| **12** | **ML** | Scikit-Learn Model Evaluation | Logistic/Ridge, 5-Fold CV | [Open Colab 🔗](https://colab.research.google.com/drive/1lw7kFeyQgyBLAJjre5u32Zclo4of3zYz) |
| **13** | **ML** | Market Basket Analysis | Apriori, Association Rules | [Open Colab 🔗](https://colab.research.google.com/drive/10w9QKpHSnV-FeLJHp-z98saC4Zx02LdG) |
| **14** | **ML** | Anomaly Detection (Breast Cancer) | Isolation Forest, LOF | [Open Colab 🔗](https://colab.research.google.com/drive/1rc8VrN1vusP8MG0k_pmMWpWFdyhk9zTn) |

---

## 🧪 Tech Stack Summary
* **ML/Data:** Scikit-Learn, Pandas, NumPy, MLxtend
* **Deep Learning/CV:** YOLOv8, OpenCV, Hugging Face
* **NLP:** NLTK, VADER, TextBlob
* **RL:** Gymnasium, Stable Baselines3

---

# 01.  NLP Discovery: Classification, Clustering, and Sentiment Analysis

## 📊 Dataset
**Twitter US Airline Sentiment**
* **Source:** A collection of ~14,640 tweets regarding major U.S. airlines.
* **Features:** Includes tweet text, sentiment labels (positive, neutral, negative), and metadata like negative reasons (e.g., "Late Flight" or "Customer Service Issue").


## 🎯 Project Goal
The goal of this project is to demonstrate the practical application of both **Supervised** and **Unsupervised Learning** in Natural Language Processing (NLP). By using real-world social media data, the project explores how machines can interpret human emotion and automatically organize large volumes of unstructured text into meaningful categories.


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

---

# 04. 🎓 Student Engagement Prediction & Explainable AI (XAI)

This project addresses the "Black Box" nature of machine learning in education. By leveraging behavioral data from Learning Management Systems (LMS), I developed a predictive model that not only identifies student engagement levels but also provides transparent explanations for *why* a student is categorized as high or low risk.


## 🎯 Project Goals
* **Predictive Modeling:** Accurately classify student engagement into **Low**, **Moderate**, or **High** categories.
* **XAI Integration:** Implement **SHAP** and **LIME** to provide actionable insights for educators.
* **Strategic Intervention:** Identify key behavioral triggers (e.g., low forum activity vs. high video consumption) to help course designers improve retention.


## 📊 Dataset & Methodology

### 1. Data Source
* **Source:** Behavioral data harvested from **Learning Management Systems (LMS)**.
* **Key Features:** Time spent on materials, frequency of logins, quiz participation, video completion rates, and assignment submission patterns.

### 2. Machine Learning Workflow
* **Preprocessing:** Defining robust metrics for "Engagement" and cleaning raw behavioral logs.
* **Classification:** Training various models to find the best balance between accuracy and interpretability.



### 3. Explainability (The XAI Layer)
To bridge the gap between data and human decision-making, we use:
* **SHAP:** Used for **Global Interpretability**—identifying which features matter most across the entire student population.
* **LIME:** Used for **Local Interpretability**—explaining specific, individual student predictions to help teachers understand a single student's unique struggle.



## 💡 Key Insights & Outcomes
* **Behavioral Mapping:** Identified that specific combinations of activity (like late-night logins vs. consistent quiz attempts) are higher predictors of success than total time spent.
* **Risk Detection:** Created a framework to flag at-risk students with a clear "Reason Code" based on the XAI output.
* **Design Guidance:** Provided evidence-based suggestions for educators to optimize course flow and digital interventions.

---

## 🚀 Technical Implementation
The complete code, from data ingestion to the generation of XAI explanation plots, is available here:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1q_fedwouif9ZdfYtWJo-deVYxxmRBmMO?usp=sharing)**

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Science:** Pandas, Scikit-Learn
* **Explainable AI:** SHAP, LIME
* **Visualization:** Seaborn, Matplotlib

# 05. 🤖 Reinforcement Learning: Algorithm Comparison on Pendulum-v1

This project features a systematic performance analysis of three prominent Reinforcement Learning (RL) algorithms—**PPO**, **A2C**, and **TD3**—within the **Pendulum-v1** continuous control environment (Gymnasium). The study evaluates how different actor-critic architectures handle the challenges of torque control and gravity compensation.


## 🌍 Environment: Pendulum-v1
The goal is to keep a frictionless pendulum standing upright by applying torque.
* **Action Space:** Continuous (Torque).
* **Observation Space:** Trigonometric functions of the angle ($\sin\theta$, $\cos\theta$) and angular velocity.
* **Challenge:** The agent must learn to swing up and balance, requiring precise continuous control.



---

## 🛠️ Methodology & Training
Three distinct algorithms were trained for a consistent budget of **30,000 timesteps** to ensure a fair "head-to-head" comparison:

1.  **PPO (Proximal Policy Optimization):** An on-policy gradient method known for stability and reliability.
2.  **A2C (Advantage Actor-Critic):** A synchronous, deterministic variant of A3C that balances the policy (Actor) and value function (Critic).
3.  **TD3 (Twin Delayed DDPG):** An off-policy algorithm designed to reduce overestimation bias in continuous action spaces.

### 📈 Training Workflow
* **Logging:** Episode-level rewards were captured using the `Monitor` wrapper.
* **Smoothing:** Applied rolling averages to training logs to visualize learning trends clearly.
* **Reproducibility:** Used modular helper functions and set seeds to ensure consistent results.





## 📊 Performance Evaluation
The models were put through rigorous testing over multiple evaluation episodes. Key metrics included:
* **Mean Total Reward:** How well the agent learned to balance the pendulum.
* **Reward Consistency:** Measured via Variance and Standard Deviation to check for training stability.
* **Inference Speed:** Assessing the computational efficiency of each model.


## 💡 Comparative Insights
The project concludes with a deep dive into the strengths and weaknesses of each approach:
* **PPO vs. A2C:** Analyzing why on-policy methods often provide smoother convergence in this specific environment.
* **TD3 Performance:** Evaluating if the added complexity of "Twin Critics" provided a significant advantage for a relatively simple pendulum task.
* **Trade-offs:** A discussion on sample efficiency versus wall-clock training time.


## 🚀 Technical Implementation
The complete implementation, including video recordings of the agents and reward visualizations, is available here:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1bGYF6QU3SlSHUfcUEzcoUwkoAj7dkNz6#scrollTo=06238624)**

---

## 🧪 Tech Stack
* **Framework:** Gymnasium (OpenAI Gym)
* **RL Library:** Stable Baselines3
* **Core Tools:** Python, PyTorch, NumPy
* **Visualization:** Matplotlib, Seaborn, OpenCV (for video rendering)

# 06. 🏠 Housing Value Prediction: Decision Tree Regression

This project implements a supervised learning pipeline to predict housing prices using a **Decision Tree Regressor**. By analyzing various property features, the model learns to estimate market values and provides a clear visualization of predictive accuracy versus actual real estate data.

---

## 🎯 Project Objective
The goal is to build a robust regression model capable of mapping complex housing features to a continuous target variable (Price). The project focuses on the trade-off between model complexity and generalization, ensuring the tree does not overfit the training data.

---

## 🛠️ Data Pipeline

### 1. Data Preparation
* **Dataset:** `housing_dataset.csv` containing key features like square footage, location metrics, and property age.
* **Preprocessing:** Separation of independent features ($X$) and the target variable ($y$).
* **Data Splitting:** Implementation of a Train/Test split to validate the model on unseen data, ensuring unbiased performance reporting.

### 2. Model Implementation
* **Algorithm:** `DecisionTreeRegressor` (Scikit-Learn).
* **Training:** The model was fitted to the training set, allowing the tree to partition the data space into hyper-rectangles based on feature thresholds.



---

## 📊 Performance Evaluation
To determine the model's reliability, three standard regression metrics were utilized:

| Metric | Definition | Purpose |
| :--- | :--- | :--- |
| **MSE** | Mean Squared Error | Penalizes larger errors more heavily; useful for detecting outliers. |
| **MAE** | Mean Absolute Error | Provides a direct "average error" in the same units as the house price. |
| **R²** | Coefficient of Determination | Explains what percentage of the price variance is captured by the model. |

### 📈 Visual Validation
A **Scatter Plot (Actual vs. Predicted)** was generated to visually audit the model. A perfect model would show all points along a $45^\circ$ diagonal line; deviations from this line highlight specific price ranges where the model may require further tuning.



---

## 🚀 Technical Implementation
The complete Python code, data cleaning steps, and evaluation charts are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1zcnDRer7v7pBzXATRqwJ2YcUTNL1aBZK?usp=sharing)**

---

## 🧪 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn


# 07. 💳 Customer Segmentation: K-Means Clustering for Credit Card Data

This project utilizes **Unsupervised Machine Learning** to identify distinct consumer archetypes within credit card usage data. By applying the **K-Means Clustering** algorithm, raw transaction histories are transformed into actionable customer segments to drive data-driven marketing and financial strategies.

---

## 🎯 Project Objective
The goal is to partition a diverse customer base into groups with similar spending behaviors, credit limits, and payment patterns. This allows for personalized product offerings, optimized interest rates, and improved customer retention.


## 🛠️ Data Pipeline & Clustering
* **Preprocessing:** Handled missing values via imputation and applied **StandardScaler** to normalize features like `Balance` and `Credit_Limit` for distance-based calculations.
* **K-Means Workflow:** Optimized the number of clusters ($k$) using the **Elbow Method** to minimize intra-cluster variance.

* **Segment Identification:** Classified customers into profiles such as "High-Spenders," "Installment Users," and "Prudent Payers" to enable targeted marketing and risk management.



## 🚀 Technical Implementation
The complete data pipeline, including cluster profiling and visualizations, is available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1DRy-0KrirAAelPnosIct3HUp3eT0hfin?usp=sharing)**

---

## 🧪 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (K-Means)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

# 08. 🚢 Titanic Survival Prediction: EDA & Binary Classification

This project implements a classic machine learning workflow to predict passenger survival on the Titanic. It covers the entire pipeline from exploratory data analysis (EDA) and feature engineering to training and evaluating a **Logistic Regression** model.

---

## 🎯 Project Objective
The goal is to determine the key factors that influenced survival rates (such as age, gender, and class) and build a classification model that accurately predicts whether a passenger survived the disaster based on these features.


## 🛠️ Data Pipeline & Methodology
* **Exploratory Data Analysis (EDA):** Visualized survival distributions across various demographics to identify significant patterns, such as the "women and children first" protocol.

* **Data Cleaning:** Handled missing values through median imputation for `Age` and removal of high-sparsity columns like `Cabin`. Irrelevant features (e.g., `Ticket`, `Name`) were dropped to reduce noise.
* **Feature Encoding:** Converted categorical variables (`Sex`, `Embarked`) into numerical format using **LabelEncoder** for model compatibility.
* **Model Training:** Utilized **Logistic Regression** to establish a baseline for binary classification, splitting the data into training and testing sets for unbiased validation.

* **Model Evaluation:** Performance was assessed using a **Confusion Matrix** and **Classification Report** (Precision, Recall, and F1-Score) to ensure a balanced view of predictive power beyond simple accuracy.



## 🚀 Technical Implementation
The complete data cleaning process, statistical analysis, and model performance metrics are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1unnS2NgqvGfl0rFFhSTZXaliYcmwDFHe?usp=sharing)**

---

## 🧪 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn


# 09. 🌸 Iris Species Classification: Support Vector Machines (SVM)

This project explores the classic Iris dataset to classify three species of flowers—Setosa, Versicolor, and Virginica—based on their morphological measurements. By utilizing a **Support Vector Machine (SVM)**, the model identifies optimal decision boundaries to separate these biological classes.


## 🎯 Project Objective
The goal is to build a high-precision classifier that leverages the distinct physical characteristics of petals and sepals to automate species identification. The project emphasizes the importance of feature relationships and linear versus non-linear separability.


## 🛠️ Data Insights & Pipeline
* **Dataset Characteristics:** A perfectly balanced dataset of 150 samples (50 per species). Features include sepal/petal length and width.
* **Exploratory Data Analysis (EDA):** Initial inspection via `sns.pairplot` revealed that **Setosa** is linearly separable from other species, while **Versicolor** and **Virginica** exhibit slight overlap in feature space.

* **Pre-processing:** Verified a clean dataset with zero missing values and confirmed numerical data types for all feature inputs.
* **Model Selection:** Implemented **SVM (Support Vector Machine)**, an ideal choice for this dataset due to its effectiveness in high-dimensional spaces and its ability to define clear margins between classes.



## 🚀 Technical Implementation
The complete analysis, including the statistical breakdown and the pairplot visualizations, can be found in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1KjWKt5zjekUk_0mwILv_ckWbAqPhs_pd#scrollTo=2e9e50ec)**

---

## 🧪 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (SVM)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib

# 10. ❤️ Heart Disease Classification: Binary vs. Multiclass Analysis

This project explores the predictive power of Machine Learning in healthcare, specifically focusing on cardiovascular diagnostics. It compares the efficiency of **Logistic Regression** and **Random Forest** across two distinct challenges: simple presence-absence detection (Binary) and disease severity grading (Multiclass).


## 🎯 Project Objectives
1.  **Binary Classification:** Predict whether a patient has heart disease (1) or not (0).
2.  **Multiclass Classification:** Predict the specific severity level of heart disease across 5 distinct classes (0 to 4).
3.  **Comparative Evaluation:** Analyze how model performance degrades when moving from binary labels to complex, imbalanced multi-class data.

---

## 🔬 Methodology & Key Results

### Phase 1: Binary Classification
Focused on high-accuracy screening using two competitive classifiers.
* **Logistic Regression:** Achieved an accuracy of **0.848** with a high **ROC-AUC of 0.920**, demonstrating excellent class separation.
* **Random Forest:** Emerged as the top performer with an accuracy of **0.853** and a **ROC-AUC of 0.930**.


### Phase 2: Multiclass Classification (Disease Severity)
A more complex task using a Random Forest classifier restricted to numeric features.
* **Performance:** Overall accuracy dropped significantly to **0.484**.
* **The Imbalance Challenge:** The model performed well on the majority class (Class 0) but struggled with minority classes (2, 3, and 4). 
* **Critical Finding:** Class 4 (highest severity) yielded **0 precision and recall**, highlighting a significant "Class Imbalance" issue where the model failed to identify any instances of the rarest category.



## 💡 Summary of Insights
* **Model Robustness:** Random Forest showed a marginal lead in binary tasks due to its ability to handle non-linear relationships.
* **Data Quality:** The multiclass phase demonstrated that a simplified feature set and skewed data distribution are primary barriers to accurate medical grading.
* **Future Directions:** Improving the multiclass model would require advanced techniques like **SMOTE (Oversampling)** or weighted loss functions to account for the underrepresented severity levels.


## 🚀 Technical Implementation
The complete data analysis, performance reports, and class distribution charts are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1l6ot3dpvMzf6a07UQ3LCSK-ORply0wVU#scrollTo=1NAkZRMYvkR0)**


## 🧪 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest)
* **Evaluation:** ROC-AUC, Precision-Recall, Confusion Matrix
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn


# 11. 📩 SMS Spam Detection: Natural Language Processing (NLP)

This project features a specialized **NLP pipeline** designed to filter mobile messages. Using the **NLTK** library, it transforms raw, unstructured SMS text into a cleaned format to distinguish between 'spam' (fraudulent) and 'ham' (legitimate) communications.


## 🎯 Project Objective
To build an efficient text-cleaning engine that isolates high-impact keywords by stripping away linguistic noise such as punctuation and common stopwards.


## 🛠️ Data & NLP Pipeline
* **Dataset:** SMS messages labeled as **Ham** or **Spam**.
* **Preprocessing (NLTK):**
    * **Tokenization:** Segmenting sentences into individual words.
    * **Noise Removal:** Stripping punctuation and special characters.
    * **Stopword Filtering:** Removing common filler words (e.g., "the", "is") using the NLTK corpus.

* **Feature Engineering:** Implemented a reusable cleaning function to prepare a "Bag of Words" for downstream machine learning models.


## 🚀 Technical Implementation
The complete text-cleaning logic and data exploration charts are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/12cskgKD47gk3qHRq89CfMS7ZzYXRaUyn#scrollTo=7f90ae27)**

---

## 🧪 Tech Stack
* **NLP Library:** NLTK (Natural Language Toolkit)
* **Data Science:** Python, Pandas, NumPy
* **Visualization:** WordCloud, Matplotlib


# 12. 📑 SMS Analytics: Advanced Preprocessing & EDA

This project focuses on the critical initial phases of building an **SMS Spam Detection system**. It transitions from raw, noisy text to a structured, feature-rich dataset through rigorous cleaning and quantitative linguistic analysis.

---

## 🎯 Project Objective
To engineer a high-fidelity text-cleaning pipeline and perform **Exploratory Data Analysis (EDA)** to identify the dominant vocabulary and patterns in mobile communications.

---

## 🛠️ Methodology & Insights
* **Text Refinement:** Implemented a custom pipeline for punctuation removal, lowercase normalization, and stopword filtering.
* **Domain-Specific Filtering:** Removed contextually irrelevant tokens (e.g., 'jurong', 'crazy') to sharpen the model's focus on predictive terms.

* **Visual Linguistics:** Generated a **Word Cloud** to immediately identify the most frequent terms across the dataset.

* **Quantitative Analysis:** Isolated the **Top 20 most common words** (e.g., 'call', 'free', 'ur'), providing the statistical foundation for future feature engineering and classification.

---

## 🚀 Technical Implementation
The complete preprocessing logic and frequency analysis charts are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/14Xskqbadn8rdTOBXY2Dt08QDXn4hF-c8#scrollTo=68503b62)**

---

## 🧪 Tech Stack
* **Language:** Python
* **NLP Library:** NLTK
* **Data Science:** Pandas, NumPy
* **Visualization:** WordCloud, Seaborn, Matplotlib


# 13. 🔍 Multi-Domain Object Detection: Traditional & Deep Learning

This project showcases a versatile approach to object detection, comparing classical image processing with state-of-the-art **YOLOv8** architectures. It spans diverse domains, from celestial blob detection to real-time traffic and wildlife identification.

---

## 🎯 Project Objective
To demonstrate the adaptability of detection algorithms across varying scales and contexts, including astronomical data, urban environments, and natural habitats.


## 🛠️ Methodology & Outcomes

### 1. Astronomical Blob Detection
* **Technique:** Laplacian of Gaussian (LoG) via `skimage`.
* **Result:** Successfully localized **285 stars** in celestial imagery using traditional computer vision.


### 2. Urban & Wildlife Detection (YOLOv8)
* **Traffic:** Utilized **YOLOv8n** to detect **29 cars and 1 truck**, proving efficiency for real-time monitoring.
* **Wildlife:** Applied the high-accuracy **YOLOv8x** model to identify a diverse set of animals (Elephants, Zebras, Giraffes).


### 3. Domestic Object Recognition
* **Result:** Identified a variety of household items including **bananas, apples, oranges, and broccoli**, highlighting the model's granular classification capabilities.

---

## 🚀 Technical Implementation
The complete detection pipeline, including model weight selection and annotated output visualizations, is available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1Dk5U2vMJrpkssVuBYQjB_fNliaEfSE6v?usp=sharing)**

---

## 🧪 Tech Stack
* **Deep Learning:** YOLOv8 (Ultralytics)
* **Image Processing:** Scikit-Image (`skimage`)
* **Libraries:** OpenCV, Matplotlib, Python


# 14. 💳 Credit Card Segmentation: Behavioral Clustering & Insights

This project utilizes **K-Means Clustering** to segment credit card users into distinct behavioral profiles. By analyzing credit limits and engagement channels, the model identifies specific customer archetypes to drive personalized financial services.


## 🎯 Project Objective
To partition a customer database into actionable segments, allowing for optimized credit limit allocations and targeted digital transformation strategies.


## 🛠️ Data Insights & Segmentation
* **Clustering Methodology:** Applied K-Means (non-PCA) to isolate two primary customer segments based on `Avg_Credit_Limit` and `Total_Credit_Cards`.


### 📊 Segment Profiles
* **Cluster 0 (Standard):** Characterized by lower credit limits, fewer cards, and high dependency on phone support (manual banking).
* **Cluster 1 (Premium/Digital):** High-limit users with multiple cards and high engagement across online and in-person banking channels.


## 💡 Strategic Recommendations
* **Digital Migration (Cluster 0):** Promote online services to reduce call volumes and offer incremental credit limit increases to incentivize usage.
* **Loyalty & Retention (Cluster 1):** Deploy premium card products and personalized financial advisory services to maximize lifetime value.



## 🚀 Technical Implementation
The complete clustering logic, scatter plot visualizations, and feature distribution analysis are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1jnXbXoKwF7InGVHxG4quYH-ofAowZ6#scrollTo=c23cc652)**

---

## 🧪 Tech Stack
* **Machine Learning:** Scikit-Learn (K-Means)
* **Data Science:** Python, Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib


# 15. 🎭 Comparative Sentiment Analysis: TextBlob, VADER, and Transformers

This project evaluates the nuances of Natural Language Processing (NLP) by comparing three distinct sentiment analysis methodologies. By testing clear and ambiguous statements, the project highlights the strengths of rule-based systems versus deep learning models in capturing human emotion.

---

## 🎯 Project Objective
To benchmark **TextBlob**, **VADER**, and **Hugging Face Transformers** against real-world feedback to determine which architecture most accurately handles linguistic nuance and "borderline" sentiment.

---

## 📊 Methodology & Comparative Results

### 1. The Multi-Model Approach
* **TextBlob:** Rule-based polarity/subjectivity scoring.
* **VADER:** Lexicon and rule-based tool specifically attuned to social media sentiments.
* **Transformers:** State-of-the-art deep learning model (Hugging Face) for context-aware classification.

### 2. Key Findings: Accuracy vs. Nuance
| Input Text | TextBlob | VADER | Transformer |
| :--- | :--- | :--- | :--- |
| "Lecture was well structured..." | ✅ Positive | ✅ Positive | ✅ Positive (High Conf.) |
| "Service is terrible..." | ❌ Negative | ❌ Negative | ❌ Negative (High Conf.) |
| "Content is okay, nothing special." | ✅ Positive | ❌ Negative | ❌ Negative |



**Critical Insight:** While all models agree on extreme sentiments, "Text 3" reveals that rule-based models (TextBlob) can struggle with phrases like "nothing special," whereas Transformers correctly interpret the underlying negative tone.



## 🚀 Technical Implementation
The complete comparative logic, model loading scripts, and sentiment score visualizations are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1ZQX8Eam0EshocTujlA0hF396pcvyxred#scrollTo=3e6f0c1c)**

---

## 🧪 Tech Stack
* **NLP Framework:** Hugging Face Transformers
* **Libraries:** VADER, TextBlob
* **Data Science:** Python, Pandas
* **Visualization:** Matplotlib, Seaborn


# 16. 📉 Scikit-Learn Case Study: Dual Model Evaluation

This project provides a comprehensive benchmarking of **Classification** and **Regression** pipelines using Scikit-Learn. By applying **Logistic Regression** to medical data and **Ridge Regression** to housing data, it demonstrates robust model validation techniques, including 5-fold cross-validation and error distribution analysis.

---

## 🎯 Project Objective
To implement a standardized evaluation framework that measures predictive reliability through both statistical metrics (Accuracy, MSE, $R^2$) and visual diagnostics (ROC Curves, Residual Plots).

---

## 🔬 Methodology & Key Results

### 1. Classification (Breast Cancer Diagnosis)
* **Model:** Logistic Regression.
* **Outcome:** Achieved near-perfect separability with an **AUC ≈ 1.00**.
* **Confusion Matrix:** High precision with only **2 misclassifications** out of 143 test cases, proving the model's reliability in clinical binary classification.
 

### 2. Regression (California Housing)
* **Model:** Ridge Regression.
* **Outcome:** Captured general market trends but showed increased variance at higher price points.
* **Diagnostic Insight:** The **Predicted vs. Actual** plot revealed underestimation for high-value properties, suggesting a need for non-linear modeling for luxury real estate segments.
 

## 🚀 Technical Implementation
The complete evaluation suite, including the cross-validation logic and performance visualizations, is available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1lw7kFeyQgyBLAJjre5u32Zclo4of3zYz#scrollTo=YY1U4f3xzVrk)**

---

## 🧪 Tech Stack
* **Machine Learning:** Scikit-Learn (Logistic Regression, Ridge Regression)
* **Validation:** 5-Fold Cross-Validation, ROC-AUC, Confusion Matrix
* **Data Science:** Python, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn


# 17. 🛒 Market Basket Analysis: Apriori & Association Rules

This project applies the **Apriori Algorithm** to retail transaction data to uncover hidden purchasing patterns. By calculating **Support, Confidence, and Lift**, the analysis identifies which products are frequently bought together to optimize cross-selling and inventory strategies.

---

## 🎯 Project Objective
To extract actionable consumer insights from transaction histories, enabling data-driven decisions for product bundling, store layout, and targeted marketing.

---

## 📊 Key Findings & Rules
* **Most Frequent Items:** Jackets (54%), Shirts (52%), and Scarves (51%) dominate individual purchases.
* **Primary Anchor Pair:** **(Jacket, Shirt)** is the most frequent combination with **29% support**, representing a fundamental "outfit" building block.


### 📈 Strongest Association Rules
| Rule (Antecedent → Consequent) | Confidence | Lift | Strategic Action |
| :--- | :--- | :--- | :--- |
| **Shoes → Jeans** | 51% | **1.19** | Highest correlation; cross-sell Jeans with footwear. |
| **Shoes → Shirt** | 56% | 1.08 | Recommend Shirts to shoe buyers. |
| **Shirt ↔ Jacket** | 56% | 1.03 | Reciprocal relationship; ideal for "Complete the Look" bundles. |

---

## 💡 Actionable Business Insights
* **Strategic Bundling:** Create "Shoe & Jeans" or "Jacket & Shirt" sets with minor discounts to increase Average Order Value (AOV).
* **Digital Merchandising:** Implement "Customers also bought" triggers on product pages for high-lift items like Shoes and Jackets.
* **Store Optimization:** Place high-support items (Jackets/Shirts) in proximity to encourage natural basket growth.
---

## 🚀 Technical Implementation
The complete Apriori implementation, including the generation of frequent itemsets and the filtered association rules table, is available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/10w9QKpHSnV-FeLJHp-z98saC4Zx02LdG#scrollTo=3a02283c)**

---

## 🧪 Tech Stack
* **Algorithm:** Apriori (Association Rule Mining)
* **Libraries:** MLxtend, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

# 18. 🚨 Anomaly Detection: Isolation Forest vs. Local Outlier Factor (LOF)

This project explores **Unsupervised Learning** techniques to identify outliers within the Breast Cancer dataset. By comparing global (Isolation Forest) and local (LOF) detection methods, the study highlights how consensus-based flagging improves the reliability of anomaly detection in medical data.

---

## 🎯 Project Objective
To isolate highly irregular data points that deviate from the standard cluster of diagnostic samples, which could represent rare clinical cases, data entry errors, or extreme biological variations.

---

## 🛠️ Methodology & Detection Models
* **Isolation Forest:** A tree-based approach that isolates anomalies by randomly selecting features and split values. Anomalies are "easier" to isolate and thus have shorter path lengths.

* **Local Outlier Factor (LOF):** A density-based method that identifies outliers by comparing the local density of a point to the densities of its neighbors.


---

## 📊 Results & Consensus Analysis
The models were applied to a dataset of **569 total samples**:

| Metric | Isolation Forest | Local Outlier Factor (LOF) |
| :--- | :--- | :--- |
| **Anomalies Detected** | 29 | 29 |
| **Top Anomalies** | Lowest Decision Scores | Most Negative LOF Scores |

### 💡 The "Method Overlap" Insight
* **Consensus Count:** **13 samples** were flagged by **both** algorithms.
* **Significance:** Points identified by multiple disparate mathematical approaches are highly likely to be true anomalies, providing a robust "confidence filter" for data auditing.

---

## 🚀 Technical Implementation
The complete implementation, scoring logic, and outlier distribution analysis are available in the Google Colab notebook:

**[🔗 View Google Colab Notebook](https://colab.research.google.com/drive/1rc8VrN1vusP8MG0k_pmMWpWFdyhk9zTn?usp=sharing)**

---

## 🧪 Tech Stack
* **Machine Learning:** Scikit-Learn (IsolationForest, LocalOutlierFactor)
* **Data Science:** Python, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn