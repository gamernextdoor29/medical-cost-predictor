🏥 Medical Cost Predictor

A machine learning project that predicts individual medical insurance costs based on demographic and health-related features. This project applies data preprocessing, feature engineering, and regression modeling to generate accurate cost estimates.

📌 Project Overview

Healthcare costs can vary significantly depending on factors like age, BMI, smoking habits, and more. This project builds a predictive model to estimate medical charges using these variables.

Key highlights:

Cleaned and validated real-world dataset
Performed exploratory data analysis (EDA)
Engineered meaningful health-related features
Applied transformation techniques to improve model performance
Built a regression pipeline with polynomial features
📂 Project Structure
├── data/
│   └── insurance.csv          # Dataset
├── src/
│   └── codes.py               # Main modeling code
├── outputs/
│   └── (visualizations)       # Generated plots
├── README.md
🧹 Data Preprocessing
✅ Missing Values

Checked for null values:

df.isnull().any().sum()
Result: No missing values found
✅ Duplicate Handling
Detected 1 duplicate row

Verified and removed:

df = df.drop_duplicates()
📊 Exploratory Data Analysis (EDA)
🔍 Target Variable Distribution

Checked skewness of charges:

df['charges'].skew()
Result: 1.515 (positively skewed)
Action: Applied log transformation to normalize distribution
📈 Visual Analysis

Generated key plots:

BMI vs Charges (colored by smoker status)
Age vs Charges (trend line)
Charges distribution across:
Regions
Number of children
🧠 Feature Engineering

Created domain-informed features to improve prediction:

obese_smoker
healthy_non_smoker
healthy_smoker
obese_non_smoker

These features capture combined health risk profiles for better modeling.

⚙️ Model Pipeline

Built using Scikit-learn Pipeline:

Pipeline([
    ('preprocessing', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])
Steps:
Feature scaling
Polynomial feature expansion (degree=2)
Linear regression model
🔄 Training Process
Train/Test Split: 80/20
Applied log transformation to target (log1p)
Predictions converted back using expm1
📏 Model Evaluation

Metrics used:

R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

Also evaluated:

Train vs Test MAE (to detect overfitting/underfitting)
Error percentage relative to average charges
📉 Visualizations
🔹 Actual vs Predicted
Scatter plot comparing predictions with true values
Includes reference diagonal line
🔹 Residual Distribution
Histogram of prediction errors
Helps assess model bias and variance


🚀 How to Run

Clone the repository:

git clone https://github.com/gamernextdoor29/medical-cost-predictor.git
cd medical-cost-predictor

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn scipy

Run the script:

python src/codes.py
🛠️ Tech Stack
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
SciPy
💡 Key Insights
Smoking status significantly increases medical costs
BMI combined with smoking creates high-risk groups
Log transformation greatly improves model performance
Polynomial features help capture non-linear relationships
📌 Future Improvements
Try advanced models (e.g., Random Forest, Gradient Boosting)
Hyperparameter tuning
Deploy as a web app (Streamlit or Flask)
Add cross-validation for more robust evaluation

👤 Author
Mujeeb Bello

Mujeeb Bello
