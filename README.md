# 🏠 House Price Predictor — Machine Learning

A beginner-friendly Python ML project that trains and compares 3 regression models to predict house prices using features like square footage, bedrooms, neighborhood, and condition.

---

## 📸 Dashboard Preview

![ML Dashboard](house_price_dashboard.png)

---

## 🔍 What This Project Does

- Loads and cleans a structured housing dataset (300 homes, 12 features)
- Encodes categorical features (neighborhood, condition, school rating)
- Splits data into **80% train / 20% test**
- Trains and compares **3 ML models**:
  - Linear Regression
  - Random Forest
  - Gradient Boosting *(best performer)*
- Evaluates models using MAE, RMSE, and R² score
- Visualizes feature importance, residuals, and actual vs predicted prices
- Makes a **custom house price prediction**

---

## 📊 Model Results

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | ~$57,800 | 0.72 |
| Random Forest | ~$48,400 | 0.77 |
| **Gradient Boosting** | **~$27,700** | **0.93** ✅ |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature encoding |
| `scikit-learn` | Train/test split, model training, evaluation |
| `matplotlib` | Chart layout and rendering |
| `seaborn` | Statistical visualizations |
| `numpy` | RMSE calculation |

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
```

**2. Install dependencies**
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

**3. Run the script**
```bash
python house_price_predictor.py
```

The dashboard will display on screen and save as `house_price_dashboard.png`.

---

## 📁 Project Structure

```
house-price-predictor/
│
├── house_price_predictor.py    # Full ML pipeline script
├── house_prices.csv            # Dataset (300 rows, 12 features)
├── house_price_dashboard.png   # Output dashboard image
└── README.md                   # Project documentation
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `neighborhood` | Area of the city (Downtown, Lakefront, etc.) |
| `bedrooms` | Number of bedrooms (1–6) |
| `bathrooms` | Number of bathrooms (1–4) |
| `sqft` | Square footage (500–5000) |
| `age_years` | Age of the house in years |
| `garage_spaces` | Number of garage spaces (0–2) |
| `has_pool` | Pool present (0 or 1) |
| `floors` | Number of floors (1–3) |
| `condition` | House condition (Excellent / Good / Fair / Poor) |
| `crime_rate` | Neighborhood crime rate (Low / Medium / High) |
| `school_rating` | Local school rating (A / B / C) |
| `price` | 🎯 Target — sale price in USD |

---

## 💡 Key Insights

- **Gradient Boosting** significantly outperforms Linear Regression (R² 0.93 vs 0.72)
- **House condition** and **school rating** are the most important features
- **Square footage** matters less than expected — location quality dominates
- Residuals are roughly centered at $0, meaning the model isn't systematically over/under-predicting

---

## 🧠 What I Learned

- How to encode categorical features with `LabelEncoder`
- How to split data into train/test sets with `train_test_split`
- How to train and compare multiple regression models
- How to interpret MAE, RMSE, and R² evaluation metrics
- How to read a feature importance chart and residual plot

---

*This is Project 2 of 5 in my Python Data/ML learning roadmap.*
