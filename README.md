# Car MPG Prediction App 🚗

A Python-based machine learning project to **predict the fuel efficiency (MPG) of cars** using historical Auto MPG data. The project includes **training multiple regression models**, selecting the best model, and creating a **Streamlit web app** for interactive predictions.

---

## **Features**

* Predicts **numeric MPG** based on car features:

  * Horsepower
  * Weight
  * Acceleration
  * Model Year
  * Origin (1=US, 2=Europe, 3=Japan)
* Trains multiple regression models:

  * Linear Regression, Ridge, Lasso
  * K-Nearest Neighbors (KNN)
  * Support Vector Regression (SVR)
  * Decision Tree, Random Forest
  * Gradient Boosting, XGBoost
* Automatically selects **best-performing model** based on **R² score**.
* Web app using **Streamlit** with a **Predict button**.
* Applies **log transformation** and **scaling** to features for consistency with training.

---

## **Installation**

1. Clone this repository:

```bash
git clone https://github.com/yourusername/car-mpg-prediction.git
cd car-mpg-prediction
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Requirements**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `streamlit`, `matplotlib`

---

## **Usage**

### 1. Train the models

```bash
python train_model.py
```

* Trains all regression models.
* Saves the **best model** as `best_model.pkl` and the **scaler** as `scaler.pkl`.

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

* Enter car features in the app.
* Click **Predict MPG**.
* The predicted fuel efficiency will be displayed.

---

## **Example Input**

| Feature      | Example Value |
| ------------ | ------------- |
| Horsepower   | 95            |
| Weight (lbs) | 2372          |
| Acceleration | 15            |
| Model Year   | 70            |
| Origin       | 2             |

**Predicted MPG:** 24.00 (approx.)

---

## **Project Structure**

```
car-mpg-prediction/
│
├─ auto-mpg.csv           # Original dataset
├─ train_model.py         # Script to train models and save best model
├─ app.py                 # Streamlit web app
├─ best_model.pkl         # Saved best model after training
├─ scaler.pkl             # Saved feature scaler
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation
```

---

## **Notes**

* Ensure that the **log transformation** is applied consistently for any new inputs in the app.
* The app predicts **numeric MPG**, not categories.
* Use R², RMSE, and MSE as primary metrics for regression model performance.

---
