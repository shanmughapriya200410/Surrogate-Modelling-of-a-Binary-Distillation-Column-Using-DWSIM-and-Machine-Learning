import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. MANUAL DATA LOADING (No Pandas Parser)
# ==========================================
data = []
# We use 'errors=ignore' to bypass any weird DWSIM characters
try:
    with open('distillation_dataset.csv', 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for line in lines:
            # Split by comma and clean whitespace
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    # Try to convert the first two items to numbers
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                    data.append([x_val, y_val])
                except ValueError:
                    continue # Skip the header or lines with text

    df = pd.DataFrame(data, columns=['Reflux_Ratio', 'xD'])
    
    if df.empty:
        print("Error: No data was loaded. Please check if 'distillation_dataset.csv' is in the folder.")
    else:
        print(f"Success! Loaded {len(df)} rows.")
        print(df.head())

except FileNotFoundError:
    print("Error: The file 'distillation_dataset.csv' was not found in this folder.")

# ==========================================
# 2. MACHINE LEARNING & PLOTTING
# ==========================================
if not df.empty:
    X = df[['Reflux_Ratio']]
    y = df['xD'].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "SVM": SVR(kernel='rbf')
    }

    plt.figure(figsize=(15, 5))
    results = {}

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        results[name] = {'R2': r2_score(y_test, preds), 'MAE': mean_absolute_error(y_test, preds)}

        plt.subplot(1, 3, i+1)
        plt.scatter(y_test, preds, alpha=0.6, color='darkcyan')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(name)
        plt.xlabel('Actual xD')
        plt.ylabel('Predicted xD')

    print("\n--- RESULTS ---")
    print(pd.DataFrame(results).T)
    plt.tight_layout()
    plt.show()