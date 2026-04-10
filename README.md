# Surrogate-Modelling-of-a-Binary-Distillation-Column-Using-DWSIM-and-Machine-Learning


## 1. Project Overview
This project involves the development of a Machine Learning (ML) surrogate model to approximate a rigorous chemical process simulation. A binary distillation column separating Benzene and Toluene was modeled in **DWSIM**, and the resulting data was used to train various ML regressors to predict distillate purity ($x_D$) based on the Reflux Ratio.

### Key Objectives:
* Generate high-fidelity steady-state data using DWSIM Sensitivity Analysis.
* Preprocess and scale simulation data for ML training.
* Compare Linear Regression, Random Forest, and SVM models.
* Identify the most robust surrogate for real-time process monitoring.

---

## 2. File Structure
* `dwsim.dwxmz`: The converged DWSIM flowsheet file.
* `distillation_dataset.csv`: The dataset containing 100 simulation points (Reflux Ratio vs. Purity).
* `surrogate_model_distillation.py`: The Python script for data processing, model training, and visualization.
* `README.md`: This instruction file.
* `Report.pdf`: Comprehensive technical report.
* `Results_Summary.pdf`: Concise summary of final metrics and observations.

---

## 3. Requirements
To run the Python script, ensure you have the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
**Software used:**
* **DWSIM:** Version 8.x or higher (for flowsheet viewing).
* **Python:** Version 3.8 or higher.

---

## 4. How to Run the Files

### Step 1: DWSIM Simulation (Optional)
If you wish to view or modify the simulation:
1. Open **DWSIM**.
2. Load `dwsim.dwxmz`.
3. Go to **Tools > Sensitivity Analysis** to see the parametric sweep setup used to generate the data.

### Step 2: Running the Surrogate Model
1. Place `distillation_dataset.csv` and `surrogate_model_distillation.py` in the same directory.
2. Open your terminal or IDE (VS Code, PyCharm, etc.).
3. Run the script:
   ```bash
   python surrogate_model_distillation.py
   ```

---

## 5. Implementation Details
* **Simulation Range:** Reflux Ratio was varied from **1.5 to 4.0**.
* **Pre-processing:** The script handles `latin1` encoding and implements `StandardScaler` for feature normalization.
* **Evaluation Metrics:** Models are evaluated based on **$R^2$ Score** and **Mean Absolute Error (MAE)**.
* **Validation:** An 80/20 train-test split was utilized to verify the model's predictive capabilities on unseen data.

---

## 6. Results
* **Best Model:** Random Forest Regressor.
* **Accuracy ($R^2$):** ~0.983.
* **Observations:** The Random Forest model effectively captures the non-linear "S-curve" of the distillation process, whereas Linear Regression fails to account for the asymptotic behavior at high purity levels.

---

## 7. Author
**[SHANMUGHA PRIYA]**   
FOSSEE Fellowship Screening Task

