# MLOps Assignment Report
# GROUP 28 
APARAJEETA BEHERA :  2024aa05254@wilp.bits-pilani.ac.in : 100% 

AYAN DEB . : 2024aa05430@wilp.bits-pilani.ac.in : 100% 

HARISH N . : 2024aa05528@wilp.bits-pilani.ac.in : 100 % 

K. P. SHINE . : 2024aa05572@wilp.bits-pilani.ac.in : 100%

SUNIL KUMAR PANDEY . : 2024aa05222@wilp.bits-pilani.ac.in : 100% 


## Heart Disease Prediction Pipeline

### 1. Data Acquisition & EDA ✓
- Downloaded UCI Heart Disease dataset (303 samples, 14 features)
- Performed data cleaning (handled missing values)
- Created visualizations:
  - Class distribution histogram
  - Age distribution histogram  
  - Correlation heatmap
  - Cholesterol vs Age scatter plot

### 2. Feature Engineering & Model Development ✓
- Feature scaling using StandardScaler
- Trained 2 models:
  - Logistic Regression: Accuracy 0.869, ROC-AUC 0.951
  - Random Forest: Accuracy 0.885, ROC-AUC 0.951
- Selected best model: Logistic Regression (based on simplicity and similar performance)

### 3. Experiment Tracking with MLflow ✓
- Logged 6 experiments with different hyperparameters
- Tracked parameters, metrics, and artifacts
- MLflow UI accessible at http://localhost:5000

### 4. Model Packaging & Reproducibility ✓
- Created reusable model package class
- Saved preprocessing pipeline
- Created frozen requirements.txt

### 5. CI/CD Pipeline & Automated Testing ✓
- Created unit tests for data and model
- Setup GitHub Actions workflow
- Tests cover: data quality, model loading, predictions

### 6. Model Containerization ✓ ATTEMPTED
- Created Dockerfile
- **Note:** Docker build attempted but failed due to virtualization not enabled on system
- API tested successfully locally on port 5000

### 7. Production Deployment ✓ (Local)
- API deployed locally on port 5000
- Endpoints tested and working:
  - `GET /` - Returns API status
  - `POST /predict` - Makes predictions with confidence scores

### 8. Monitoring & Logging ✓
- Implemented request logging
- Created simple metrics tracking
- Logs saved to `logs/api.log`

### 9. Repository Structure ✓
bits_mlops_healthdisease/
├── data/ # Dataset files
│── api/ # Flask API
├── notebooks/ # EDA notebooks
├── src/ # Source code
│ ├── data_preprocessing/ # Data cleaning
│ ├── model_training/ # Model training
│ ├── experiment_tracking/ # MLflow
│ └── monitoring/ # Monitoring
├── tests/ # Unit tests
├── models/ # Trained models
├── reports/ # Reports & plots
├── .github/workflows/ # CI/CD pipeline
├── Dockerfile # Containerization
├── docker-compose.yml # Local deployment
├── requirements.txt # Dependencies
└── README.md # Documentation


### API Testing Results
Health Check: {"status": "healthy", "model_loaded": true}
Prediction Test: {"prediction": 0, "probability": 0.150, "confidence": "high", "diagnosis": "No heart disease detected"}


### Screenshots
- EDA Visualizations: `reports/eda_visualizations.png`
- Model Comparison: `reports/model_comparison.png`
- MLflow UI: See attached screenshots
- <img width="936" height="377" alt="image" src="https://github.com/user-attachments/assets/49d75581-fa01-4969-b9a9-c5acecf9d392" />

- API Response: See attached screenshots

### Challenges & Solutions
1. **Docker Virtualization Issue**: Documented as attempted but requires BIOS settings change
2. **Path Issues in API**: Fixed using absolute paths
3. **Python 3.13 Compatibility**: Used latest package versions

### Conclusion
Successfully built end-to-end MLOps pipeline for heart disease prediction. All requirements met except Docker deployment which requires system-level virtualization enablement.
