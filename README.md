# 💳 CreditIQ: Credit Risk Modelling

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Credit Risk Modelling application that predicts loan default probability, generates credit scores, and assigns ratings.**

[Demo](your-demo-link) • [Report Bug](your-issues-link) • [Request Feature](your-issues-link)

</div>

---

## 🎯 Overview

CreditIQ is a comprehensive credit risk assessment system that leverages machine learning to evaluate borrower creditworthiness. The application processes financial data, predicts default probability, and translates it into an intuitive credit score (300-900) with corresponding ratings.

**Key Highlights:**
- Advanced ML pipeline with hyperparameter optimization
- Handles imbalanced datasets using SMOTE + undersampling
- Interactive web interface built with Streamlit
- Production-ready model deployment

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- **Logistic Regression** model with robust preprocessing
- **Class imbalance handling** using SMOTE + undersampling techniques
- **Hyperparameter optimization** with RandomizedSearchCV and Optuna
- **Cross-validation** for reliable model performance

### 📊 Comprehensive Model Evaluation
- **Classification Metrics**: Precision, Recall, F1-score
- **Performance Indicators**: ROC-AUC (~0.98) and Gini Coefficient (~0.96)
- **Statistical Tests**: KS Statistics (~0.85) & Decile analysis
- **Visualization**: ROC curves and performance plots

### 💯 Credit Scoring System
- **Score Range**: 300 (Poor) → 900 (Excellent)
- **Risk Categories**: Poor, Average, Good, Excellent
- **Probability Mapping**: Seamless conversion from default probability to credit score

### 🖥️ Interactive Interface
- **Streamlit Web App** with intuitive user experience
- **Real-time Predictions** with instant results
- **Input Validation** and error handling
- **Responsive Design** for various screen sizes

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11** - Primary programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library

### Machine Learning & Data Processing
```
scikit-learn     # Model training, selection, and metrics
xgboost         # Advanced boosting algorithms  
optuna          # Bayesian hyperparameter optimization
imblearn        # Handling imbalanced datasets (SMOTE)
pandas          # Data manipulation and analysis
numpy           # Numerical computing
joblib          # Model serialization and persistence
```

### Visualization & UI
```
streamlit       # Interactive web application
plotly          # Interactive visualizations
matplotlib      # Static plotting
seaborn         # Statistical data visualization
```

---

## 📁 Project Structure

```
credit_risk_modelling/
│
├── 📁 artifacts/                 # Saved models and preprocessors
│   ├── model.joblib             # Trained ML model
│   └── scaler.joblib            # Feature scaler
│
├── 📁 data/                     # Dataset files (if included)
│   └── credit_data.csv          # Training dataset
│
├── 📁 notebooks/                # Jupyter notebooks
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   ├── model_training.ipynb     # Model development
│   └── evaluation.ipynb        # Model evaluation
│
├── 📁 src/                      # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── model_training.py        # ML model training pipeline
│   └── evaluation.py           # Model evaluation utilities
│
├── 📄 main.py                   # Streamlit application entry point
├── 📄 prediction_helper.py      # Prediction and scoring functions
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 .gitignore               # Git ignore rules
└── 📄 LICENSE                   # Project license
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Git (for cloning)
- Virtual environment tool (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AchiFerns/credit_risk_modelling.git
   cd credit_risk_modelling
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate          # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

---

## 💻 Usage Guide

### Web Interface Features

1. **Input Borrower Information**
   - Personal details (Age, Income)
   - Loan specifics (Amount, Tenure, Purpose)
   - Credit history (Open Accounts, Utilization)
   - Background info (Employment, Residence)

2. **Get Instant Results**
   - **Default Probability**: Percentage chance of loan default
   - **Credit Score**: Numerical score (300-900 range)
   - **Credit Rating**: Categorical assessment (Poor/Average/Good/Excellent)
   - **Risk Assessment**: Detailed risk breakdown

### API Usage (If Available)

```python
import joblib
from prediction_helper import predict_credit_risk

# Load the trained model
model = joblib.load('artifacts/model.joblib')

# Sample prediction
borrower_data = {
    'age': 35,
    'income': 50000,
    'loan_amount': 200000,
    # ... other features
}

result = predict_credit_risk(borrower_data)
print(f"Credit Score: {result['score']}")
print(f"Rating: {result['rating']}")
```

---

## 📈 Model Performance

### Key Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| **ROC-AUC** | ~0.98 | Excellent discrimination ability |
| **Gini Coefficient** | ~0.96 | Strong predictive power |
| **KS Statistic** | ~0.85 | High separation between classes |
| **Precision** | ~0.94 | Low false positive rate |
| **Recall** | ~0.92 | High true positive detection |

### Model Validation
- **Cross-validation**: 5-fold stratified CV
- **Test Set Performance**: Consistent with validation metrics
- **Overfitting Check**: Learning curves show good generalization

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run main.py
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Deploy with one click

#### Heroku
```bash
# Create Procfile
echo "web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
```

---

## 🔮 Future Enhancements

### Machine Learning Improvements
- [ ] **Advanced Models**: Integration of CatBoost, LightGBM, and XGBoost
- [ ] **Ensemble Methods**: Stacking and blending multiple models
- [ ] **Feature Engineering**: Automated feature creation and selection
- [ ] **Deep Learning**: Neural network architectures for complex patterns

### Model Interpretability
- [ ] **SHAP Integration**: Feature importance and prediction explanations
- [ ] **LIME Support**: Local interpretable model-agnostic explanations
- [ ] **Feature Contribution**: Individual feature impact visualization

### Technical Enhancements
- [ ] **FastAPI Backend**: RESTful API for model serving
- [ ] **Database Integration**: PostgreSQL/MongoDB for data storage
- [ ] **Real-time Processing**: Kafka/Redis for streaming predictions
- [ ] **MLOps Pipeline**: MLflow for experiment tracking and model registry

### User Experience
- [ ] **Advanced Dashboard**: Interactive charts and analytics
- [ ] **Batch Processing**: Upload CSV files for bulk predictions
- [ ] **Historical Tracking**: User prediction history and trends
- [ ] **Mobile Optimization**: Responsive design improvements

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

### Submitting Changes
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Credit risk data from [source]
- **Inspiration**: Industry best practices in credit scoring
- **Community**: Open source libraries and contributors

---

## 📞 Contact & Support

- **Author**: [Your Name](https://github.com/AchiFerns)
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/AchiFerns/credit_risk_modelling](https://github.com/AchiFerns/credit_risk_modelling)

### Getting Help
- 🐛 **Bug Reports**: [Create an issue](your-issues-link)
- 💡 **Feature Requests**: [Request a feature](your-issues-link)
- 💬 **Questions**: [Start a discussion](your-discussions-link)

---

<div align="center">

**Made with ❤️ for better credit risk assessment**

⭐ Star this repository if you found it helpful!

</div>
