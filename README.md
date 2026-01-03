# VertexML - Automated Machine Learning Platform

VertexML is a comprehensive, end-to-end automated machine learning platform designed to streamline the journey from raw data to deployable models. It features a premium Streamlit interface and integrates advanced preprocessing, model selection, and monitoring capabilities.

## 🚀 Key Features

- **Intelligent Preprocessing**: Automated handling of missing values, encoding, scaling, and feature selection.
- **Auto-Detection**: Automatic detection of task types (Classification vs. Regression).
- **Multi-Algorithm Support**: Train and compare multiple models including XGBoost, LightGBM, CatBoost, Random Forest, and more.
- **Advanced Training Modes**: Supports Auto-Ensemble and Co-Training strategies.
- **Performance Evaluation**: Comprehensive metrics with interactive Plotly visualizations.
- **Model Export**: Seamless export to Pickle, Joblib, or ONNX formats.
- **Enterprise Reporting**: Generate detailed PDF reports of dataset analysis and model performance.
- **AI-Powered Insights**: Integrated Gemini/GPT-based dataset analysis and algorithmic recommendations.
- **Monitoring & Retraining**: Built-in data drift detection and automated retraining pipelines.

## 🛠 Project Structure

- `app/`: Primary application logic and Streamlit interface.
- `data/`: Local storage for training datasets (ignored by Git).
- `models/`: Storage for trained model binaries (ignored by Git).
- `reports/`: Generated PDF and JSON analysis reports (ignored by Git).
- `retraining/`: Scripts and logic for model update pipelines.

## 📂 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Clone and Install
```bash
# Clone the repository
git clone <repository-url>
cd AutoML

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the `app/` directory or set environment variables:
```env
OPENAI_API_KEY=your_key_here
# OR
GOOGLE_API_KEY=your_key_here
```

## 🏃 Running the Application

Launch the platform using Streamlit:

```bash
streamlit run app/main.py
```

---
*Developed for efficient and accessible Machine Learning development.*