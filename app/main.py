import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_RANDOM_STATE, MAX_TRIALS
from preprocessing.preprocessor import EnhancedUniversalDataPreprocessor, get_processed_data, load_large_file

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('VertexML')
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    IsolationForest
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error,
    r2_score, explained_variance_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    roc_curve, auc, roc_auc_score, confusion_matrix,
    mean_absolute_error
)
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.decomposition import PCA, NMF, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_regression, RFE
from sklearn.semi_supervised import (
    SelfTrainingClassifier, LabelPropagation, LabelSpreading
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep, time, perf_counter
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from umap import UMAP
import warnings
import pickle
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import io
from fpdf import FPDF
import base64
from io import BytesIO
import tempfile
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import datetime
from scipy import stats
from scipy.stats import randint, uniform

MODEL_REGISTRY = {
    "classification": {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "SVM": SVC,
    },
    "regression": {
        "Linear Regression": LinearRegression,
        "Random Forest": RandomForestRegressor,
        "XGBoost": XGBRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "Ridge Regression": Ridge,
    },
    "clustering": {
        "K-Means": KMeans,
        "DBSCAN": DBSCAN,
        "Hierarchical": AgglomerativeClustering,
        "Gaussian Mixture": GaussianMixture,
    },
    "anomaly_detection": {
        "Isolation Forest": IsolationForest,
        "Local Outlier Factor": LocalOutlierFactor,
    },
}

MODEL_DEFAULT_PARAMS = {
    "classification": {
        "Logistic Regression": {"max_iter": 2000, "random_state": DEFAULT_RANDOM_STATE},
        "Random Forest": {"random_state": DEFAULT_RANDOM_STATE},
        "Gradient Boosting": {"random_state": DEFAULT_RANDOM_STATE},
        "XGBoost": {"eval_metric": "logloss", "random_state": DEFAULT_RANDOM_STATE, "verbosity": 0},
        "LightGBM": {"random_state": DEFAULT_RANDOM_STATE},
        "SVM": {"probability": True, "random_state": DEFAULT_RANDOM_STATE},
    },
    "regression": {
        "Linear Regression": {},
        "Random Forest": {"random_state": DEFAULT_RANDOM_STATE},
        "XGBoost": {"random_state": DEFAULT_RANDOM_STATE},
        "Gradient Boosting": {"random_state": DEFAULT_RANDOM_STATE},
        "Ridge Regression": {},
    },
    "clustering": {
        "K-Means": {"random_state": 42, "n_init": 10},
        "DBSCAN": {},
        "Hierarchical": {},
        "Gaussian Mixture": {"random_state": 42},
    },
    "anomaly_detection": {
        "Isolation Forest": {"random_state": 42},
        "Local Outlier Factor": {},
    },
}

# Check for required packages
try:
    import kaleido
except ImportError:
    st.warning("Kaleido package is required for PDF export. Install with: pip install -U kaleido")

warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="VertexML",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Premium UI Styling
def load_css():
    css_path = Path("assets/styles.css")
    if css_path.exists():
        with css_path.open("r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Unable to load CSS from assets/styles.css")


def apply_custom_styles(theme_mode="light"):
    if theme_mode == "dark":
        theme_vars = """
        :root {
            --app-bg: #0b1220;
            --app-bg-gradient: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            --surface: rgba(30, 41, 59, 0.72);
            --surface-strong: rgba(30, 41, 59, 0.9);
            --surface-border: rgba(148, 163, 184, 0.25);
            --accent-primary: #0ea5e9;
            --accent-secondary: #6366f1;
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --card-shadow: 0 12px 26px rgba(2, 6, 23, 0.35);
        }
        """
    else:
        theme_vars = """
        :root {
            --app-bg: #f8fafc;
            --app-bg-gradient: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            --surface: rgba(255, 255, 255, 0.9);
            --surface-strong: rgba(255, 255, 255, 0.98);
            --surface-border: rgba(15, 23, 42, 0.12);
            --accent-primary: #0284c7;
            --accent-secondary: #2563eb;
            --text-main: #0f172a;
            --text-muted: #475569;
            --card-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        }
        """

    load_css()
    st.markdown(f"<style>{theme_vars}</style>", unsafe_allow_html=True)

# ==================== FIXED: CoTrainer Class ====================
class CoTrainer(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator1=None, estimator2=None, max_iter=100, random_state=None):
        self.estimator1 = estimator1 or LogisticRegression(max_iter=1000)
        self.estimator2 = estimator2 or RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)
        self.max_iter = max_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        check_classification_targets(y)
        
        # Split features into two views
        n_features = X.shape[1]
        split = n_features // 2
        view1 = X[:, :split]
        view2 = X[:, split:]
        
        # Initialize estimators
        self.estimator1_ = clone(self.estimator1)
        self.estimator2_ = clone(self.estimator2)
        
        # Get labeled data
        labeled = y != -1
        X1_labeled = view1[labeled]
        X2_labeled = view2[labeled]
        y_labeled = y[labeled]
        
        # Train initial models
        self.estimator1_.fit(X1_labeled, y_labeled)
        self.estimator2_.fit(X2_labeled, y_labeled)
        
        # Get unlabeled data
        X1_unlabeled = view1[~labeled]
        X2_unlabeled = view2[~labeled]

        if X1_unlabeled.shape[0] == 0:
            return self

        # Co-training loop
        for _ in range(self.max_iter):
            # Predict probabilities on unlabeled data
            prob1 = self.estimator1_.predict_proba(X1_unlabeled)
            prob2 = self.estimator2_.predict_proba(X2_unlabeled)
            
            # Get most confident predictions from each view
            conf1 = np.max(prob1, axis=1)
            conf2 = np.max(prob2, axis=1)
            
            # Select most confident predictions
            n_to_add = min(100, len(conf1) // 10)
            
            if n_to_add == 0:
                break
                
            idx1 = np.argsort(conf1)[-n_to_add:]
            idx2 = np.argsort(conf2)[-n_to_add:]
            
            # Add pseudo-labeled data
            X1_labeled = np.vstack([X1_labeled, X1_unlabeled[idx1]])
            X2_labeled = np.vstack([X2_labeled, X2_unlabeled[idx2]])
            y_labeled = np.concatenate([
                y_labeled,
                np.argmax(prob1[idx1], axis=1),
                np.argmax(prob2[idx2], axis=1)
            ])
            
            # Remove added samples from unlabeled
            mask = np.ones(len(X1_unlabeled), dtype=bool)
            mask[idx1] = False
            mask[idx2] = False
            X1_unlabeled = X1_unlabeled[mask]
            X2_unlabeled = X2_unlabeled[mask]
            
            # Retrain models
            self.estimator1_.fit(X1_labeled, y_labeled)
            self.estimator2_.fit(X2_labeled, y_labeled)
            
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        
        # Split features into two views
        n_features = X.shape[1]
        split = n_features // 2
        view1 = X[:, :split]
        view2 = X[:, split:]
        
        # Predict from both views and average probabilities
        prob1 = self.estimator1_.predict_proba(view1)
        prob2 = self.estimator2_.predict_proba(view2)
        avg_prob = (prob1 + prob2) / 2
        
        return np.argmax(avg_prob, axis=1)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        
        # Split features into two views
        n_features = X.shape[1]
        split = n_features // 2
        view1 = X[:, :split]
        view2 = X[:, split:]
        
        # Predict from both views and average probabilities
        prob1 = self.estimator1_.predict_proba(view1)
        prob2 = self.estimator2_.predict_proba(view2)
        avg_prob = (prob1 + prob2) / 2
        
        return avg_prob

# ==================== FIXED: Performance Calibrator ====================
class PerformanceCalibrator:
    """Calibrates GPT estimates based on actual performance"""
    
    def __init__(self):
        self.history = []
        self.calibration_factors = {
            'classification': 0.85,
            'regression': 0.90,
            'clustering': 0.80,
            'dimensionality_reduction': 0.85,
            'anomaly_detection': 0.75
        }
    
    def calibrate_estimate(self, gpt_estimate: Union[str, float], task_type: str, 
                          actual_performance: float = None) -> float:
        """Calibrate GPT estimate based on history"""
        # Extract numeric value from estimate string
        if isinstance(gpt_estimate, str):
            numeric_est = self._extract_numeric_estimate(gpt_estimate)
        else:
            numeric_est = gpt_estimate
        
        # Apply calibration factor
        calibrated = numeric_est * self.calibration_factors.get(task_type, 0.85)
        
        # Update calibration if actual performance is provided
        if actual_performance is not None:
            self._update_calibration(task_type, numeric_est, actual_performance)
            # Return actual if it's significantly different
            if abs(calibrated - actual_performance) > 0.15:
                calibrated = actual_performance
        
        return calibrated
    
    def _extract_numeric_estimate(self, estimate_str: str) -> float:
        """Extract numeric value from estimate string"""
        try:
            # Handle ranges like "85-90%"
            if '-' in estimate_str:
                parts = estimate_str.replace('%', '').split('-')
                if '%' in estimate_str:
                    return (float(parts[0]) + float(parts[1])) / 200.0
                else:
                    return (float(parts[0]) + float(parts[1])) / 2.0
            # Handle single values
            elif '%' in estimate_str:
                return float(estimate_str.replace('%', '')) / 100.0
            else:
                return float(estimate_str)
        except:
            return 0.5  # Default
    
    def _update_calibration(self, task_type: str, estimated: float, actual: float):
        """Update calibration factor based on actual performance"""
        if estimated > 0:  # Avoid division by zero
            ratio = actual / estimated
            self.calibration_factors[task_type] = \
                0.7 * self.calibration_factors[task_type] + 0.3 * min(max(ratio, 0.5), 1.5)
            
            # Keep history for analysis
            self.history.append({
                'task_type': task_type,
                'estimated': estimated,
                'actual': actual,
                'ratio': ratio,
                'timestamp': datetime.datetime.now()
            })

# ==================== FIXED: Enhanced AutoML Model ====================
class EnhancedAutoMLModel:
    def __init__(self, task_type, model_choice, n_clusters=None, encoding_dim=None, 
                 handle_imbalance=False, enable_tuning=True, tuning_trials=MAX_TRIALS):
        self.task_type = task_type
        self.model_choice = model_choice
        self.n_clusters = n_clusters
        self.encoding_dim = encoding_dim
        self.handle_imbalance = handle_imbalance
        self.enable_tuning = enable_tuning
        self.tuning_trials = max(1, int(tuning_trials))
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
    def _get_hyperparameter_grid(self):
        """Define hyperparameter grids for different models"""
        if self.task_type == 'classification':
            if self.model_choice == "Random Forest":
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            elif self.model_choice == "XGBoost":
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                }
            elif self.model_choice == "Logistic Regression":
                return {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            elif self.model_choice == "SVM":
                return {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
        
        elif self.task_type == 'regression':
            if self.model_choice == "Random Forest":
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            elif self.model_choice == "XGBoost":
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
        
        elif self.task_type == 'clustering':
            if self.model_choice == "K-Means":
                return {
                    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'init': ['k-means++', 'random'],
                    'n_init': [10, 20, 30]
                }
        
        return {}
    
    def _init_base_model(self):
        """Initialize base model without tuning"""
        model_cls = MODEL_REGISTRY.get(self.task_type, {}).get(self.model_choice)
        if model_cls is None:
            return None

        init_kwargs = MODEL_DEFAULT_PARAMS.get(self.task_type, {}).get(self.model_choice, {}).copy()

        if self.task_type == 'classification' and self.handle_imbalance:
            if self.model_choice in {"Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"}:
                init_kwargs['class_weight'] = 'balanced'

        if self.task_type == 'clustering':
            if self.model_choice == "K-Means":
                init_kwargs['n_clusters'] = self.n_clusters
            elif self.model_choice == "Hierarchical":
                init_kwargs['n_clusters'] = self.n_clusters
            elif self.model_choice == "Gaussian Mixture":
                init_kwargs['n_components'] = self.n_clusters

        if self.task_type == 'anomaly_detection':
            if self.model_choice == "Local Outlier Factor":
                init_kwargs['novelty'] = True

        return model_cls(**init_kwargs)
    
    def fit_with_tuning(self, X, y, cv_folds=5):
        """Fit model with hyperparameter tuning and cross-validation"""
        logger.info(
            "Starting model fit for task_type=%s, model_choice=%s, enable_tuning=%s, tuning_trials=%s",
            self.task_type,
            self.model_choice,
            self.enable_tuning,
            self.tuning_trials
        )
        start_time = perf_counter()

        def _get_cv_splitter(task_type, X_data, y_data, desired_folds):
            if task_type != 'classification':
                n_splits = min(max(2, desired_folds), X_data.shape[0])
                return KFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

            y_series = pd.Series(y_data)
            class_counts = y_series.value_counts()
            min_class_count = int(class_counts.min()) if not class_counts.empty else 0
            n_classes = int(class_counts.shape[0])

            if n_classes >= 2 and min_class_count >= 2:
                n_splits = min(max(2, desired_folds), X_data.shape[0], min_class_count)
                return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)

            # Fallback to regular KFold when stratification is not possible.
            n_splits = min(max(2, desired_folds), X_data.shape[0])
            return KFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
        
        if not self.enable_tuning or X.shape[0] < 100:
            # Use base model for small datasets
            self.model = self._init_base_model()
            
            if self.task_type in ['classification', 'regression', 'anomaly_detection']:
                self.model.fit(X, y)
            elif self.task_type in ['clustering', 'dimensionality_reduction']:
                self.model.fit(X)
            
            # Compute cross-validation scores for supervised tasks
            if self.task_type in ['classification', 'regression']:
                cv = _get_cv_splitter(self.task_type, X, y, desired_folds=3)
                
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                self.cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            # Perform hyperparameter tuning
            base_model = self._init_base_model()
            param_grid = self._get_hyperparameter_grid()
            
            if param_grid and self.task_type in ['classification', 'regression']:
                effective_folds = cv_folds
                cv = _get_cv_splitter(self.task_type, X, y, desired_folds=effective_folds)
                
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                
                def _optuna_objective(trial):
                    trial_params = {}
                    for param_name, param_values in param_grid.items():
                        if isinstance(param_values, list):
                            trial_params[param_name] = trial.suggest_categorical(param_name, param_values)
                        elif isinstance(param_values, tuple) and len(param_values) == 2:
                            low, high = param_values
                            if isinstance(low, int) and isinstance(high, int):
                                trial_params[param_name] = trial.suggest_int(param_name, low, high)
                            else:
                                trial_params[param_name] = trial.suggest_float(param_name, low, high)
                        else:
                            raise ValueError(f"Unsupported hyperparameter format for {param_name}: {param_values}")

                    trial_model = clone(base_model).set_params(**trial_params)
                    scores = cross_val_score(trial_model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                    return float(np.mean(scores))

                study = optuna.create_study(
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE)
                )
                study.optimize(_optuna_objective, n_trials=self.tuning_trials)

                self.best_params = study.best_params
                self.model = clone(base_model).set_params(**self.best_params)
                self.model.fit(X, y)

                try:
                    self.cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                except Exception:
                    self.cv_scores = np.array([float(study.best_value)])
            elif self.task_type == 'clustering' and param_grid:
                # For clustering, we need to evaluate different number of clusters
                best_score = -1
                best_model = None
                best_params = {}
                
                for n_clusters in param_grid.get('n_clusters', [2, 3, 4, 5]):
                    model = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=10)
                    model.fit(X)
                    
                    # Evaluate clustering quality
                    try:
                        score = silhouette_score(X, model.labels_)
                    except:
                        score = -1
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params = {'n_clusters': n_clusters}
                
                self.model = best_model
                self.best_params = best_params
                self.cv_scores = [best_score]
            else:
                # No tuning grid, use base model
                self.model = base_model
                if self.task_type in ['classification', 'regression', 'anomaly_detection']:
                    self.model.fit(X, y)
                elif self.task_type in ['clustering', 'dimensionality_reduction']:
                    self.model.fit(X)
        
        training_time = perf_counter() - start_time
        logger.info(
            "Completed model fit in %.3f seconds. Best params=%s cv_scores=%s",
            training_time,
            self.best_params,
            getattr(self, 'cv_scores', None)
        )
        return training_time
    
    def get_model(self):
        """Get the trained model"""
        return self.model

# ==================== LOCAL DATASET ANALYZER (No External API Required) ====================
class LocalDatasetAnalyzer:
    """
    Fully independent, data-driven dataset analyzer.
    Analyzes dataset properties using statistical methods — no external API required.
    """
    def __init__(self):
        self.calibrator = PerformanceCalibrator()
    
    def analyze_dataset(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Analyze dataset using statistical methods and return structured insights."""
        try:
            summary = self._create_dataset_summary(df, target_col)
            return self._generate_analysis(df, summary, target_col)
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _create_dataset_summary(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Create comprehensive dataset summary."""
        summary: Dict[str, Any] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict(),
            "numerical_columns": list(df.select_dtypes(include=np.number).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "basic_stats": {},
            "correlation_info": {}
        }

        if summary["numerical_columns"]:
            summary["basic_stats"] = df[summary["numerical_columns"]].describe().to_dict()
            if len(summary["numerical_columns"]) >= 2:
                try:
                    corr_matrix = df[summary["numerical_columns"]].corr()
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                "features": [corr_matrix.columns[i], corr_matrix.columns[j]],
                                "correlation": float(corr_matrix.iloc[i, j])
                            })
                    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                    summary["correlation_info"]["top_correlations"] = corr_pairs[:5]
                except Exception:
                    pass

        if target_col and target_col in df.columns:
            summary["target_analysis"] = {
                "name": target_col,
                "dtype": str(df[target_col].dtype),
                "unique_values": int(df[target_col].nunique()),
                "missing_count": int(df[target_col].isnull().sum()),
                "value_counts": df[target_col].value_counts().to_dict()
            }
            if pd.api.types.is_numeric_dtype(df[target_col]):
                summary["task_type"] = "regression" if df[target_col].nunique() > 10 else "classification"
            else:
                summary["task_type"] = "classification"

        return summary

    def _compute_quality_score(self, df: pd.DataFrame, summary: Dict) -> Tuple[int, List[str], List[str]]:
        """Compute a quality score 0–100 from dataset properties."""
        score = 100
        issues: List[str] = []
        strengths: List[str] = []
        n, m = df.shape

        if n < 100:
            score -= 25
            issues.append("Very small dataset (<100 rows) — high risk of overfitting.")
        elif n < 500:
            score -= 10
            issues.append("Small dataset (<500 rows) — prefer simple, regularized models.")
        elif n >= 5000:
            strengths.append(f"Sufficient data ({n:,} rows) for training complex models.")

        total_cells = max(n * m, 1)
        missing = sum(summary["missing_values"].values())
        missing_pct = missing / total_cells
        if missing_pct > 0.3:
            score -= 20
            issues.append(f"High missing value ratio ({missing_pct:.1%}) — aggressive imputation needed.")
        elif missing_pct > 0.1:
            score -= 10
            issues.append(f"Moderate missing values ({missing_pct:.1%}) — imputation recommended.")
        elif missing_pct == 0.0:
            strengths.append("No missing values — complete dataset.")

        if m > n:
            score -= 15
            issues.append("More features than samples — high overfitting risk; consider dimensionality reduction.")
        elif m > 100:
            score -= 5
            issues.append(f"High feature count ({m}) — feature selection may improve performance.")
        elif 5 <= m <= 50:
            strengths.append(f"Well-proportioned feature count ({m} features).")

        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_pct = dup_count / n
            score -= min(10, int(dup_pct * 50))
            issues.append(f"{dup_count} duplicate rows detected ({dup_pct:.1%}) — de-duplication recommended.")

        if "target_analysis" in summary and summary.get("task_type") == "classification":
            vc = list(summary["target_analysis"].get("value_counts", {}).values())
            if len(vc) >= 2 and min(vc) > 0:
                imbalance = max(vc) / min(vc)
                if imbalance > 5:
                    score -= 10
                    issues.append(f"Class imbalance ratio ~{imbalance:.1f}x — enable class weighting.")

        score = max(0, min(100, score))
        if not strengths:
            if score >= 60:
                strengths.append("Dataset structure is generally suitable for machine learning.")
        return score, issues, strengths

    def _compute_dataset_scores(self, df: pd.DataFrame, summary: Dict, task_type: str) -> Dict[str, float]:
        """Compute nonlinearity, sparsity, and outlier scores for model recommendation."""
        target_name = summary.get("target_analysis", {}).get("name")
        X = df.copy()
        y = None
        if target_name and target_name in df.columns:
            y = df[target_name].dropna()
            X = df.drop(columns=[target_name]).loc[y.index]

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        if numeric_features:
            nonzero_ratio = np.count_nonzero(X[numeric_features]) / max(X[numeric_features].size, 1)
            sparsity = 1.0 - nonzero_ratio

            try:
                kurt_vals = stats.kurtosis(X[numeric_features], axis=0, nan_policy="omit")
                skew_vals = stats.skew(X[numeric_features], axis=0, nan_policy="omit")
                kurt_vals = np.nan_to_num(kurt_vals)
                skew_vals = np.nan_to_num(skew_vals)
                kurt_norm = np.minimum(1.0, np.mean(np.abs(kurt_vals)) / 10.0)
                skew_norm = np.minimum(1.0, np.mean(np.abs(skew_vals)) / 5.0)
                outlier_score = (kurt_norm + skew_norm) / 2.0
            except Exception:
                outlier_score = 0.0
        else:
            sparsity = 0.0
            outlier_score = 0.0

        nonlinearity_score = 0.0
        if y is not None and numeric_features:
            try:
                if task_type == 'classification':
                    y_encoded = pd.factorize(y)[0]
                    mi_vals = mutual_info_classif(X[numeric_features], y_encoded, discrete_features='auto')
                else:
                    mi_vals = mutual_info_regression(X[numeric_features], y)
                mi_vals = np.nan_to_num(mi_vals, nan=0.0, posinf=0.0, neginf=0.0)
                mi_norm = mi_vals / (np.max(mi_vals) + 1e-8)
                mi_score = np.mean(mi_norm)

                corr_vals = []
                for col in numeric_features:
                    if X[col].nunique(dropna=False) > 1:
                        corr = np.corrcoef(X[col].fillna(X[col].mean()), y.astype(float).fillna(y.mean()))[0, 1]
                        corr_vals.append(abs(corr) if not np.isnan(corr) else 0.0)
                corr_score = np.mean(corr_vals) if corr_vals else 0.0
                correlation_gap = max(0.0, mi_score - corr_score)
                nonlinearity_score = min(1.0, 0.6 * mi_score + 0.4 * correlation_gap)
            except Exception:
                nonlinearity_score = 0.0

        return {
            'nonlinearity': float(nonlinearity_score),
            'sparsity': float(sparsity),
            'outlier': float(outlier_score),
            'nonzero_ratio': float(nonzero_ratio if numeric_features else 1.0)
        }

    def _generate_algorithm_recommendations(self, df: pd.DataFrame, summary: Dict,
                                             task_type: str) -> Dict[str, List]:
        """Generate data-driven algorithm recommendations with estimated performance ranges."""
        n, m = df.shape
        total_cells = max(n * m, 1)
        missing_pct = sum(summary["missing_values"].values()) / total_cells

        size_factor = 0.75 if n < 100 else (0.88 if n < 500 else 1.0)
        quality_factor = max(0.80, 1.0 - missing_pct * 0.5)
        factor = size_factor * quality_factor

        scores = self._compute_dataset_scores(df, summary, task_type)
        nonlinearity = scores["nonlinearity"]
        sparsity = scores["sparsity"]
        outlier = scores["outlier"]
        density = scores["nonzero_ratio"]

        def acc_range(base: float, spread: float = 0.05) -> str:
            lo = max(40, int((base - spread) * 100))
            hi = min(97, int((base + spread) * 100))
            return f"{lo}-{hi}%"

        def r2_range(base: float, spread: float = 0.06) -> str:
            return f"{max(0.05, base - spread):.2f}-{min(0.97, base + spread):.2f}"

        base_acc = min(0.90, 0.80 * factor)
        base_r2 = min(0.88, 0.72 * factor)

        if task_type == 'classification':
            algos = [
                {
                    "algorithm": "XGBoost",
                    "score": base_acc - 0.04 + 0.05 * nonlinearity + 0.02 * outlier,
                    "reason": (
                        "High nonlinearity detected; boosting models are preferred."
                        if nonlinearity > 0.45 else
                        "Good choice for moderate nonlinearity with structured tabular data."
                    )
                },
                {
                    "algorithm": "LightGBM",
                    "score": base_acc - 0.08 + 0.05 * nonlinearity + 0.01 * (1 - sparsity),
                    "reason": (
                        "LightGBM is recommended for large datasets with complex interactions."
                        if nonlinearity > 0.35 else
                        "Strong gradient boosting option with efficient training."
                    )
                },
                {
                    "algorithm": "Random Forest",
                    "score": base_acc - 0.07 + 0.03 * outlier + 0.01 * (1 - sparsity),
                    "reason": (
                        "Robust to outliers and mixed feature types."
                        if outlier > 0.35 else
                        "Versatile ensemble for general-purpose classification."
                    )
                },
                {
                    "algorithm": "Gradient Boosting",
                    "score": base_acc - 0.10 + 0.04 * nonlinearity,
                    "reason": "Effective boosted trees for non-linear relationships and medium-sized datasets."
                },
                {
                    "algorithm": "Logistic Regression",
                    "score": base_acc - 0.13 + 0.03 * (1 - nonlinearity) + 0.02 * (1 - sparsity),
                    "reason": (
                        "High sparsity and linear structure favor logistic regression."
                        if sparsity > 0.6 else
                        "Simple and interpretable baseline for classification."
                    )
                },
            ]
        elif task_type == 'regression':
            algos = [
                {
                    "algorithm": "XGBoost",
                    "score": base_r2 - 0.06 + 0.05 * nonlinearity + 0.02 * outlier,
                    "reason": (
                        "High nonlinearity detected; boosting models are preferred."
                        if nonlinearity > 0.45 else
                        "Strong non-linear regression model for structured data."
                    )
                },
                {
                    "algorithm": "Gradient Boosting",
                    "score": base_r2 - 0.12 + 0.04 * nonlinearity,
                    "reason": "Good choice for capturing moderate non-linear relationships in regression tasks."
                },
                {
                    "algorithm": "Random Forest",
                    "score": base_r2 - 0.10 + 0.03 * outlier,
                    "reason": "Robust to outliers and effective for mixed-feature regression."
                },
                {
                    "algorithm": "Ridge Regression",
                    "score": base_r2 - 0.18 + 0.03 * (1 - nonlinearity),
                    "reason": (
                        "Regularized linear model for stable performance on moderately linear data."
                        if nonlinearity < 0.35 else
                        "Good linear baseline when the signal is not strongly non-linear."
                    )
                },
                {
                    "algorithm": "Linear Regression",
                    "score": base_r2 - 0.20 + 0.02 * (1 - nonlinearity),
                    "reason": "Interpretable baseline for predominantly linear relationships."
                },
            ]
        else:
            algos = []

        recommendations: Dict[str, List] = {"classification": [], "regression": [],
                                           "clustering": [], "dimensionality_reduction": [],
                                           "anomaly_detection": []}

        if task_type in ['classification', 'regression']:
            if task_type == 'classification':
                recs = []
                for item in sorted(algos, key=lambda x: x["score"], reverse=True):
                    recs.append({
                        "algorithm": item["algorithm"],
                        "estimated_accuracy": acc_range(item["score"]),
                        "reason": item["reason"],
                        "calibrated_numeric": self.calibrator.calibrate_estimate(acc_range(item["score"]), task_type)
                    })
                recommendations[task_type] = recs
            else:
                recs = []
                for item in sorted(algos, key=lambda x: x["score"], reverse=True):
                    recs.append({
                        "algorithm": item["algorithm"],
                        "estimated_r2": r2_range(item["score"]),
                        "reason": item["reason"],
                        "calibrated_numeric": self.calibrator.calibrate_estimate(r2_range(item["score"]), task_type)
                    })
                recommendations[task_type] = recs
        else:
            recommendations[task_type] = [
                {"algorithm": "K-Means", "estimated_silhouette": "0.30-0.55",
                 "reason": "Fast and scalable — best for compact, spherical clusters."},
                {"algorithm": "DBSCAN", "estimated_silhouette": "0.20-0.45",
                 "reason": "Handles arbitrary cluster shapes and identifies noise/outliers."},
                {"algorithm": "Hierarchical", "estimated_silhouette": "0.25-0.50",
                 "reason": "No need to pre-specify cluster count; produces an informative dendrogram."},
                {"algorithm": "Gaussian Mixture", "estimated_silhouette": "0.25-0.48",
                 "reason": "Soft probabilistic assignments with flexible covariance shapes."},
            ]

        return recommendations

    def _generate_preprocessing_recommendations(self, summary: Dict) -> List[str]:
        recs = ["StandardScaler applied to normalize all numerical features."]
        if sum(summary["missing_values"].values()) > 0:
            recs.append("Median imputation for numerical columns; most-frequent for categorical columns.")
        if summary.get("categorical_columns"):
            recs.append(f"OneHotEncoding applied to {len(summary['categorical_columns'])} categorical feature(s).")
        if len(summary.get("numerical_columns", [])) > 10:
            recs.append("VarianceThreshold applied to remove near-zero variance features.")
        recs.append("Recommended train/test split: 80/20 with stratification for classification tasks.")
        return recs

    def _generate_insights(self, df: pd.DataFrame, summary: Dict, task_type: str) -> List[str]:
        insights: List[str] = []
        n, m = df.shape
        insights.append(f"Dataset contains {n:,} samples and {m} features.")
        num_c = len(summary.get("numerical_columns", []))
        cat_c = len(summary.get("categorical_columns", []))
        insights.append(f"{num_c} numerical and {cat_c} categorical features detected.")

        if "correlation_info" in summary and "top_correlations" in summary["correlation_info"]:
            top_corr = summary["correlation_info"]["top_correlations"]
            if top_corr:
                best = top_corr[0]
                insights.append(
                    f"Strongest correlation: {best['features'][0]} ↔ {best['features'][1]} "
                    f"(r = {best['correlation']:.2f})."
                )

        if "target_analysis" in summary:
            ta = summary["target_analysis"]
            if task_type == "classification":
                insights.append(f"Target '{ta['name']}' has {ta['unique_values']} unique classes.")
            else:
                insights.append(f"Target '{ta['name']}' is continuous — regression task detected.")

        if n > 10000:
            insights.append("Large dataset: ensemble and gradient boosting methods recommended.")
        elif n < 200:
            insights.append("Small dataset: regularized models and cross-validation are essential.")
        if m > n:
            insights.append("High dimensionality: apply PCA or feature selection to mitigate overfitting.")
        return insights

    def _generate_warnings(self, df: pd.DataFrame, summary: Dict) -> List[str]:
        warnings_list: List[str] = []
        n, m = df.shape
        total_cells = max(n * m, 1)
        missing_pct = sum(summary["missing_values"].values()) / total_cells
        if missing_pct > 0.1:
            warnings_list.append(f"High missing value ratio ({missing_pct:.1%}) may degrade model performance.")
        if n < 100:
            warnings_list.append("Very small dataset — results may be statistically unstable.")
        if m > n:
            warnings_list.append("More features than samples — regularization is strongly recommended.")
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings_list.append(f"{dup_count} duplicate rows may bias model training.")
        if not warnings_list:
            warnings_list.append("No critical data quality issues detected.")
        return warnings_list

    def _generate_next_steps(self, summary: Dict, task_type: str) -> List[str]:
        steps = [
            "Use 'Run Guided AutoML Pipeline' to run insights, benchmark, and training in one workflow.",
            "Review the benchmark and recommendation cards before final model selection.",
        ]
        if task_type == "classification":
            steps.append("For imbalanced classes, enable 'Handle class imbalance' in the training options.")
        elif task_type == "regression":
            steps.append("Review the Actual vs Predicted scatter plot to assess model fit quality.")
        elif task_type == "clustering":
            steps.append("Experiment with different cluster counts and evaluate the Silhouette score.")
        steps.append("Download the trained model after evaluation for deployment or further analysis.")
        return steps

    def _estimate_training_time(self, n_samples: int, n_features: int) -> str:
        complexity = n_samples * n_features
        if complexity < 10_000:
            return "< 30 seconds"
        elif complexity < 100_000:
            return "30 seconds – 2 minutes"
        elif complexity < 1_000_000:
            return "2 – 10 minutes"
        else:
            return "10+ minutes (consider subsampling for exploratory runs)"

    def _generate_summary_markdown(self, df: pd.DataFrame, summary: Dict,
                                    quality_score: int, insights: List[str],
                                    task_type: str) -> str:
        n, m = df.shape
        task_display = task_type.replace("_", " ").title() if task_type else "Unknown"
        md = (
            f"**Dataset:** {n:,} rows × {m} columns  \n"
            f"**Detected Task:** {task_display}  \n"
            f"**Quality Score:** {quality_score}/100\n\n"
            f"### Key Observations\n"
        )
        for obs in insights[:6]:
            md += f"- {obs}\n"
        return md

    def _generate_analysis(self, df: pd.DataFrame, summary: Dict,
                            target_col: str = None) -> Dict[str, Any]:
        """Orchestrate the full data-driven analysis."""
        task_type = summary.get("task_type", "unknown")
        n, m = df.shape
        quality_score, quality_issues, quality_strengths = self._compute_quality_score(df, summary)
        algo_recs     = self._generate_algorithm_recommendations(df, summary, task_type)
        pre_recs      = self._generate_preprocessing_recommendations(summary)
        insights      = self._generate_insights(df, summary, task_type)
        warnings_list = self._generate_warnings(df, summary)
        next_steps    = self._generate_next_steps(summary, task_type)
        est_time      = self._estimate_training_time(n, m)
        summary_md    = self._generate_summary_markdown(df, summary, quality_score, insights, task_type)

        return {
            "dataset_quality": {
                "score": quality_score,
                "issues": quality_issues,
                "strengths": quality_strengths
            },
            "algorithm_recommendations": algo_recs,
            "data_preprocessing_recommendations": pre_recs,
            "insights": insights,
            "warnings": warnings_list,
            "next_steps": next_steps,
            "estimated_training_time": est_time,
            "realism_factors": warnings_list,
            "summary_markdown": summary_md
        }

# ==================== HELPER FUNCTIONS FOR DIFFERENT TASK TYPES ====================

def perform_clustering(X, algorithm='K-Means', n_clusters=3):
    """Perform clustering on the dataset"""
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=10)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters, random_state=DEFAULT_RANDOM_STATE)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=DEFAULT_RANDOM_STATE, n_init=10)
    
    labels = model.fit_predict(X)
    
    # Calculate metrics
    metrics = {}
    unique_labels = np.unique(labels[labels != -1]) if -1 in labels else np.unique(labels)
    
    if len(unique_labels) > 1:
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
        except:
            metrics['silhouette_score'] = None
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        except:
            metrics['calinski_harabasz_score'] = None
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        except:
            metrics['davies_bouldin_score'] = None
    
    return labels, metrics, model

def perform_dimensionality_reduction(X, algorithm='PCA', n_components=2):
    """Perform dimensionality reduction on the dataset"""
    if algorithm == 'PCA':
        model = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    elif algorithm == 't-SNE':
        model = TSNE(n_components=n_components, random_state=DEFAULT_RANDOM_STATE, perplexity=30)
    elif algorithm == 'UMAP':
        model = UMAP(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    elif algorithm == 'ICA':
        model = FastICA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    elif algorithm == 'NMF':
        model = NMF(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    else:
        model = PCA(n_components=n_components, random_state=DEFAULT_RANDOM_STATE)
    
    X_reduced = model.fit_transform(X)
    
    metrics = {}
    if hasattr(model, 'explained_variance_ratio_'):
        metrics['explained_variance_ratio'] = model.explained_variance_ratio_
        metrics['total_variance_explained'] = np.sum(model.explained_variance_ratio_)
    
    return X_reduced, metrics, model

def perform_anomaly_detection(X, algorithm='Isolation Forest', contamination=0.1):
    """Perform anomaly detection on the dataset"""
    if algorithm == 'Isolation Forest':
        model = IsolationForest(contamination=contamination, random_state=DEFAULT_RANDOM_STATE)
    elif algorithm == 'Local Outlier Factor':
        model = LocalOutlierFactor(contamination=contamination, novelty=True)
    else:
        model = IsolationForest(contamination=contamination, random_state=DEFAULT_RANDOM_STATE)
    
    if algorithm == 'Local Outlier Factor':
        model.fit(X)
        predictions = model.predict(X)
    else:
        predictions = model.fit_predict(X)
    
    # Convert predictions: -1 for anomalies, 1 for normal
    anomaly_labels = np.where(predictions == -1, 1, 0)
    anomaly_count = np.sum(anomaly_labels)
    
    return anomaly_labels, anomaly_count, model

def plot_cluster_results(X, labels, algorithm_name):
    """Visualize clustering results"""
    # Reduce dimensions for visualization if needed
    if X.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                         c=labels, cmap='tab20', alpha=0.7, s=50)
    ax.set_xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
    ax.set_ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
    ax.set_title(f'{algorithm_name} - Cluster Visualization')
    plt.colorbar(scatter, ax=ax)
    
    return fig

def plot_dimensionality_reduction(X_reduced, algorithm_name):
    """Visualize dimensionality reduction results"""
    if X_reduced.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=30)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'{algorithm_name} - Dimensionality Reduction')
        ax.grid(True, alpha=0.3)
        return fig
    return None

def plot_anomaly_detection(X, anomaly_labels, algorithm_name):
    """Visualize anomaly detection results"""
    # Reduce dimensions for visualization if needed
    if X.shape[1] > 2:
        reducer = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot normal points
    normal_mask = anomaly_labels == 0
    if np.any(normal_mask):
        ax.scatter(X_reduced[normal_mask, 0], X_reduced[normal_mask, 1], 
                  c='blue', alpha=0.5, s=30, label='Normal')
    
    # Plot anomalies
    anomaly_mask = anomaly_labels == 1
    if np.any(anomaly_mask):
        ax.scatter(X_reduced[anomaly_mask, 0], X_reduced[anomaly_mask, 1], 
                  c='red', alpha=0.8, s=50, label='Anomaly', marker='x')
    
    ax.set_xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
    ax.set_ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
    ax.set_title(f'{algorithm_name} - Anomaly Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def get_performance_string(algo_dict: Dict, task_type: str) -> str:
    """Get formatted performance string based on task type"""
    def to_range(val):
        try:
            if isinstance(val, str):
                if '%' in val:
                    n = float(val.replace('%', '').strip()) / 100.0
                elif '-' in val: # Already a range
                    return val
                else:
                    n = float(val.strip())
            else:
                n = float(val)
            low = max(0, int(n * 100) - 4)
            high = min(100, int(n * 100) + 4)
            return f"{low}-{high}%"
        except:
            return str(val)

    if task_type == 'classification':
        return to_range(algo_dict.get('estimated_accuracy', 'N/A'))
    elif task_type == 'regression':
        return f"R²: {to_range(algo_dict.get('estimated_r2', 'N/A'))}"
    elif task_type == 'clustering':
        return f"Silhouette: {to_range(algo_dict.get('estimated_silhouette', 'N/A'))}"
    elif task_type == 'dimensionality_reduction':
        return f"Variance: {to_range(algo_dict.get('estimated_variance', 'N/A'))}"
    elif task_type == 'anomaly_detection':
        return f"Precision: {to_range(algo_dict.get('estimated_precision', 'N/A'))}"
    return "N/A"

def display_calibrated_accuracy_estimates(analysis_result: Dict[str, Any], task_type: str, 
                                         actual_performance: float = None):
    """Display calibrated accuracy estimates with comparison to actual"""
    if "error" in analysis_result or "algorithm_recommendations" not in analysis_result:
        return
    
    recommendations = analysis_result["algorithm_recommendations"]
    task_recommendations = recommendations.get(task_type, [])
    
    if not task_recommendations:
        return
    
    st.subheader("Algorithm Insights")
    st.write("Performance forecasts based on your dataset characteristics:")
    
    # Create comparison table
    algo_data = []
    for algo in task_recommendations[:5]:  # Show top 5
        row = {
            "Algorithm": algo.get("algorithm", "Unknown"),
            "Estimated Performance": get_performance_string(algo, task_type),
            "Reason": algo.get("reason", "No reason provided")[:100] + "..." if len(algo.get("reason", "")) > 100 else algo.get("reason", "")
        }
        
        if 'calibrated_numeric' in algo:
            conf = algo['calibrated_numeric']
            low = max(0, int(conf * 100) - 4)
            high = min(100, int(conf * 100) + 4)
            row["Confidence"] = f"{low}-{high}%"
        
        algo_data.append(row)
    
    # Display as table
    df_display = pd.DataFrame(algo_data)
    st.dataframe(df_display, height=250, use_container_width=True)
    
    # Show best algorithm with calibration
    if task_recommendations:
        best_algo = max(task_recommendations, key=lambda x: x.get('calibrated_numeric', 0.5))
        
        if actual_performance is not None:
            # Compare with actual
            est_perf = best_algo.get('calibrated_numeric', 0.5)
            diff = actual_performance - est_perf
            
            col1, col2, col3 = st.columns(3)
            with col1:
                low = max(0, int(est_perf * 100) - 4)
                high = min(100, int(est_perf * 100) + 4)
                st.metric("Estimated", f"{low}-{high}%")
            with col2:
                st.metric("Actual", f"{actual_performance:.1%}")
            with col3:
                st.metric("Difference", f"{diff:+.1%}", 
                         delta_color="normal" if abs(diff) < 0.1 else "inverse")
            
            if abs(diff) > 0.15:
                st.warning(f"Significant difference between estimated and actual performance.")
        else:
            st.info(f"**Recommended Algorithm**: {best_algo.get('algorithm', 'Unknown')} "
                    f"(Estimated: {get_performance_string(best_algo, task_type)})")

def enhanced_generate_report(model, X_test, y_test, task_type, cv_scores=None):
    """Enhanced report generation with cross-validation and detailed metrics"""
    try:
        if task_type == 'classification':
            y_pred = model.predict(X_test)
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC AUC if probabilities available
            roc_auc = None
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    elif len(np.unique(y_test)) > 2:
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except:
                    pass
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": cm,
                "cv_scores": cv_scores if cv_scores is not None else [],
                "cv_mean": np.mean(cv_scores) if cv_scores is not None else None,
                "cv_std": np.std(cv_scores) if cv_scores is not None else None,
                "predictions": y_pred,
                "probabilities": y_proba,
                "actuals": y_test
            }
            
        elif task_type == 'regression':
            y_pred = model.predict(X_test)
            
            # Multiple error metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            explained_var = explained_variance_score(y_test, y_pred)
            
            return {
                "r2": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "explained_variance": explained_var,
                "cv_scores": cv_scores if cv_scores is not None else [],
                "cv_mean": np.mean(cv_scores) if cv_scores is not None else None,
                "cv_std": np.std(cv_scores) if cv_scores is not None else None,
                "predictions": y_pred,
                "actuals": y_test
            }
            
        elif task_type == 'clustering':
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.predict(X_test)
            
            metrics = {}
            unique_labels = np.unique(labels[labels != -1]) if -1 in labels else np.unique(labels)
            
            if len(unique_labels) > 1:
                try:
                    # Sample if too large for silhouette score
                    if X_test.shape[0] > 5000:
                        np.random.seed(DEFAULT_RANDOM_STATE)
                        sample_indices = np.random.choice(X_test.shape[0], 5000, replace=False)
                        X_sample = X_test[sample_indices]
                        labels_sample = labels[sample_indices]
                        if len(np.unique(labels_sample[labels_sample != -1])) > 1:
                            metrics['silhouette_score'] = silhouette_score(X_sample, labels_sample)
                    else:
                        if len(np.unique(labels[labels != -1])) > 1:
                            metrics['silhouette_score'] = silhouette_score(X_test, labels)
                except:
                    metrics['silhouette_score'] = None
                
                try:
                    if len(unique_labels) > 1:
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_test, labels)
                except:
                    metrics['calinski_harabasz_score'] = None
                
                try:
                    if len(unique_labels) > 1:
                        metrics['davies_bouldin_score'] = davies_bouldin_score(X_test, labels)
                except:
                    metrics['davies_bouldin_score'] = None
            
            return {
                "labels": labels,
                "metrics": metrics,
                "cluster_counts": pd.Series(labels).value_counts().to_dict(),
                "n_clusters": len(unique_labels)
            }
            
        elif task_type == 'dimensionality_reduction':
            transformed = model.fit_transform(X_test)
            
            explained_variance = None
            if hasattr(model, 'explained_variance_ratio_'):
                explained_variance = model.explained_variance_ratio_.sum()
            
            return {
                "transformed_data": transformed,
                "explained_variance": explained_variance
            }
            
        elif task_type == 'anomaly_detection':
            if isinstance(model, LocalOutlierFactor):
                y_pred = model.fit_predict(X_test)
            else:
                y_pred = model.fit_predict(X_test)
            
            return {
                "anomalies": y_pred,
                "anomaly_count": np.sum(y_pred == -1)
            }
            
    except Exception as e:
        st.error(f"Error generating enhanced report: {str(e)}")
        return None

def plot_roc_curve(y_test, y_proba, class_names=None):
    try:
        if y_proba is None:
            st.warning("ROC curve cannot be plotted as the model does not support probability predictions.")
            return None
            
        y_test = np.array(y_test)
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 and y_proba.shape[1] > 1 else 2
        
        if n_classes == 2:
            y_scores = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend(loc="lower right")
            return fig
            
        else:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            unique_classes = np.unique(y_test)
            y_test_binarized = label_binarize(y_test, classes=unique_classes)

            for i, class_label in enumerate(unique_classes):
                if i < y_test_binarized.shape[1] and i < y_proba.shape[1]:
                    fpr[class_label], tpr[class_label], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
                    if np.isnan(fpr[class_label]).any() or np.isnan(tpr[class_label]).any():
                        roc_auc[class_label] = float('nan')
                    else:
                        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
            
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr["micro"], tpr["micro"],
                    label=f'micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                    color='deeppink', linestyle=':', linewidth=4)
            
            colors = plt.cm.get_cmap('Set1', len(unique_classes))
            for i, (class_label, color) in enumerate(zip(unique_classes, colors.colors)):
                class_name_str = f'Class {class_label}'
                if class_names is not None and class_label < len(class_names):
                    class_name_str = class_names[class_label]

                if class_label in fpr:
                    auc_score = roc_auc.get(class_label, float('nan'))
                    label_text = f'ROC curve of {class_name_str} (AUC = {auc_score:.2f})' if not np.isnan(auc_score) else f'ROC curve of {class_name_str} (AUC = nan)'
                    ax.plot(fpr[class_label], tpr[class_label], color=color, lw=2, label=label_text)

            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Multi-class ROC Curve')
            ax.legend(loc="lower right")
            return fig
            
    except Exception as e:
        st.error(f"Error plotting ROC curve: {str(e)}")
        return None

def plot_prediction_comparison(report):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(report['actuals'], report['predictions'], alpha=0.5)
        ax.plot([min(report['actuals']), max(report['actuals'])], 
                [min(report['actuals']), max(report['actuals'])], 
                'r--', label='Perfect Prediction')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        ax.legend()
        return fig
    except Exception as e:
        st.error(f"Error plotting prediction comparison: {str(e)}")
        return None

def display_gpt_analysis_results(analysis_result: Dict[str, Any], task_type: str = None):
    """Display analysis results in Streamlit"""
    if "error" in analysis_result:
        st.error(f"GPT Analysis Error: {analysis_result['error']}")
        return
    
    # Dataset Quality Score
    if "dataset_quality" in analysis_result:
        quality = analysis_result["dataset_quality"]
        score = quality.get("score", 50)
        
        # Professional Alignment for Quality Section
        st.markdown(f"""
        <div style="background: var(--surface); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--surface-border); margin-bottom: 2rem; box-shadow: var(--card-shadow);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin: 0; color: var(--text-muted); font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;">Dataset Reliability</h3>
                    <div style="font-size: 3rem; font-weight: 800; color: var(--text-main);">{score}<span style="font-size: 1.2rem; color: var(--text-muted); font-weight: 400;">/100</span></div>
                </div>
                <div style="text-align: right;">
                    <span class="status-badge" style="background: {'rgba(34, 197, 94, 0.2)' if score >= 70 else 'rgba(234, 179, 8, 0.2)' if score >= 50 else 'rgba(239, 68, 68, 0.2)'}; color: {'#4ade80' if score >= 70 else '#facc15' if score >= 50 else '#f87171'}; border: 1px solid {'#22c55e' if score >= 70 else '#eab308' if score >= 50 else '#ef4444'};">
                        {'High Quality' if score >= 70 else 'Standard' if score >= 50 else 'Action Required'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Combined Gauge and Strengths side-by-side
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### Performance Potential")
            fig, ax = plt.subplots(figsize=(4, 1.2))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.barh(0, score, color='#38bdf8', height=0.6)
            ax.barh(0, 100, color=(1, 1, 1, 0.1), height=0.6, zorder=0)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks([])
            ax.tick_params(colors='#64748b', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)

        with col2:
            if quality.get("issues") or quality.get("strengths"):
                if quality.get("strengths"):
                    st.markdown("##### Strategic Strengths")
                    for strength in quality["strengths"][:3]: # Limit to top 3 for side-by-side
                        st.markdown(f"<small>• {strength}</small>", unsafe_allow_html=True)
                if quality.get("issues"):
                    st.markdown("##### Improvement Areas")
                    for issue in quality["issues"][:3]:
                        st.markdown(f"<small>• {issue}</small>", unsafe_allow_html=True)
        
    
    # Algorithm Recommendations with Accuracy Estimates
    if "algorithm_recommendations" in analysis_result and task_type:
        with st.expander("Model Guidance", expanded=False):
            display_calibrated_accuracy_estimates(analysis_result, task_type)
    
    # Other sections
    if "data_preprocessing_recommendations" in analysis_result:
        st.markdown("---")
        st.markdown("### Preprocessing Roadmap")
        for rec in analysis_result["data_preprocessing_recommendations"]:
            st.markdown(f"• {rec}")
    
    if "insights" in analysis_result:
        st.markdown("---")
        st.markdown("### Strategic Insights")
        for insight in analysis_result["insights"]:
            st.markdown(f"• {insight}")
    
    if "warnings" in analysis_result:
        st.markdown("---")
        for warning in analysis_result["warnings"]:
            st.warning(warning)
    
    if "next_steps" in analysis_result:
        st.markdown("---")
        st.markdown("### Tactical Next Steps")
        for step in analysis_result["next_steps"]:
            st.markdown(f"• {step}")
    
    if "estimated_training_time" in analysis_result:
        st.markdown(f"**Estimated Compute Time (projection):** {analysis_result['estimated_training_time']}")
    
    # Display markdown summary
    if "summary_markdown" in analysis_result:
        st.subheader("Analysis Summary")
        st.markdown(analysis_result["summary_markdown"])

# ==================== DATA-DRIVEN ANALYSIS ENGINE ====================
class AutomatedAlgorithmBenchmarker:
    """
    Analyzes dataset properties and runs quick AutoML benchmarking 
    to provide strictly data-driven recommendations.
    """
    def __init__(self, task_type):
        self.task_type = task_type
        
    def analyze_dataset_stats(self, X, y=None):
        """Analyze statistical properties of the dataset"""
        logger.info("Analyzing dataset stats for task type %s", self.task_type)
        stats_insights = []
        if X is None:
            return stats_insights

        n_samples, n_features = X.shape
        
        # 1. Sparsity check
        sparsity = (np.count_nonzero(X) / X.size) if X.size > 0 else 0
        if sparsity < 0.2:
            stats_insights.append(f"High Sparsity ({sparsity:.1%}): Linear models or Sparse-friendly ensembles recommended.")
        
        # 2. Linearity vs Non-linearity check (Correlation vs MI)
        if y is not None and n_samples > 100:
            try:
                # Sample for speed
                idx = np.random.choice(n_samples, min(1000, n_samples), replace=False)
                X_s = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
                y_s = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx]
                
                # Simple check: MI vs Linear correlation
                if self.task_type == 'classification':
                    mi = mutual_info_classif(X_s, y_s).mean()
                else:
                    mi = mutual_info_regression(X_s, y_s).mean()
                
                if mi > 0.15:
                    stats_insights.append("Strong Non-linear patterns detected: Tree-based models (XGBoost, RF) will likely outperform linear models.")
                else:
                    stats_insights.append("Data exhibits potential linear trends: Regularized Linear models may generalize better.")
            except:
                pass
        
        # 3. Scale analysis
        if n_features > 0:
            feat_max = np.abs(X).max().max() if hasattr(X, 'max') else np.max(np.abs(X))
            if feat_max > 1000:
                stats_insights.append("Wide feature ranges detected: Scaling is critical (StandardScaler applied).")

        # 4. Unsupervised / Outlier checks (Kurtosis/Skewness)
        if y is None and n_samples > 50:
            try:
                # Check for high kurtosis as indicator of outlier potential
                if hasattr(X, 'kurtosis'):
                    kurt = X.kurtosis().mean()
                else:
                    kurt = stats.kurtosis(X).mean()
                
                if kurt > 3:
                    stats_insights.append(f"High Kurtosis ({kurt:.1f}) detected: Significant outlier potential identified. Robust models like Isolation Forest recommended.")
                elif self.task_type == 'anomaly_detection':
                    stats_insights.append("Low Kurtosis: Data is relatively compact, but subtle local outliers may still exist.")
            except:
                pass

        return stats_insights

    def run_pilot_benchmark(self, X, y):
        """Runs a quick benchmarking of representative models"""
        if X.shape[0] < 10:
            logger.info("Pilot benchmark skipped: insufficient data (%s rows).", X.shape[0])
            return ["Dataset too small for reliable benchmarking pilot (less than 10 rows)."]

        logger.info("Running pilot benchmark for %s samples, task type %s.", X.shape[0], self.task_type)
        results = []
        n_samples = X.shape[0]
        sample_size = min(2000, n_samples)
        
        X_s = X[:sample_size] if isinstance(X, np.ndarray) else X.iloc[:sample_size]
        y_s = y[:sample_size] if isinstance(y, np.ndarray) else y.iloc[:sample_size]

        if self.task_type == 'classification':
            models = {
                "Logistic Regression (Baseline)": LogisticRegression(max_iter=500),
                "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=30, max_depth=8),
                "XGBoost (Boosting)": XGBClassifier(n_estimators=30, max_depth=4, verbosity=0)
            }
            scoring = 'accuracy'
        elif self.task_type == 'regression':
            models = {
                "Linear Regression (Baseline)": LinearRegression(),
                "Random Forest (Ensemble)": RandomForestRegressor(n_estimators=30, max_depth=8),
                "XGBoost (Boosting)": XGBRegressor(n_estimators=30, max_depth=4, verbosity=0)
            }
            scoring = 'r2'
        else:
            logger.warning("Pilot benchmark unavailable for task type %s.", self.task_type)
            return ["Benchmarking not available for this task type."]

        cv_val = 3 if sample_size >= 30 else 2
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_s, y_s, cv=cv_val, scoring=scoring)
                average_score = float(scores.mean())
                results.append({'model': name, 'score': average_score})
                logger.info("Benchmark result: %s => %s", name, average_score)
            except Exception as exc:
                logger.warning("Benchmark failed for %s: %s", name, exc)
                continue

        if not results:
            logger.error("Pilot benchmark failed: no successful model evaluations.")
            return ["Benchmarking failed: Check data format."]

        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        recommendations = []
        icons = ["1.", "2.", "3."]
        for i, res in enumerate(sorted_results):
            icon = icons[i] if i < len(icons) else "•"
            low = max(0, int(res['score'] * 100) - 4)
            high = min(100, int(res['score'] * 100) + 4)
            recommendations.append(f"{icon} **{res['model']}** (Est. Accuracy/Score: {low}-{high}%)")

        if n_samples < 50:
            recommendations.insert(0, "**Note:** Results may be unstable due to small sample size.")

        return recommendations

@st.cache_data(show_spinner=False)
def cached_analyze_dataset_stats(task_type, X, y=None):
    logger.info("Cached dataset stats analysis for task_type=%s", task_type)
    return AutomatedAlgorithmBenchmarker(task_type).analyze_dataset_stats(X, y)

@st.cache_data(show_spinner=False)
def cached_run_pilot_benchmark(task_type, X, y):
    logger.info("Cached pilot benchmark for task_type=%s, samples=%s", task_type, X.shape[0])
    return AutomatedAlgorithmBenchmarker(task_type).run_pilot_benchmark(X, y)

@st.cache_data(show_spinner=False)
def cached_dataset_analysis(df, target_col=None):
    logger.info("Cached dataset analysis for dataframe shape=%s target=%s", df.shape, target_col)
    return LocalDatasetAnalyzer().analyze_dataset(df, target_col)

def run_supervised_guided_pipeline(
    df, target_col, active_task_type, model_choice,
    handle_imbalance, enable_feature_selection,
    enable_hyperparameter_tuning, scaled_trials,
    cv_folds, test_size_float
):
    """Run the guided supervised training pipeline outside Streamlit callbacks."""
    logger.info(
        "Executing guided supervised pipeline for target=%s, model=%s, task_type=%s",
        target_col, model_choice, active_task_type
    )

    df_valid = df[df[target_col].notna()].copy()
    y_raw = df_valid[target_col]
    stratify = None
    if active_task_type == 'classification':
        class_counts = y_raw.value_counts()
        if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
            stratify = y_raw

    step_times = {}

    step_start = perf_counter()
    train_df, test_df = train_test_split(
        df_valid,
        test_size=test_size_float,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=stratify
    )
    step_times['Train/Test split'] = perf_counter() - step_start
    train_indices = train_df.index
    test_indices = test_df.index

    step_start = perf_counter()
    split_preprocessor = EnhancedUniversalDataPreprocessor(
        use_dense=True,
        enable_feature_selection=enable_feature_selection
    )
    X_train, y_train, _, _ = split_preprocessor.process(
        train_df.copy(),
        target_col,
        task_type_override=active_task_type
    )
    X_test, y_test, _, _ = split_preprocessor.transform(
        test_df.copy(),
        target_col,
        task_type_override=active_task_type
    )
    step_times['Preprocessing'] = perf_counter() - step_start

    automl_model = EnhancedAutoMLModel(
        task_type=active_task_type,
        model_choice=model_choice,
        handle_imbalance=handle_imbalance if active_task_type == 'classification' else False,
        enable_tuning=enable_hyperparameter_tuning,
        tuning_trials=scaled_trials
    )
    training_time = automl_model.fit_with_tuning(X_train, y_train, cv_folds=cv_folds)
    step_times['Model training'] = training_time
    model = automl_model.get_model()

    step_start = perf_counter()
    report = enhanced_generate_report(
        model, X_test, y_test, active_task_type,
        cv_scores=automl_model.cv_scores
    )
    step_times['Evaluation'] = perf_counter() - step_start

    return {
        'model': model,
        'preprocessor': split_preprocessor,
        'report': report,
        'training_time': training_time,
        'automl_model': automl_model,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'step_times': step_times,
        'X_test': X_test,
        'y_test': y_test
    }

def get_model_downloads(model, task_type, model_name, model_choice, preprocessor=None):
    """Generate download buttons for the trained model and preprocessing bundle."""
    st.subheader("Download Trained Model")
    
    # Extract preprocessing components when available
    scaler = None
    encoder = None
    feature_selector = None
    metadata = {
        'task_type': task_type,
        'model_choice': model_choice,
        'feature_names': None,
        'feature_columns': None,
        'dropped_constant_cols': None,
        'dropped_high_card_cols': None,
        'use_dense': None,
        'enable_feature_selection': None,
        'variance_threshold_used': None,
        'label_classes': None,
    }

    if preprocessor is not None:
        try:
            scaler = preprocessor.preprocessor.named_transformers_['num']['scaler']
        except Exception:
            scaler = None
        try:
            encoder = preprocessor.preprocessor.named_transformers_['cat']['encoder']
        except Exception:
            encoder = None
        feature_selector = getattr(preprocessor, 'feature_selector', None)
        metadata.update({
            'feature_names': getattr(preprocessor, 'feature_names', None),
            'feature_columns': getattr(preprocessor, '_feature_columns', None),
            'dropped_constant_cols': getattr(preprocessor, '_dropped_constant_cols', None),
            'dropped_high_card_cols': getattr(preprocessor, '_dropped_high_card_cols', None),
            'use_dense': getattr(preprocessor, 'use_dense', None),
            'enable_feature_selection': getattr(preprocessor, 'enable_feature_selection', None),
            'variance_threshold_used': getattr(preprocessor, '_variance_threshold_used', None),
            'label_classes': getattr(getattr(preprocessor, 'label_encoder', None), 'classes_', None),
        })

    full_pipeline = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'feature_selector': feature_selector,
        'metadata': metadata,
    }

    col1, col2 = st.columns(2)
    base_key = f"{model_name}_{task_type}"

    # Pickle format
    bundle_bytes = pickle.dumps(full_pipeline)
    with col1:
        st.download_button(
            label="Download as Pickle",
            data=bundle_bytes,
            file_name=f'{model_name}.pkl',
            mime='application/octet-stream',
            key=f"{base_key}_pkl"
        )
        st.caption("Standard Python serialization format")

    # Joblib format bundle
    buffer = io.BytesIO()
    joblib.dump(full_pipeline, buffer)
    with col2:
        st.download_button(
            label="Download as Joblib",
            data=buffer.getvalue(),
            file_name=f'{model_name}.joblib',
            mime='application/octet-stream',
            key=f"{base_key}_joblib"
        )
        st.caption("Bundle includes model, scaler, encoder, feature selector, and metadata")

def enhanced_debug_data_quality(df, target_col=None):
    """Enhanced data quality debugging with preprocessing checks"""
    st.subheader("🔍 Enhanced Data Quality Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        missing_total = df.isna().sum().sum()
        missing_pct = missing_total / (df.shape[0] * df.shape[1])
        st.metric("Missing Values", f"{missing_total} ({missing_pct:.1%})")
    with col4:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_count)
    
    if target_col:
        st.subheader(f"🎯 Target Analysis: {target_col}")
        target_series = df[target_col]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Value Distribution:**")
            fig, ax = plt.subplots(figsize=(8, 4))
            if pd.api.types.is_numeric_dtype(target_series):
                target_series.hist(ax=ax, bins=30)
                ax.set_title(f'Distribution of {target_col}')
            else:
                target_series.value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Class Distribution of {target_col}')
            st.pyplot(fig)
        
        with col2:
            st.write("**Statistics:**")
            if pd.api.types.is_numeric_dtype(target_series):
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                    'Value': [
                        target_series.mean(),
                        target_series.std(),
                        target_series.min(),
                        target_series.quantile(0.25),
                        target_series.quantile(0.50),
                        target_series.quantile(0.75),
                        target_series.max()
                    ]
                })
                st.dataframe(stats_df, hide_index=True)
            else:
                st.write(f"Unique values: {target_series.nunique()}")
                value_counts = target_series.value_counts()
                if len(value_counts) > 0:
                    st.write(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0] / len(target_series):.1%})")

# ==================== MAIN APPLICATION ====================
def main():
    # Session state initialization
    if 'ui_theme' not in st.session_state:
        st.session_state.ui_theme = "Light"

    # Sidebar configuration
    st.sidebar.title("VertexML")
    st.sidebar.markdown("---")
    st.sidebar.success("AI Engine Active (Local)")

    st.sidebar.subheader("Interface")
    selected_theme = st.sidebar.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.ui_theme == "Light" else 1,
        key="ui_theme_selector"
    )
    st.session_state.ui_theme = selected_theme
    apply_custom_styles("dark" if st.session_state.ui_theme == "Dark" else "light")

    # Hero Section
    st.markdown("""
    <div class="hero-panel">
        <h1 class="hero-title">VertexML</h1>
        <p class="hero-subtitle">End-to-end model training platform with guided workflows for first-time users.</p>
        <div class="hero-badge">Guided AutoML Workspace</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state initialization
    keys = {
        'performance_calibrator': PerformanceCalibrator(),
        'actual_vs_estimated': [],
        'gpt_analysis': None,
        'analysis_signature': None,
        'analysis_target': None,
        'supervised_suggestions_signature': None,
        'data_insights': None,
        'pilot_results': None,
        'clustering_insights': None,
        'dr_insights': None,
        'anomaly_insights': None,
        'guided_pipeline_signature': None,
        'guided_pipeline_results': None,
        'run_supervised_clicked': False,
        'run_clustering_clicked': False,
        'run_dr_clicked': False,
        'run_anomaly_clicked': False
    }
    for key, value in keys.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Enhanced configuration
    st.sidebar.subheader("Enhanced Settings")
    enable_hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True)
    enable_cross_validation = st.sidebar.checkbox("Enable Cross-Validation", value=True)
    enable_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=True)
    training_depth = st.sidebar.selectbox(
        "Training Depth",
        ["Fast (seconds)", "Standard", "Thorough"],
        index=0,
        help="Higher depth explores more hyperparameter combinations and can take longer."
    )

    tuning_trials_map = {
        "Fast (seconds)": 3,
        "Standard": 8,
        "Thorough": 20
    }
    tuning_trials = tuning_trials_map.get(training_depth, 15)
    
    # Initialize local analyzer (no external API required)
    analyzer = LocalDatasetAnalyzer()
    
    # Session state management
    def reset_all_runs():
        keys_to_reset = ['run_supervised_clicked', 'run_clustering_clicked', 
                        'run_dr_clicked', 'run_anomaly_clicked',
                        'data_insights', 'pilot_results', 'clustering_insights',
                        'dr_insights', 'anomaly_insights', 'analysis_target',
                        'supervised_suggestions_signature', 'guided_pipeline_signature',
                        'guided_pipeline_results']
        for key in keys_to_reset:
            if key in st.session_state:
                st.session_state[key] = False if 'clicked' in key else None
    
    for key in ['run_supervised_clicked', 'run_clustering_clicked', 'run_dr_clicked', 'run_anomaly_clicked']:
        if key not in st.session_state:
            st.session_state[key] = False
    
    # File uploader
    uploaded_file = st.file_uploader("Upload dataset", 
                                     type=["csv", "xlsx", "xls", "parquet", "feather", "txt"], 
                                     on_change=reset_all_runs)
    
    if uploaded_file:
        df = load_large_file(uploaded_file)
        if df is None:
            st.error("Failed to load dataset")
            return
        
        st.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")
        
        # Dataset preview
        with st.expander("Preview Dataset", expanded=False):
            st.dataframe(df, height=300, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with col2:
                st.metric("Missing Values", f"{df.isna().sum().sum():,}")
            with col3:
                st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
        
        # Enhanced debug mode
        if st.checkbox("Enable Enhanced Debug Mode", key="enhanced_debug"):
            target_for_debug = st.selectbox("Select target for debug analysis", [None] + list(df.columns))
            enhanced_debug_data_quality(df, target_for_debug)

        # Task selection (Moved up to avoid UnboundLocalError)
        task_type = st.radio("Select task type:", 
                            ["Supervised Learning", "Clustering", "Dimensionality Reduction", 
                             "Anomaly Detection"],
                            horizontal=True, key="task_type_radio", on_change=reset_all_runs)

        st.markdown("""
        <div class='action-panel'>
            <p class='action-panel-title'>Dataset Intelligence Center</p>
            <p class='action-panel-text'>Your report is generated automatically. Choose a target, review the guidance if needed, then start the guided training workflow.</p>
        </div>
        """, unsafe_allow_html=True)

        # Auto-generate Data-Driven Analysis Report (fully local, no API key needed)
        current_signature = f"{uploaded_file.name}:{uploaded_file.size}:{df.shape[0]}:{df.shape[1]}"
        if st.session_state.get('analysis_signature') != current_signature:
            with st.spinner("Building dataset intelligence report..."):
                analysis_result = cached_dataset_analysis(df)

                if "error" not in analysis_result:
                    st.session_state.gpt_analysis = analysis_result
                    st.session_state.analysis_signature = current_signature
                else:
                    st.error(f"Analysis Error: {analysis_result['error']}")

        analysis_target = None
        if task_type == "Supervised Learning" and 'target_col' in st.session_state:
            analysis_target = st.session_state.get('target_col')
            if analysis_target and st.session_state.get('analysis_target') != analysis_target:
                refreshed_analysis = cached_dataset_analysis(df, analysis_target)
                if "error" not in refreshed_analysis:
                    st.session_state.gpt_analysis = refreshed_analysis
                    st.session_state.analysis_target = analysis_target
        
        # Persistently display report if it exists
        if st.session_state.gpt_analysis:
            with st.expander("Intelligent Dataset Report", expanded=True):
                # Determine current task type for calibrated suggestions
                display_task = None
                if task_type == "Supervised Learning":
                    if analysis_target and st.session_state.gpt_analysis.get('summary_markdown'):
                        display_task = st.session_state.get('manual_task_mode', 'classification').lower()
                    else:
                        display_task = st.session_state.get('manual_task_mode', 'classification').lower()
                elif task_type == "Clustering":
                    display_task = 'clustering'
                elif task_type == "Dimensionality Reduction":
                    display_task = 'dimensionality_reduction'
                elif task_type == "Anomaly Detection":
                    display_task = 'anomaly_detection'
                
                display_gpt_analysis_results(st.session_state.gpt_analysis, display_task)
        
        # ==================== SUPERVISED LEARNING ====================
        if task_type == "Supervised Learning":
            st.subheader("Supervised Learning Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Select target column", df.columns, 
                                         key="target_col", on_change=reset_all_runs)
            with col2:
                manual_task_mode = st.radio("Prediction goal:", ["Auto (Recommended)", "Classification", "Regression"], 
                                           horizontal=True, key="manual_task_mode")
            
            if target_col:
                try:
                    # Automatic Preprocessing
                    proc_res = get_processed_data(df, target_col, enable_feature_selection)
                    X, y, detected_task_type, indices, preprocessor = \
                        proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
                    
                    # Beginner-friendly mode selection with safety fallback.
                    active_task_type = detected_task_type if manual_task_mode == "Auto (Recommended)" else manual_task_mode.lower()

                    if active_task_type == 'classification':
                        y_series = pd.Series(y)
                        unique_classes = int(y_series.nunique())
                        unique_ratio = unique_classes / max(len(y_series), 1)
                        if pd.api.types.is_numeric_dtype(y_series) and unique_classes > 20 and unique_ratio > 0.2:
                            st.warning(
                                "Classification was selected, but the target behaves like a continuous variable "
                                "(many unique numeric values). Switched to regression automatically."
                            )
                            active_task_type = 'regression'
                    
                    st.markdown(f"""
                        <div class='info-banner'>
                            <strong>Preprocessing Complete:</strong>
                            Dataset optimized for <b>{active_task_type}</b>.
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display target statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Samples", len(y))
                    with col2:
                        if active_task_type == 'classification':
                            st.metric("Classes", y.nunique() if hasattr(y, 'nunique') else len(np.unique(y)))
                        else:
                            st.metric("Range", f"{y.min():.2f} - {y.max():.2f}")
                    
                    # Algorithm selection
                    st.subheader("Data-Driven Algorithm Discovery")
                    
                    benchmarker = AutomatedAlgorithmBenchmarker(active_task_type)
                    suggestions_signature = f"{current_signature}:{target_col}:{active_task_type}:{len(y)}:{X.shape[1]}"

                    # Generate guidance before training so users can review suggestions first.
                    if st.session_state.get('supervised_suggestions_signature') != suggestions_signature:
                        with st.spinner("Preparing pre-training guidance..."):
                            st.session_state.data_insights = cached_analyze_dataset_stats(active_task_type, X, y)
                            if not st.session_state.data_insights:
                                st.session_state.data_insights = [
                                    "Standard dataset structure detected. No extreme outliers or sparsity issues identified."
                                ]
                            st.session_state.pilot_results = cached_run_pilot_benchmark(active_task_type, X, y)
                            st.session_state.supervised_suggestions_signature = suggestions_signature

                    st.markdown("""
                    <div class='action-panel'>
                        <p class='action-panel-title'>Training First</p>
                        <p class='action-panel-text'>The primary action below trains the model first. Guidance and benchmark summaries are shown before training for quick review.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Optional guidance, kept secondary to the training workflow.
                    gpt_suggestions = []
                    if st.session_state.gpt_analysis:
                        gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get(active_task_type, [])

                    has_dataset_checks = bool(st.session_state.get('data_insights') or st.session_state.get('pilot_results'))
                    if gpt_suggestions or has_dataset_checks:
                        with st.expander("Model Recommendation Brief", expanded=False):
                            if gpt_suggestions:
                                st.markdown("**Recommended Models**")
                                for suggestion in gpt_suggestions[:3]:
                                    perf_str = get_performance_string(suggestion, active_task_type)
                                    st.write(f"• **{suggestion['algorithm']}** - {perf_str}")

                            if st.session_state.get('data_insights') or st.session_state.get('pilot_results'):
                                st.markdown("**Validation Summary**")

                            if st.session_state.get('data_insights'):
                                for insight in st.session_state.data_insights:
                                    st.write(f"• {insight}")

                            if st.session_state.get('pilot_results'):
                                for res in st.session_state.pilot_results:
                                    cleaned_result = str(res)
                                    cleaned_result = cleaned_result.replace("Est. Accuracy/Score", "Estimated score")
                                    cleaned_result = cleaned_result.replace("Pilot Benchmark", "Validation")
                                    cleaned_result = cleaned_result.replace("1. ", "")
                                    cleaned_result = cleaned_result.replace("2. ", "")
                                    cleaned_result = cleaned_result.replace("3. ", "")
                                    st.write(f"• {cleaned_result}")

                            st.info("These recommendations are prepared before training to help you choose a stronger starting model.")
                    
                    # Model selection
                    if active_task_type == 'classification':
                        models = [
                            "Random Forest", "XGBoost", "Logistic Regression",
                            "Gradient Boosting", "LightGBM", "SVM"
                        ]
                        default_idx = 0
                        handle_imbalance = st.checkbox("Handle class imbalance", 
                                                      value=y.value_counts(normalize=True).max() > 0.7)
                    else:
                        models = [
                            "Random Forest", "XGBoost", "Linear Regression",
                            "Gradient Boosting", "Ridge Regression"
                        ]
                        default_idx = 0
                        handle_imbalance = False
                    
                    model_choice = st.selectbox("Select Model", models, index=default_idx, key="model_choice")
                    
                    # Configuration
                    col1, col2 = st.columns(2)
                    with col1:
                        ratio_choice = st.selectbox("Training/Testing Ratio", 
                                                   ["80/20", "70/30", "60/40", "50/50"], 
                                                   index=0, key="ratio_choice")
                        test_size_float = float(ratio_choice.split('/')[1]) / 100.0
                    
                    with col2:
                        cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 
                                            disabled=not enable_cross_validation)

                    # Scale the search effort to the uploaded dataset size.
                    dataset_size = len(y)
                    feature_count = X.shape[1]
                    scaled_trials = max(
                        tuning_trials,
                        min(20, max(4, dataset_size // 400 + feature_count // 40))
                    )
                    
                    # Run button
                    if st.button("Run Guided AutoML Pipeline", key="run_enhanced"):
                        st.session_state.run_supervised_clicked = True
                        
                except Exception as e:
                    st.error(f"Error analyzing target: {str(e)}")
                    return
            
                # Pipeline execution
                if st.session_state.run_supervised_clicked:
                    with st.spinner("Optimizing pipeline performance..."):
                        try:
                            st.session_state.run_supervised_clicked = False

                            pipeline_signature = hashlib.sha256(
                                str(
                                    (
                                        uploaded_file.name,
                                        uploaded_file.size,
                                        target_col,
                                        active_task_type,
                                        model_choice,
                                        handle_imbalance,
                                        enable_hyperparameter_tuning,
                                        scaled_trials,
                                        cv_folds,
                                        test_size_float,
                                        enable_feature_selection
                                    )
                                ).encode('utf-8')
                            ).hexdigest()

                            if st.session_state.guided_pipeline_signature != pipeline_signature or st.session_state.guided_pipeline_results is None:
                                logger.info("Executing guided pipeline for signature %s", pipeline_signature)
                                pipeline_results = run_supervised_guided_pipeline(
                                    df,
                                    target_col,
                                    active_task_type,
                                    model_choice,
                                    handle_imbalance,
                                    enable_feature_selection,
                                    enable_hyperparameter_tuning,
                                    scaled_trials,
                                    cv_folds,
                                    test_size_float
                                )
                                st.session_state.guided_pipeline_results = pipeline_results
                                st.session_state.guided_pipeline_signature = pipeline_signature
                            else:
                                logger.info("Reusing previous guided pipeline results for signature %s", pipeline_signature)
                                pipeline_results = st.session_state.guided_pipeline_results

                            model = pipeline_results['model']
                            preprocessor = pipeline_results['preprocessor']
                            report = pipeline_results['report']
                            training_time = pipeline_results['training_time']
                            step_times = pipeline_results['step_times']
                            train_indices = pipeline_results['train_indices']
                            test_indices = pipeline_results['test_indices']

                            status_text = st.empty()
                            progress_bar = st.progress(100)
                            status_text.text("Pipeline completed successfully!")
                            
                            # Display results
                            total_runtime = sum(step_times.values())
                            st.success(f"Enhanced pipeline completed in {total_runtime:.1f} seconds!")
                            st.caption(f"Actual model fit time: {training_time:.3f} seconds")
                            with st.expander("Execution Timeline", expanded=False):
                                timing_rows = []
                                for step_name, duration in step_times.items():
                                    timing_rows.append({
                                        "Step": step_name,
                                        "Time": f"{duration:.3f} s",
                                        "Time (ms)": f"{duration * 1000:.1f} ms"
                                    })
                                timing_rows.append({
                                    "Step": "Total",
                                    "Time": f"{total_runtime:.3f} s",
                                    "Time (ms)": f"{total_runtime * 1000:.1f} ms"
                                })
                                timing_df = pd.DataFrame(timing_rows)
                                st.dataframe(timing_df, hide_index=True, use_container_width=True)
                                st.caption(
                                    f"Training depth: {training_depth}. Hyperparameter trials: {scaled_trials}. "
                                    "Short steps are shown in milliseconds; larger datasets and deeper tuning will increase runtime." 
                                )
                            
                            # Performance validation
                            st.header("Performance Validation Results")
                            
                            # Get actual performance metric
                            if active_task_type == 'classification':
                                actual_perf = report['accuracy']
                            else:
                                actual_perf = report['r2']
                            
                            # Display calibrated estimates
                            if st.session_state.gpt_analysis:
                                display_calibrated_accuracy_estimates(
                                    st.session_state.gpt_analysis,
                                    active_task_type,
                                    actual_perf
                                )
                            
                            # Detailed metrics
                            st.subheader("Detailed Performance Metrics")
                            
                            if active_task_type == 'classification':
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Accuracy", f"{report['accuracy']:.3f}")
                                with col2:
                                    st.metric("Precision", f"{report['precision']:.3f}")
                                with col3:
                                    st.metric("Recall", f"{report['recall']:.3f}")
                                with col4:
                                    st.metric("F1 Score", f"{report['f1']:.3f}")
                                
                                if report.get('roc_auc'):
                                    st.metric("ROC AUC", f"{report['roc_auc']:.3f}")
                                
                                # Cross-validation results
                                if report.get('cv_mean') is not None:
                                    st.info(f"Cross-Validation: {report['cv_mean']:.3f} ± {report['cv_std']:.3f}")
                                
                                # Visualizations
                                if report.get('probabilities') is not None:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # ROC Curve
                                        class_names = getattr(preprocessor.label_encoder, 'classes_', None) if preprocessor.label_encoder else None
                                        roc_fig = plot_roc_curve(report['actuals'], report['probabilities'], class_names)
                                        if roc_fig:
                                            st.pyplot(roc_fig)
                                    with col2:
                                        # Confusion Matrix
                                        if report.get('confusion_matrix') is not None:
                                            fig, ax = plt.subplots(figsize=(6, 5))
                                            sns.heatmap(report['confusion_matrix'], annot=True, fmt='d', 
                                                       cmap='Blues', ax=ax)
                                            ax.set_title('Confusion Matrix')
                                            ax.set_xlabel('Predicted')
                                            ax.set_ylabel('Actual')
                                            st.pyplot(fig)
                            
                            else:  # Regression
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("R² Score", f"{report['r2']:.3f}")
                                with col2:
                                    st.metric("RMSE", f"{report['rmse']:.3f}")
                                with col3:
                                    st.metric("MAE", f"{report['mae']:.3f}")
                                with col4:
                                    st.metric("Explained Variance", f"{report['explained_variance']:.3f}")
                                
                                # Cross-validation results
                                if report.get('cv_mean') is not None:
                                    st.info(f"Cross-Validation R²: {report['cv_mean']:.3f} ± {report['cv_std']:.3f}")
                                
                                # Visualizations
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(report['actuals'], report['predictions'], alpha=0.5)
                                ax.plot([report['actuals'].min(), report['actuals'].max()],
                                       [report['actuals'].min(), report['actuals'].max()], 
                                       'r--', label='Perfect Prediction')
                                ax.set_xlabel('Actual Values')
                                ax.set_ylabel('Predicted Values')
                                ax.set_title('Actual vs Predicted Values')
                                ax.legend()
                                st.pyplot(fig)
                            
                            # Best parameters if tuning was performed
                            if automl_model.best_params:
                                st.subheader("Best Hyperparameters")
                                st.json(automl_model.best_params)
                            
                            # Model downloads
                            st.subheader("Export Results")
                            get_model_downloads(model, active_task_type, 
                                               f"{model_choice}_{target_col}", 
                                               model_choice, preprocessor.preprocessor)
                            
                            # Data downloads
                            col1, col2 = st.columns(2)
                            with col1:
                                train_data = df.loc[train_indices]
                                train_csv = train_data.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Training Data",
                                    data=train_csv,
                                    file_name='train_data.csv',
                                    mime='text/csv'
                                )
                            
                            with col2:
                                test_data = df.loc[test_indices]
                                test_csv = test_data.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Test Data",
                                    data=test_csv,
                                    file_name='test_data.csv',
                                    mime='text/csv'
                                )
                            
                        except Exception as e:
                            st.error(f"Error in pipeline execution: {str(e)}")
        
        # ==================== CLUSTERING ====================
        elif task_type == "Clustering":
            st.subheader("Clustering Configuration")
            
            # Automatic Preprocessing
            proc_res = get_processed_data(df, None, enable_feature_selection)
            X, _, detected_task_type, indices, preprocessor = \
                proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
            
            st.markdown("""
                <div class='info-banner'>
                    <strong>Data Ready:</strong> Preprocessing complete and optimized.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven Discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('clustering')
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Analyze Spatial Distribution", key="analyze_spatial"):
                    st.session_state.clustering_insights = cached_analyze_dataset_stats('clustering', X)
                    if not st.session_state.clustering_insights:
                        st.session_state.clustering_insights = ["Standard spatial distribution detected. Density-based or Centroid-based clustering should perform well."]
            with col2:
                st.info("Pro Tip: Run K-Means as baseline, then try DBSCAN for irregular shapes.")
            
            if st.session_state.get('clustering_insights'):
                with st.expander("Dataset Topology Insights", expanded=True):
                    for insight in st.session_state.clustering_insights:
                        st.write(f"• {insight}")
            
            # GPT Suggestions
            if st.session_state.gpt_analysis:
                gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get('clustering', [])
                if gpt_suggestions:
                    with st.expander("AI Recommended Algorithms", expanded=True):
                        for suggestion in gpt_suggestions[:3]:
                            st.write(f"**{suggestion['algorithm']}** - Estimated Silhouette: {suggestion.get('estimated_silhouette', 'N/A')}")
            
            # Model selection
            clustering_models = ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
            model_choice = st.selectbox("Select Clustering Algorithm", clustering_models, key="clustering_model")
            
            # Parameters based on algorithm
            col1, col2 = st.columns(2)
            with col1:
                if model_choice in ["K-Means", "Hierarchical", "Gaussian Mixture"]:
                    n_clusters = st.slider("Number of clusters", 2, 20, 3, key="n_clusters")
                else:
                    n_clusters = None
                
            with col2:
                if model_choice == "DBSCAN":
                    eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1, key="eps")
                    min_samples = st.slider("Minimum samples", 1, 20, 5, key="min_samples")
            
            # Run button
            if st.button("Run Clustering Pipeline", key="run_clustering"):
                st.session_state.run_clustering_clicked = True
            
            # Pipeline execution
            if st.session_state.get("run_clustering_clicked", False):
                with st.spinner("Running clustering pipeline..."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Prepare parameters
                        status_text.text("Step 1/3: Preparing clustering parameters...")
                        progress_bar.progress(20)
                        
                        # Step 2: Run clustering
                        status_text.text("Step 2/3: Performing clustering...")
                        
                        if model_choice == "K-Means":
                            labels, metrics, model = perform_clustering(X, 'K-Means', n_clusters)
                        elif model_choice == "DBSCAN":
                            # Use DBSCAN specific parameters
                            model = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = model.fit_predict(X)
                            
                            # Calculate metrics
                            metrics = {}
                            unique_labels = np.unique(labels[labels != -1])
                            if len(unique_labels) > 1:
                                try:
                                    metrics['silhouette_score'] = silhouette_score(X, labels)
                                except:
                                    metrics['silhouette_score'] = None
                        elif model_choice == "Hierarchical":
                            labels, metrics, model = perform_clustering(X, 'Hierarchical', n_clusters)
                        elif model_choice == "Gaussian Mixture":
                            labels, metrics, model = perform_clustering(X, 'Gaussian Mixture', n_clusters)
                        
                        progress_bar.progress(60)
                        
                        # Step 3: Visualize results
                        status_text.text("Step 3/3: Visualizing results...")
                        
                        # Display results
                        st.success(f"Clustering completed successfully!")
                        
                        # Performance metrics
                        st.header("Clustering Results")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        if 'silhouette_score' in metrics and metrics['silhouette_score'] is not None:
                            with col1:
                                st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                        if 'calinski_harabasz_score' in metrics and metrics['calinski_harabasz_score'] is not None:
                            with col2:
                                st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.1f}")
                        if 'davies_bouldin_score' in metrics and metrics['davies_bouldin_score'] is not None:
                            with col3:
                                st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
                        
                        # Cluster distribution
                        cluster_counts = pd.Series(labels).value_counts().sort_index()
                        st.subheader("Cluster Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        cluster_counts.plot(kind='bar', ax=ax)
                        ax.set_xlabel('Cluster')
                        ax.set_ylabel('Number of Points')
                        ax.set_title('Cluster Size Distribution')
                        for i, v in enumerate(cluster_counts):
                            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
                        st.pyplot(fig)
                        
                        # Visualization
                        st.subheader("Cluster Visualization")
                        viz_fig = plot_cluster_results(X, labels, model_choice)
                        st.pyplot(viz_fig)
                        
                        # Add cluster labels to original data
                        df_with_clusters = df.copy()
                        df_with_clusters['Cluster'] = labels
                        
                        # Download results
                        st.subheader("Export Results")
                        cluster_csv = df_with_clusters.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data with Cluster Labels",
                            data=cluster_csv,
                            file_name='data_with_clusters.csv',
                            mime='text/csv'
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Clustering pipeline completed!")
                        
                    except Exception as e:
                        st.error(f"Error in clustering pipeline: {str(e)}")
        
        # ==================== DIMENSIONALITY REDUCTION ====================
        elif task_type == "Dimensionality Reduction":
            st.subheader("Dimensionality Reduction Configuration")
            
            # Automatic Preprocessing
            proc_res = get_processed_data(df, None, enable_feature_selection)
            X, _, detected_task_type, indices, preprocessor = \
                proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
            
            st.markdown("""
                <div class='info-banner'>
                    <strong>Data Ready:</strong> Preprocessing complete for dimensionality reduction.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Original Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('dimensionality_reduction')
            
            if st.button("Analyze Variance Structure", key="analyze_variance"):
                st.session_state.dr_insights = cached_analyze_dataset_stats('dimensionality_reduction', X)
                if not st.session_state.dr_insights:
                    st.session_state.dr_insights = ["Uniform variance structure detected. Linear reduction (PCA) is recommended as a baseline."]
                
            if st.session_state.get('dr_insights'):
                with st.expander("Dimensionality Insights", expanded=True):
                    for insight in st.session_state.dr_insights:
                        st.write(f"• {insight}")
            
            # GPT Suggestions
            if st.session_state.gpt_analysis:
                gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get('dimensionality_reduction', [])
                if gpt_suggestions:
                    with st.expander("AI Recommendations (Calibrated)", expanded=True):
                        for suggestion in gpt_suggestions[:3]:
                            st.write(f"**{suggestion['algorithm']}** - Estimated Variance: {suggestion.get('estimated_variance', 'N/A')}")
            
            # Model selection
            dr_models = ["PCA", "t-SNE", "UMAP", "ICA", "NMF"]
            model_choice = st.selectbox("Select Dimensionality Reduction Algorithm", dr_models, key="dr_model")
            
            # Parameters
            col1, col2 = st.columns(2)
            with col1:
                n_components = st.slider("Number of components", 2, min(50, X.shape[1]), 2, key="n_components")
            
            with col2:
                if model_choice == "t-SNE":
                    perplexity = st.slider("Perplexity", 5, 50, 30, key="perplexity")
            
            # Run button
            if st.button("Run Dimensionality Reduction Pipeline", key="run_dr"):
                st.session_state.run_dr_clicked = True
            
            # Pipeline execution
            if st.session_state.get("run_dr_clicked", False):
                with st.spinner("Running dimensionality reduction pipeline..."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Prepare parameters
                        status_text.text("Step 1/3: Preparing parameters...")
                        progress_bar.progress(20)
                        
                        # Step 2: Run dimensionality reduction
                        status_text.text("Step 2/3: Performing dimensionality reduction...")
                        
                        if model_choice == "PCA":
                            X_reduced, metrics, model = perform_dimensionality_reduction(X, 'PCA', n_components)
                        elif model_choice == "t-SNE":
                            model = TSNE(n_components=n_components, random_state=DEFAULT_RANDOM_STATE, perplexity=perplexity)
                            X_reduced = model.fit_transform(X)
                            metrics = {}
                        elif model_choice == "UMAP":
                            X_reduced, metrics, model = perform_dimensionality_reduction(X, 'UMAP', n_components)
                        elif model_choice == "ICA":
                            X_reduced, metrics, model = perform_dimensionality_reduction(X, 'ICA', n_components)
                        elif model_choice == "NMF":
                            X_reduced, metrics, model = perform_dimensionality_reduction(X, 'NMF', n_components)
                        
                        progress_bar.progress(60)
                        
                        # Step 3: Visualize results
                        status_text.text("Step 3/3: Visualizing results...")
                        
                        # Display results
                        st.success(f"Dimensionality reduction completed successfully!")
                        
                        # Performance metrics
                        st.header("Dimensionality Reduction Results")
                        
                        st.write(f"**Reduced Data Shape:** {X_reduced.shape[0]} samples, {X_reduced.shape[1]} components")
                        
                        # Display metrics
                        if 'total_variance_explained' in metrics:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Variance Explained", f"{metrics['total_variance_explained']:.1%}")
                            if 'explained_variance_ratio' in metrics:
                                with col2:
                                    st.metric("Variance per Component", ", ".join([f"{v:.1%}" for v in metrics['explained_variance_ratio']]))
                        
                        # Visualization
                        st.subheader("Reduced Data Visualization")
                        
                        if n_components >= 2:
                            viz_fig = plot_dimensionality_reduction(X_reduced, model_choice)
                            if viz_fig:
                                st.pyplot(viz_fig)
                            
                            # Scatter plot matrix if we have 3 or more components
                            if n_components >= 3:
                                st.subheader("3D Scatter Plot")
                                fig_3d = plt.figure(figsize=(10, 8))
                                ax = fig_3d.add_subplot(111, projection='3d')
                                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.5, s=10)
                                ax.set_xlabel('Component 1')
                                ax.set_ylabel('Component 2')
                                ax.set_zlabel('Component 3')
                                ax.set_title('3D Scatter Plot of Reduced Data')
                                st.pyplot(fig_3d)
                        
                        # Variance explained plot for PCA
                        if model_choice == "PCA" and hasattr(model, 'explained_variance_ratio_'):
                            st.subheader("Cumulative Variance Explained")
                            cumulative_variance = np.cumsum(model.explained_variance_ratio_)
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
                            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% variance')
                            ax.set_xlabel('Number of Components')
                            ax.set_ylabel('Cumulative Explained Variance')
                            ax.set_title('Cumulative Explained Variance')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        # Download reduced data
                        st.subheader("Export Results")
                        reduced_df = pd.DataFrame(X_reduced, columns=[f'Component_{i+1}' for i in range(X_reduced.shape[1])])
                        reduced_csv = reduced_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Reduced Data",
                            data=reduced_csv,
                            file_name='reduced_data.csv',
                            mime='text/csv'
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Dimensionality reduction pipeline completed!")
                        
                    except Exception as e:
                        st.error(f"Error in dimensionality reduction pipeline: {str(e)}")
        
        # ==================== ANOMALY DETECTION ====================
        elif task_type == "Anomaly Detection":
            st.subheader("Anomaly Detection Configuration")
            
            # Automatic Preprocessing
            proc_res = get_processed_data(df, None, enable_feature_selection)
            X, _, detected_task_type, indices, preprocessor = \
                proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
            
            st.markdown("""
                <div class='info-banner'>
                    <strong>Data Ready:</strong> Preprocessing complete and optimized.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven Discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('anomaly_detection')
            
            if st.button("Analyze Outlier Potential", key="analyze_outliers"):
                st.session_state.anomaly_insights = cached_analyze_dataset_stats('anomaly_detection', X)
                if not st.session_state.anomaly_insights:
                    st.session_state.anomaly_insights = ["No extreme statistical anomalies detected at first glance. Proceeding with detailed algorithmic detection."]
                
            if st.session_state.get('anomaly_insights'):
                with st.expander("Anomaly Insights", expanded=True):
                    for insight in st.session_state.anomaly_insights:
                        st.write(f"• {insight}")
            
            # AI Suggestions
            if st.session_state.gpt_analysis:
                gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get('anomaly_detection', [])
                if gpt_suggestions:
                    with st.expander("AI Recommendations (Calibrated)", expanded=True):
                        for suggestion in gpt_suggestions[:3]:
                            st.write(f"**{suggestion['algorithm']}** - Estimated Precision: {suggestion.get('estimated_precision', 'N/A')}")
            
            # Model selection
            anomaly_models = ["Isolation Forest", "Local Outlier Factor"]
            model_choice = st.selectbox("Select Anomaly Detection Algorithm", anomaly_models, key="anomaly_model")
            
            # Parameters
            contamination = st.slider("Expected anomaly proportion", 0.01, 0.5, 0.1, 0.01, 
                                     help="Expected proportion of anomalies in the data", key="contamination")
            
            # Run button
            if st.button("Run Anomaly Detection Pipeline", key="run_anomaly"):
                st.session_state.run_anomaly_clicked = True
            
            # Pipeline execution
            if st.session_state.get("run_anomaly_clicked", False):
                with st.spinner("Running anomaly detection pipeline..."):
                    try:
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Step 1: Prepare parameters
                        status_text.text("Step 1/3: Preparing anomaly detection parameters...")
                        progress_bar.progress(20)
                        
                        # Step 2: Run anomaly detection
                        status_text.text("Step 2/3: Detecting anomalies...")
                        
                        if model_choice == "Isolation Forest":
                            anomaly_labels, anomaly_count, model = perform_anomaly_detection(X, 'Isolation Forest', contamination)
                        elif model_choice == "Local Outlier Factor":
                            anomaly_labels, anomaly_count, model = perform_anomaly_detection(X, 'Local Outlier Factor', contamination)
                        
                        progress_bar.progress(60)
                        
                        # Step 3: Visualize results
                        status_text.text("Step 3/3: Visualizing results...")
                        
                        # Display results
                        st.success(f"Anomaly detection completed successfully!")
                        
                        # Performance metrics
                        st.header("Anomaly Detection Results")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", X.shape[0])
                        with col2:
                            st.metric("Anomalies Detected", anomaly_count)
                        with col3:
                            anomaly_percentage = (anomaly_count / X.shape[0]) * 100
                            st.metric("Anomaly Percentage", f"{anomaly_percentage:.1f}%")
                        
                        # Visualization
                        st.subheader("Anomaly Visualization")
                        viz_fig = plot_anomaly_detection(X, anomaly_labels, model_choice)
                        st.pyplot(viz_fig)
                        
                        # Anomaly details
                        st.subheader("Anomaly Details")
                        
                        # Add anomaly labels to original data
                        df_with_anomalies = df.copy()
                        df_with_anomalies['Is_Anomaly'] = anomaly_labels
                        df_with_anomalies['Anomaly_Score'] = 0  # Placeholder for scores
                        
                        # Show anomalies table
                        anomalies_df = df_with_anomalies[df_with_anomalies['Is_Anomaly'] == 1]
                        if not anomalies_df.empty:
                            st.write(f"**Found {len(anomalies_df)} anomalies:**")
                            st.dataframe(anomalies_df.head(20), height=400, use_container_width=True)
                            
                            # Summary statistics for anomalies vs normal
                            st.subheader("Statistics: Anomalies vs Normal")
                            
                            # Select numerical columns for comparison
                            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if numerical_cols:
                                col1, col2 = st.columns(2)
                                
                                normal_stats = df_with_anomalies[df_with_anomalies['Is_Anomaly'] == 0][numerical_cols].describe().T
                                anomaly_stats = df_with_anomalies[df_with_anomalies['Is_Anomaly'] == 1][numerical_cols].describe().T
                                
                                with col1:
                                    st.write("**Normal Data Statistics**")
                                    st.dataframe(normal_stats[['mean', 'std', 'min', 'max']].head(10))
                                
                                with col2:
                                    st.write("**Anomaly Data Statistics**")
                                    st.dataframe(anomaly_stats[['mean', 'std', 'min', 'max']].head(10))
                        
                        # Download results
                        st.subheader("Export Results")
                        anomaly_csv = df_with_anomalies.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data with Anomaly Labels",
                            data=anomaly_csv,
                            file_name='data_with_anomalies.csv',
                            mime='text/csv'
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("Anomaly detection pipeline completed!")
                        
                    except Exception as e:
                        st.error(f"Error in anomaly detection pipeline: {str(e)}")

if __name__ == '__main__':
    main()