import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    IsolationForest
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDClassifier, SGDRegressor,
    PassiveAggressiveClassifier, PassiveAggressiveRegressor
)
from sklearn.svm import SVC, SVR, OneClassSVM
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor,
    LocalOutlierFactor
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, mean_squared_error,
    r2_score, explained_variance_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    roc_curve, auc, roc_auc_score, confusion_matrix,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, OPTICS, Birch,
    MeanShift, AffinityPropagation, MiniBatchKMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, NMF, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_regression, VarianceThreshold, RFE
from sklearn.cross_decomposition import CCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import VotingClassifier, VotingRegressor
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
from time import sleep, time
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
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
def apply_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@500;700&display=swap');
        
        :root {
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
            --accent-primary: #38bdf8;
            --accent-secondary: #818cf8;
            --text-main: #f1f5f9;
        }

        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.9) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid var(--glass-border);
        }

        .stButton>button {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);
            width: 100%;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(56, 189, 248, 0.4);
            color: white;
        }

        .metric-card {
            background: var(--glass-bg);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }

        .status-badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
        }
        
        div[data-testid="stExpander"] {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
        }

        .stDataFrame {
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ==================== FIXED: CoTrainer Class ====================
class CoTrainer(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator1=None, estimator2=None, max_iter=100, random_state=None):
        self.estimator1 = estimator1 or LogisticRegression(max_iter=1000)
        self.estimator2 = estimator2 or RandomForestClassifier(random_state=42)
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

# ==================== FIXED: Enhanced Data Preprocessor ====================
@st.cache_data(show_spinner="Loading file...")
def load_large_file(uploaded_file):
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext in ['csv', 'txt']:
            return pd.read_csv(uploaded_file)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(uploaded_file)
        elif ext == 'parquet':
            return pd.read_parquet(uploaded_file)
        elif ext == 'feather':
            return pd.read_feather(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
    return None

class EnhancedUniversalDataPreprocessor:
    def __init__(self, use_dense=True, enable_feature_selection=True):
        """
        Enhanced UniversalDataPreprocessor with proper scaling and feature selection.
        """
        self.label_encoder = None
        self.preprocessor = None
        self.feature_selector = None
        self.feature_names = None
        self.use_dense = use_dense
        self.enable_feature_selection = enable_feature_selection
        self._fitted = False
        self.scaler = StandardScaler()
        self.variance_threshold = VarianceThreshold(threshold=0.01)

    def detect_task_type(self, y=None):
        if y is None:
            return 'clustering'
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            return 'regression'
        return 'classification'

    def _validate_data(self, df):
        """Clean and validate data"""
        df_clean = df.copy()
        
        # Remove constant columns
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]
        if constant_cols:
            st.info(f"Removing constant columns: {constant_cols}")
            df_clean = df_clean.drop(columns=constant_cols)
        
        # Handle high cardinality categorical features
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        high_cardinality = [col for col in categorical_cols if df_clean[col].nunique() > 50]
        if high_cardinality:
            st.info(f"Removing high cardinality features: {high_cardinality}")
            df_clean = df_clean.drop(columns=high_cardinality)
        
        return df_clean

    def process(self, df, target_col=None):
        # Clean the data first
        df_clean = self._validate_data(df)
        
        if target_col is None:
            X = df_clean.copy()
            y = None
            task_type = 'clustering'
        else:
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]
            
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]

            task_type = self.detect_task_type(y)

            if task_type == 'classification':
                self.label_encoder = LabelEncoder()
                # Ensure labels are 0-indexed for models like XGBoost/LightGBM
                y_transformed = self.label_encoder.fit_transform(y)
                y = pd.Series(y_transformed, index=y.index, name=y.name)
        
        final_indices = X.index

        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Proper StandardScaler pipeline
        num_transform = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Use dense output for better compatibility
        cat_transform = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=not self.use_dense))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', num_transform, numeric_features),
            ('cat', cat_transform, categorical_features)
        ], remainder='drop')

        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Apply variance threshold for feature selection
        original_shape = X_processed.shape[1]
        if self.enable_feature_selection and original_shape > 10:
            try:
                self.variance_threshold.fit(X_processed)
                X_processed = self.variance_threshold.transform(X_processed)
                new_shape = X_processed.shape[1]
                if new_shape < original_shape:
                    st.info(f"Feature selection: Reduced from {original_shape} to {new_shape} features")
            except Exception as e:
                st.warning(f"Feature selection skipped: {str(e)}")
        
        # Get feature names
        try:
            cat_names = self.preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        except (AttributeError, KeyError):
            cat_names = []

        # Get feature names after all transformations
        self.feature_names = list(numeric_features) + list(cat_names)
        
        # Only filter feature names if VarianceThreshold was actually fitted and used
        try:
            check_is_fitted(self.variance_threshold)
            if self.variance_threshold.get_support().size > 0:
                if len(self.variance_threshold.get_support()) == len(self.feature_names):
                    self.feature_names = np.array(self.feature_names)[self.variance_threshold.get_support()].tolist()
        except NotFittedError:
            # If not fitted, it means it was skipped or failed during fit
            pass
        
        self._fitted = True

        return X_processed, y, task_type, final_indices

# ==================== CACHED PREPROCESSING ENGINE ====================
@st.cache_data(show_spinner="Performing intelligent preprocessing...")
def get_processed_data(df, target_col=None, enable_feature_selection=True):
    """
    Cached preprocessing function to ensure data is prepared automatically and efficiently.
    """
    preprocessor = EnhancedUniversalDataPreprocessor(
        use_dense=True,
        enable_feature_selection=enable_feature_selection
    )
    # df.copy() to ensure original data is not modified and for stable hashing
    X, y, task_type, indices = preprocessor.process(df.copy(), target_col)
    
    return {
        'X': X,
        'y': y,
        'task_type': task_type,
        'indices': indices,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.feature_names
    }

# ==================== FIXED: Enhanced AutoML Model ====================
class EnhancedAutoMLModel:
    def __init__(self, task_type, model_choice, n_clusters=None, encoding_dim=None, 
                 handle_imbalance=False, enable_tuning=True):
        self.task_type = task_type
        self.model_choice = model_choice
        self.n_clusters = n_clusters
        self.encoding_dim = encoding_dim
        self.handle_imbalance = handle_imbalance
        self.enable_tuning = enable_tuning
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
        if self.task_type == 'classification':
            class_weight_param = {'class_weight': 'balanced'} if self.handle_imbalance else {}

            if self.model_choice == "Logistic Regression":
                return LogisticRegression(max_iter=2000, random_state=42, **class_weight_param)
            elif self.model_choice == "Decision Tree":
                return DecisionTreeClassifier(random_state=42, **class_weight_param)
            elif self.model_choice == "Random Forest":
                return RandomForestClassifier(random_state=42, **class_weight_param)
            elif self.model_choice == "Gradient Boosting":
                return GradientBoostingClassifier(random_state=42)
            elif self.model_choice == "XGBoost":
                return XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
            elif self.model_choice == "LightGBM":
                return LGBMClassifier(random_state=42)
            elif self.model_choice == "CatBoost":
                return CatBoostClassifier(silent=True, random_state=42)
            elif self.model_choice == "SVM":
                return SVC(probability=True, random_state=42, **class_weight_param)
            elif self.model_choice == "k-NN":
                return KNeighborsClassifier()
            elif self.model_choice == "Naive Bayes":
                return GaussianNB()
            elif self.model_choice == "MLP":
                return MLPClassifier(max_iter=1000, random_state=42)
            elif self.model_choice == "Auto-Ensemble":
                clf1 = RandomForestClassifier(n_estimators=100, random_state=42, **class_weight_param)
                clf2 = XGBClassifier(eval_metric='logloss', random_state=42)
                clf3 = LogisticRegression(max_iter=2000, random_state=42, **class_weight_param)
                return VotingClassifier(estimators=[('rf', clf1), ('xgb', clf2), ('lr', clf3)], voting='soft')
            elif self.model_choice == "Co-Training":
                return CoTrainer(random_state=42)
        
        elif self.task_type == 'regression':
            if self.model_choice == "Linear Regression":
                return LinearRegression()
            elif self.model_choice == "Random Forest":
                return RandomForestRegressor(random_state=42)
            elif self.model_choice == "XGBoost":
                return XGBRegressor(random_state=42)
            elif self.model_choice == "Gradient Boosting":
                return GradientBoostingRegressor(random_state=42)
            elif self.model_choice == "LightGBM":
                return LGBMRegressor(random_state=42)
            elif self.model_choice == "Auto-Ensemble":
                reg1 = RandomForestRegressor(n_estimators=100, random_state=42)
                reg2 = XGBRegressor(random_state=42)
                reg3 = Ridge(alpha=1.0)
                return VotingRegressor(estimators=[('rf', reg1), ('xgb', reg2), ('ridge', reg3)])
            elif self.model_choice == "SVR":
                return SVR()
            elif self.model_choice == "Ridge Regression":
                return Ridge()
        
        elif self.task_type == 'clustering':
            if self.model_choice == "K-Means":
                return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            elif self.model_choice == "DBSCAN":
                return DBSCAN()
            elif self.model_choice == "Hierarchical":
                return AgglomerativeClustering(n_clusters=self.n_clusters)
            elif self.model_choice == "Spectral":
                return SpectralClustering(n_clusters=self.n_clusters, random_state=42)
            elif self.model_choice == "Gaussian Mixture":
                return GaussianMixture(n_components=self.n_clusters, random_state=42)
        
        elif self.task_type == 'dimensionality_reduction':
            if self.model_choice == "PCA":
                return PCA(n_components=self.encoding_dim, random_state=42)
            elif self.model_choice == "t-SNE":
                return TSNE(n_components=self.encoding_dim, random_state=42)
            elif self.model_choice == "UMAP":
                return UMAP(n_components=self.encoding_dim, random_state=42)
            elif self.model_choice == "ICA":
                return FastICA(n_components=self.encoding_dim, random_state=42)
            elif self.model_choice == "NMF":
                return NMF(n_components=self.encoding_dim, random_state=42)
        
        elif self.task_type == 'anomaly_detection':
            if self.model_choice == "Isolation Forest":
                return IsolationForest(random_state=42)
            elif self.model_choice == "One-Class SVM":
                return OneClassSVM()
            elif self.model_choice == "Local Outlier Factor":
                return LocalOutlierFactor()
            elif self.model_choice == "Elliptic Envelope":
                return EllipticEnvelope(random_state=42)
        
        return None
    
    def fit_with_tuning(self, X, y, cv_folds=5):
        """Fit model with hyperparameter tuning and cross-validation"""
        start_time = time()
        
        if not self.enable_tuning or X.shape[0] < 100:
            # Use base model for small datasets
            self.model = self._init_base_model()
            
            if self.task_type in ['classification', 'regression', 'anomaly_detection']:
                self.model.fit(X, y)
            elif self.task_type in ['clustering', 'dimensionality_reduction']:
                self.model.fit(X)
            
            # Compute cross-validation scores for supervised tasks
            if self.task_type in ['classification', 'regression']:
                cv = StratifiedKFold(n_splits=min(3, X.shape[0]), shuffle=True, random_state=42) \
                    if self.task_type == 'classification' else \
                    KFold(n_splits=min(3, X.shape[0]), shuffle=True, random_state=42)
                
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                self.cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            # Perform hyperparameter tuning
            base_model = self._init_base_model()
            param_grid = self._get_hyperparameter_grid()
            
            if param_grid and self.task_type in ['classification', 'regression']:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) \
                    if self.task_type == 'classification' else \
                    KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
                
                search = RandomizedSearchCV(
                    base_model, param_grid,
                    n_iter=5,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                search.fit(X, y)
                
                self.model = search.best_estimator_
                self.best_params = search.best_params_
                self.cv_scores = search.cv_results_['mean_test_score']
            elif self.task_type == 'clustering' and param_grid:
                # For clustering, we need to evaluate different number of clusters
                best_score = -1
                best_model = None
                best_params = {}
                
                for n_clusters in param_grid.get('n_clusters', [2, 3, 4, 5]):
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
        
        training_time = time() - start_time
        return training_time
    
    def get_model(self):
        """Get the trained model"""
        return self.model

# ==================== FIXED: Enhanced GPT Dataset Analyzer ====================
class EnhancedGPTDatasetAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize GPT dataset analyzer with calibration"""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.calibrator = PerformanceCalibrator()
        if not self.api_key:
            st.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it in the app.")
            self.client = None
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.api_version = "new"
        except ImportError:
            try:
                import openai
                openai.api_key = self.api_key
                self.client = openai
                self.api_version = "old"
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def analyze_dataset(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Analyze dataset using GPT and return calibrated insights"""
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        try:
            # Create dataset summary
            dataset_summary = self._create_dataset_summary(df, target_col)
            
            # Create prompt for GPT with calibration guidance
            prompt = self._create_calibrated_analysis_prompt(df, dataset_summary, target_col)
            
            # Call GPT API
            if self.api_version == "new":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data science expert providing realistic, calibrated analysis of datasets. Provide conservative estimates based on typical real-world performance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                gpt_analysis = response.choices[0].message.content
            else:
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data science expert providing realistic, calibrated analysis of datasets. Provide conservative estimates based on typical real-world performance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                gpt_analysis = response.choices[0].message.content
            
            # Parse GPT response and calibrate estimates
            analysis_result = self._parse_and_calibrate_gpt_response(gpt_analysis, dataset_summary)
            analysis_result["gpt_raw_response"] = gpt_analysis
            
            return analysis_result
            
        except Exception as e:
            return {"error": f"GPT analysis failed: {str(e)}"}
    
    def _create_dataset_summary(self, df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Create comprehensive dataset summary"""
        dataset_summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict(),
            "numerical_columns": list(df.select_dtypes(include=np.number).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "basic_stats": {},
            "correlation_info": {}
        }
        
        # Add basic statistics for numerical columns
        if dataset_summary["numerical_columns"]:
            dataset_summary["basic_stats"] = df[dataset_summary["numerical_columns"]].describe().to_dict()
            
            # Calculate correlations if there are at least 2 numerical columns
            if len(dataset_summary["numerical_columns"]) >= 2:
                try:
                    corr_matrix = df[dataset_summary["numerical_columns"]].corr()
                    # Get top 5 correlations
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                "features": [corr_matrix.columns[i], corr_matrix.columns[j]],
                                "correlation": float(corr_matrix.iloc[i, j])
                            })
                    
                    # Sort by absolute correlation
                    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                    dataset_summary["correlation_info"]["top_correlations"] = corr_pairs[:5]
                except:
                    pass
        
        # Add target column analysis if provided
        if target_col and target_col in df.columns:
            dataset_summary["target_analysis"] = {
                "name": target_col,
                "dtype": str(df[target_col].dtype),
                "unique_values": int(df[target_col].nunique()),
                "missing_count": int(df[target_col].isnull().sum()),
                "value_counts": df[target_col].value_counts().to_dict()
            }
            
            # Detect task type
            if pd.api.types.is_numeric_dtype(df[target_col]):
                if df[target_col].nunique() > 10:
                    dataset_summary["task_type"] = "regression"
                else:
                    dataset_summary["task_type"] = "classification"
            else:
                dataset_summary["task_type"] = "classification"
        
        return dataset_summary
    
    def _create_calibrated_analysis_prompt(self, df: pd.DataFrame, summary: Dict, target_col: str = None) -> str:
        """Create prompt for GPT with emphasis on realistic estimates"""
        prompt = f"""
        Analyze this dataset and provide REALISTIC, CONSERVATIVE insights including estimated accuracy for different algorithms.
        IMPORTANT: Provide estimates that reflect TYPICAL real-world performance, not optimistic best-case scenarios.
        
        DATASET SUMMARY:
        - Shape: {summary['shape']} (rows x columns)
        - Total Features: {len(summary['columns'])}
        - Numerical Features: {len(summary['numerical_columns'])}
        - Categorical Features: {len(summary['categorical_columns'])}
        - Missing Values Total: {sum(summary['missing_values'].values())}
        """
        
        # Add data quality warnings
        issues = []
        if df.shape[0] < 100:
            issues.append("Very small dataset")
        if df.shape[1] > df.shape[0]:
            issues.append("High dimensionality (more features than samples)")
        if sum(summary['missing_values'].values()) / (df.shape[0] * df.shape[1]) > 0.1:
            issues.append("High missing value ratio (>10%)")
        
        if issues:
            prompt += "\nDATA QUALITY CONCERNS:\n"
            for issue in issues:
                prompt += f"- {issue}\n"
        
        if 'correlation_info' in summary and 'top_correlations' in summary['correlation_info']:
            prompt += "\nTOP CORRELATIONS:\n"
            for corr in summary['correlation_info']['top_correlations'][:3]:
                prompt += f"- {corr['features'][0]} & {corr['features'][1]}: {corr['correlation']:.3f}\n"
        
        if target_col and 'target_analysis' in summary:
            target_info = summary['target_analysis']
            prompt += f"""
            
            TARGET COLUMN ANALYSIS:
            - Name: {target_info['name']}
            - Data Type: {target_info['dtype']}
            - Unique Values: {target_info['unique_values']}
            - Missing Values: {target_info['missing_count']}
            - Task Type: {summary.get('task_type', 'Unknown')}
            """
            
            # Add realism factors
            if summary.get('task_type') == 'classification':
                total = sum(target_info['value_counts'].values())
                if total > 0:
                    max_class = max(target_info['value_counts'].values())
                    min_class = min(target_info['value_counts'].values())
                    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                    prompt += f"\n- Class Imbalance Ratio: {imbalance_ratio:.2f}"
                    if imbalance_ratio > 3:
                        prompt += " (Significant imbalance - REDUCE accuracy estimates by 10-15%)"
        
        prompt += """
        
        Please provide REALISTIC analysis in the following JSON structure:
        {
            "dataset_quality": {
                "score": 0-100,
                "issues": ["list of issues"],
                "strengths": ["list of strengths"]
            },
            "algorithm_recommendations": {
                "classification": [
                    {"algorithm": "Auto-Ensemble", "estimated_accuracy": "80-85%", "reason": "Combines RF, XGBoost, and Logistic Regression for maximum robustness"},
                    {"algorithm": "XGBoost", "estimated_accuracy": "75-80%", "reason": "Excellent for tabular data with complex patterns"},
                    {"algorithm": "Random Forest", "estimated_accuracy": "70-75%", "reason": "Explanation why this algorithm is suitable with realistic expectations"},
                    {"algorithm": "Logistic Regression", "estimated_accuracy": "65-70%", "reason": "Explanation"}
                ],
                "regression": [
                    {"algorithm": "Auto-Ensemble", "estimated_r2": "0.75-0.85", "reason": "Ensemble of RF, XGBoost, and Ridge for superior predictive power"},
                    {"algorithm": "XGBoost", "estimated_r2": "0.70-0.80", "reason": "High performance on non-linear regression tasks"},
                    {"algorithm": "Random Forest", "estimated_r2": "0.60-0.70", "reason": "Explanation"},
                    {"algorithm": "Linear Regression", "estimated_r2": "0.55-0.65", "reason": "Explanation"}
                ],
                "clustering": [
                    {"algorithm": "Algorithm Name", "estimated_silhouette": "0.3-0.5", "reason": "Explanation"},
                    {"algorithm": "Another Algorithm", "estimated_silhouette": "0.25-0.45", "reason": "Explanation"}
                ],
                "dimensionality_reduction": [
                    {"algorithm": "Algorithm Name", "estimated_variance": "65-80%", "reason": "Explanation"},
                    {"algorithm": "Another Algorithm", "estimated_variance": "60-75%", "reason": "Explanation"}
                ],
                "anomaly_detection": [
                    {"algorithm": "Algorithm Name", "estimated_precision": "75-85%", "reason": "Explanation"},
                    {"algorithm": "Another Algorithm", "estimated_precision": "70-80%", "reason": "Explanation"}
                ]
            },
            "data_preprocessing_recommendations": ["list of preprocessing steps"],
            "insights": ["list of key insights"],
            "warnings": ["list of warnings"],
            "next_steps": ["list of recommended next steps"],
            "estimated_training_time": "Estimate of training time (be realistic)",
            "realism_factors": ["list of factors that might reduce performance"]
        }
        
        IMPORTANT GUIDELINES FOR ESTIMATES:
        1. For classification: Start with 60-70% for baseline, 70-80% for good models, 80-90% only for exceptional cases
        2. For regression: R² of 0.5-0.7 is typical, 0.7-0.8 is good, >0.8 is excellent
        3. Account for dataset size: Small datasets (<1000 samples) reduce estimates by 10-20%
        4. Account for data quality issues: Missing values, noise reduce estimates
        5. Be CONSERVATIVE - real-world performance is usually lower than theoretical
        
        Also provide a brief summary in markdown format at the end.
        """
        
        return prompt
    
    def _parse_and_calibrate_gpt_response(self, response: str, dataset_summary: Dict) -> Dict[str, Any]:
        """Parse GPT response and calibrate estimates"""
        try:
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
            else:
                parsed_data = self._create_fallback_response(response)
            
            # Calibrate the estimates
            for task_type in ['classification', 'regression', 'clustering', 
                            'dimensionality_reduction', 'anomaly_detection']:
                if task_type in parsed_data.get('algorithm_recommendations', {}):
                    for algo in parsed_data['algorithm_recommendations'][task_type]:
                        # Extract and calibrate
                        if task_type == 'classification':
                            est_str = algo.get('estimated_accuracy', '60%')
                            calibrated = self.calibrator.calibrate_estimate(est_str, task_type)
                            # Format back to percentage string
                            low = int((calibrated - 0.05) * 100)
                            high = int((calibrated + 0.05) * 100)
                            algo['estimated_accuracy'] = f"{max(50, low)}-{min(95, high)}%"
                            algo['calibrated_numeric'] = calibrated
                        
                        elif task_type == 'regression':
                            est_str = algo.get('estimated_r2', '0.5')
                            calibrated = self.calibrator.calibrate_estimate(est_str, task_type)
                            low = calibrated - 0.1
                            high = calibrated + 0.1
                            algo['estimated_r2'] = f"{max(0.1, low):.2f}-{min(0.95, high):.2f}"
                            algo['calibrated_numeric'] = calibrated
            
            # Add calibration info
            parsed_data['calibration_info'] = {
                'calibration_factors': self.calibrator.calibration_factors,
                'calibration_samples': len(self.calibrator.history)
            }
            
            # Extract markdown summary
            summary_match = re.search(r'## Summary.*?(?=\n##|\Z)', response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                parsed_data["summary_markdown"] = summary_match.group(0)
            else:
                parsed_data["summary_markdown"] = "## Summary\n" + response.split("}")[-1].strip()
            
            return parsed_data
            
        except Exception as e:
            st.warning(f"Error parsing GPT response: {str(e)}")
            return self._create_fallback_response(response)
    
    def _create_fallback_response(self, response: str) -> Dict[str, Any]:
        """Create conservative fallback response"""
        return {
            "dataset_quality": {"score": 50, "issues": [], "strengths": []},
            "algorithm_recommendations": {
                "classification": [
                    {"algorithm": "Random Forest", "estimated_accuracy": "65-75%", "reason": "Good for most classification tasks"},
                    {"algorithm": "XGBoost", "estimated_accuracy": "70-80%", "reason": "Excellent for complex patterns"},
                    {"algorithm": "Logistic Regression", "estimated_accuracy": "60-70%", "reason": "Good baseline model"}
                ],
                "regression": [
                    {"algorithm": "Random Forest", "estimated_r2": "0.60-0.75", "reason": "Robust for regression problems"},
                    {"algorithm": "XGBoost", "estimated_r2": "0.65-0.80", "reason": "High accuracy with complex data"},
                    {"algorithm": "Linear Regression", "estimated_r2": "0.50-0.65", "reason": "Simple baseline model"}
                ],
                "clustering": [
                    {"algorithm": "K-Means", "estimated_silhouette": "0.3-0.5", "reason": "Good for spherical clusters"},
                    {"algorithm": "DBSCAN", "estimated_silhouette": "0.2-0.4", "reason": "Finds arbitrary shaped clusters"}
                ],
                "dimensionality_reduction": [
                    {"algorithm": "PCA", "estimated_variance": "60-75%", "reason": "Standard linear reduction"},
                    {"algorithm": "UMAP", "estimated_variance": "65-80%", "reason": "Good for visualization"}
                ],
                "anomaly_detection": [
                    {"algorithm": "Isolation Forest", "estimated_precision": "75-85%", "reason": "Effective for high dimensions"},
                    {"algorithm": "Local Outlier Factor", "estimated_precision": "70-80%", "reason": "Good for varying densities"}
                ]
            },
            "data_preprocessing_recommendations": ["Handle missing values", "Scale numerical features", "Encode categorical variables"],
            "insights": ["Analysis provided in summary"],
            "warnings": ["Conservative estimates used - actual performance may vary"],
            "next_steps": ["Proceed with model training"],
            "estimated_training_time": "2-5 minutes",
            "realism_factors": ["Using conservative estimates"],
            "summary_markdown": f"## Analysis Summary\n{response}"
        }

# ==================== HELPER FUNCTIONS FOR DIFFERENT TASK TYPES ====================

def perform_clustering(X, algorithm='K-Means', n_clusters=3):
    """Perform clustering on the dataset"""
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'Spectral':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
    elif algorithm == 'Gaussian Mixture':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
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
        model = PCA(n_components=n_components, random_state=42)
    elif algorithm == 't-SNE':
        model = TSNE(n_components=n_components, random_state=42, perplexity=30)
    elif algorithm == 'UMAP':
        model = UMAP(n_components=n_components, random_state=42)
    elif algorithm == 'ICA':
        model = FastICA(n_components=n_components, random_state=42)
    elif algorithm == 'NMF':
        model = NMF(n_components=n_components, random_state=42)
    else:
        model = PCA(n_components=n_components, random_state=42)
    
    X_reduced = model.fit_transform(X)
    
    metrics = {}
    if hasattr(model, 'explained_variance_ratio_'):
        metrics['explained_variance_ratio'] = model.explained_variance_ratio_
        metrics['total_variance_explained'] = np.sum(model.explained_variance_ratio_)
    
    return X_reduced, metrics, model

def perform_anomaly_detection(X, algorithm='Isolation Forest', contamination=0.1):
    """Perform anomaly detection on the dataset"""
    if algorithm == 'Isolation Forest':
        model = IsolationForest(contamination=contamination, random_state=42)
    elif algorithm == 'One-Class SVM':
        model = OneClassSVM()
    elif algorithm == 'Local Outlier Factor':
        model = LocalOutlierFactor(contamination=contamination, novelty=True)
    elif algorithm == 'Elliptic Envelope':
        model = EllipticEnvelope(contamination=contamination, random_state=42)
    else:
        model = IsolationForest(contamination=contamination, random_state=42)
    
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
        reducer = PCA(n_components=2, random_state=42)
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
        reducer = PCA(n_components=2, random_state=42)
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
                        np.random.seed(42)
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
            if isinstance(model, LinearDiscriminantAnalysis):
                if y_test is None:
                    raise ValueError("LDA for dimensionality reduction requires target labels (y).")
                transformed = model.fit_transform(X_test, y_test)
            else:
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
    st.subheader("Intelligent Dataset Report")
    
    if "error" in analysis_result:
        st.error(f"GPT Analysis Error: {analysis_result['error']}")
        return
    
    # Dataset Quality Score
    if "dataset_quality" in analysis_result:
        quality = analysis_result["dataset_quality"]
        score = quality.get("score", 50)
        
        # Professional Alignment for Quality Section
        st.markdown(f"""
        <div style="background: var(--glass-bg); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--glass-border); margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin: 0; color: #94a3b8; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.05em;">Dataset Reliability</h3>
                    <div style="font-size: 3rem; font-weight: 800; color: #fff;">{score}<span style="font-size: 1.2rem; color: #64748b; font-weight: 400;">/100</span></div>
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
            ax.tick_params(colors='#94a3b8', labelsize=8)
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
        st.markdown(f"**Projected Compute Time:** {analysis_result['estimated_training_time']}")
    
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
        stats_insights = []
        if X is None: return stats_insights

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
            return ["Dataset too small for reliable benchmarking pilot (less than 10 rows)."]

        results = []
        # Sample data for performance (max 2000 rows)
        n_samples = X.shape[0]
        sample_size = min(2000, n_samples)
        
        # Use simple slicing for speed and stability
        X_s = X[:sample_size] if isinstance(X, np.ndarray) else X.iloc[:sample_size]
        y_s = y[:sample_size] if isinstance(y, np.ndarray) else y.iloc[:sample_size]

        with st.spinner(f"Running data-driven pilot benchmark on {sample_size} samples..."):
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
                return ["Benchmarking not available for this task type."]

            # Use 3-fold or simple 2-fold if data is very small
            cv_val = 3 if sample_size >= 30 else 2
            
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_s, y_s, cv=cv_val, scoring=scoring)
                    results.append({'model': name, 'score': scores.mean()})
                except:
                    continue

        if not results:
            return ["Benchmarking failed: Check data format."]

        # Sort and return recommendations (provide top 3)
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

def get_model_downloads(model, task_type, model_name, model_choice, preprocessor=None):
    """Generate download buttons for the trained model in various formats"""
    st.subheader("Download Trained Model")
    
    col1, col2 = st.columns(2)
    
    base_key = f"{model_name}_{task_type}"
    
    # Pickle format
    model_bytes = pickle.dumps(model)
    with col1:
        st.download_button(
            label="Download as Pickle",
            data=model_bytes,
            file_name=f'{model_name}.pkl',
            mime='application/octet-stream',
            key=f"{base_key}_pkl"
        )
        st.caption("Standard Python serialization format")
    
    # Joblib format (better for large models)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    with col2:
        st.download_button(
            label="Download as Joblib",
            data=buffer.getvalue(),
            file_name=f'{model_name}.joblib',
            mime='application/octet-stream',
            key=f"{base_key}_joblib"
        )
        st.caption("Better for large numpy arrays")

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
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="font-size: 4rem; font-weight: 800; margin-bottom: 0.5rem; background: linear-gradient(90deg, #fff 30%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            VertexML
        </h1>
        <p style="font-size: 1.4rem; color: #94a3b8; max-width: 100%; margin: 0 auto; line-height: 1.6; white-space: nowrap;">
            A end to end model training platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state initialization
    keys = {
        'performance_calibrator': PerformanceCalibrator(),
        'actual_vs_estimated': [],
        'gpt_analysis': None,
        'data_insights': None,
        'pilot_results': None,
        'clustering_insights': None,
        'dr_insights': None,
        'anomaly_insights': None,
        'run_supervised_clicked': False,
        'run_clustering_clicked': False,
        'run_dr_clicked': False,
        'run_anomaly_clicked': False
    }
    for key, value in keys.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Sidebar configuration
    st.sidebar.title("VertexML")
    st.sidebar.markdown("---")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        st.sidebar.success("AI Engine Connected")
    else:
        st.sidebar.warning("AI Engine Offline (Set OPENAI_API_KEY)")
    
    # Enhanced configuration
    st.sidebar.subheader("Enhanced Settings")
    enable_hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True)
    enable_cross_validation = st.sidebar.checkbox("Enable Cross-Validation", value=True)
    enable_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=True)
    
    # Initialize GPT Analyzer
    gpt_analyzer = EnhancedGPTDatasetAnalyzer(openai_api_key)
    
    # Session state management
    def reset_all_runs():
        keys_to_reset = ['run_supervised_clicked', 'run_clustering_clicked', 
                        'run_dr_clicked', 'run_anomaly_clicked',
                        'data_insights', 'pilot_results', 'clustering_insights',
                        'dr_insights', 'anomaly_insights']
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

        # GPT Analysis
        if openai_api_key:
            if st.button("Generate Intelligent Report", key="analyze_gpt"):
                with st.spinner("Analyzing dataset patterns and structures..."):
                    analysis_result = gpt_analyzer.analyze_dataset(df)
                    
                    if "error" not in analysis_result:
                        st.success("Report generated successfully!")
                        st.session_state.gpt_analysis = analysis_result
                    else:
                        st.error(f"GPT Analysis Error: {analysis_result['error']}")
            
            # Persistently display report if it exists
            if st.session_state.gpt_analysis:
                with st.expander("Intelligent Dataset Report", expanded=True):
                    # Determine current task type for calibrated suggestions
                    display_task = None
                    if task_type == "Supervised Learning":
                        # We'll refine this once detected_task_type is known or chosen
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
                # User request: Provide 2 options 1)Classification 2)Regression before suggesting algorithm
                manual_task_mode = st.radio("Select task mode:", ["Classification", "Regression"], 
                                           horizontal=True, key="manual_task_mode")
            
            if target_col:
                try:
                    # Automatic Preprocessing
                    proc_res = get_processed_data(df, target_col, enable_feature_selection)
                    X, y, detected_task_type, indices, preprocessor = \
                        proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
                    
                    # Override detected type with manual choice if they differ
                    active_task_type = manual_task_mode.lower()
                    
                    st.markdown(f"""
                        <div style='background: rgba(56, 189, 248, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;'>
                            <span style='color: #38bdf8; font-weight: 600;'>Preprocessing Complete:</span> 
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Analyze Dataset Patterns", key="analyze_patterns"):
                            st.session_state.data_insights = benchmarker.analyze_dataset_stats(X, y)
                            if not st.session_state.data_insights:
                                st.session_state.data_insights = ["Standard dataset structure detected. No extreme outliers or sparsity issues identified."]
                    with col2:
                        if st.button("Run Pilot Benchmark", key="run_pilot"):
                            st.session_state.pilot_results = benchmarker.run_pilot_benchmark(X, y)
                    
                    # Display Insights
                    if st.session_state.get('data_insights'):
                        with st.expander("Statistical Insights", expanded=True):
                            for insight in st.session_state.data_insights:
                                st.write(f"• {insight}")
                    
                    # Display Pilot Results
                    if st.session_state.get('pilot_results'):
                        with st.expander("Benchmarking Pilot Results", expanded=True):
                            for res in st.session_state.pilot_results:
                                st.write(f"• {res}")
                            st.info("Auto-Ensemble is recommended to combine these strengths.")
                    
                    # AI Recommendations
                    if st.session_state.gpt_analysis:
                        gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get(active_task_type, [])
                        
                        if gpt_suggestions:
                            with st.expander("AI Recommended Algorithms (Calibrated)", expanded=True):
                                for suggestion in gpt_suggestions[:3]:
                                    perf_str = get_performance_string(suggestion, active_task_type)
                                    st.write(f"**{suggestion['algorithm']}** - {perf_str}")
                    
                    # Model selection
                    if active_task_type == 'classification':
                        models = [
                            "Auto-Ensemble", "Random Forest", "XGBoost", "Logistic Regression", 
                            "Gradient Boosting", "LightGBM", "SVM", "k-NN"
                        ]
                        default_idx = 0
                        handle_imbalance = st.checkbox("Handle class imbalance", 
                                                      value=y.value_counts(normalize=True).max() > 0.7)
                    else:
                        models = [
                            "Auto-Ensemble", "Random Forest", "XGBoost", "Linear Regression",
                            "Gradient Boosting", "LightGBM", "Ridge Regression"
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
                    
                    # Run button
                    if st.button("Run Enhanced Pipeline", key="run_enhanced"):
                        st.session_state.run_supervised_clicked = True
                        
                except Exception as e:
                    st.error(f"Error analyzing target: {str(e)}")
                    return
            
                # Pipeline execution
                if st.session_state.run_supervised_clicked:
                    with st.spinner("Optimizing pipeline performance..."):
                        try:
                            # Immediate reset to prevent double-runs on other UI clicks
                            # (but we keep the logic inside this branch for the remainder of this run)
                            
                            # Step 1: Preprocessing (Retrieved from cache)
                            status_text = st.empty()
                            progress_bar = st.progress(0)
                            status_text.text("Step 1/4: Retaining preprocessed data...")
                            proc_res = get_processed_data(df, target_col, enable_feature_selection)
                            X, y, _, indices, preprocessor = \
                                proc_res['X'], proc_res['y'], proc_res['task_type'], proc_res['indices'], proc_res['preprocessor']
                            progress_bar.progress(25)
                            
                            # Step 2: Train-test split
                            status_text.text("Step 2/4: Splitting data...")
                            stratify = y if active_task_type == 'classification' else None
                            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                                X, y, indices, 
                                test_size=test_size_float, 
                                random_state=42,
                                stratify=stratify
                            )
                            progress_bar.progress(35)
                            
                            # Step 3: Model training
                            status_text.text("Step 3/4: Training model with hyperparameter tuning...")
                            automl_model = EnhancedAutoMLModel(
                                task_type=active_task_type,
                                model_choice=model_choice,
                                handle_imbalance=handle_imbalance if active_task_type == 'classification' else False,
                                enable_tuning=enable_hyperparameter_tuning
                            )
                            
                            training_time = automl_model.fit_with_tuning(X_train, y_train, cv_folds=cv_folds)
                            model = automl_model.get_model()
                            progress_bar.progress(75)
                            
                            # Step 4: Evaluation
                            status_text.text("Step 4/4: Evaluating model...")
                            report = enhanced_generate_report(
                                model, X_test, y_test, active_task_type,
                                cv_scores=automl_model.cv_scores
                            )
                            progress_bar.progress(100)
                            status_text.text("Pipeline completed successfully!")
                            
                            # Display results
                            st.success(f"Enhanced pipeline completed in {training_time:.1f} seconds!")
                            
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
            
            st.markdown(f"""
                <div style='background: rgba(56, 189, 248, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;'>
                    <span style='color: #38bdf8; font-weight: 600;'>Data Ready:</span> Preprocessing complete and optimized.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven Discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('clustering')
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Analyze Spatial Distribution", key="analyze_spatial"):
                    st.session_state.clustering_insights = benchmarker.analyze_dataset_stats(X)
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
            clustering_models = ["K-Means", "DBSCAN", "Hierarchical", "Spectral", "Gaussian Mixture"]
            model_choice = st.selectbox("Select Clustering Algorithm", clustering_models, key="clustering_model")
            
            # Parameters based on algorithm
            col1, col2 = st.columns(2)
            with col1:
                if model_choice == "K-Means" or model_choice == "Hierarchical" or model_choice == "Spectral" or model_choice == "Gaussian Mixture":
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
                        elif model_choice == "Spectral":
                            labels, metrics, model = perform_clustering(X, 'Spectral', n_clusters)
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
            
            st.markdown(f"""
                <div style='background: rgba(56, 189, 248, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;'>
                    <span style='color: #38bdf8; font-weight: 600;'>Data Ready:</span> Preprocessing complete for dimensionality reduction.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Original Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('dimensionality_reduction')
            
            if st.button("Analyze Variance Structure", key="analyze_variance"):
                st.session_state.dr_insights = benchmarker.analyze_dataset_stats(X)
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
                    with st.expander("GPT Recommendations (Calibrated)", expanded=True):
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
                            model = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
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
            
            st.markdown(f"""
                <div style='background: rgba(56, 189, 248, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;'>
                    <span style='color: #38bdf8; font-weight: 600;'>Data Ready:</span> Preprocessing complete and optimized.
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"**Data Shape:** {X.shape[0]} samples, {X.shape[1]} features")
            
            # Algorithm selection
            st.subheader("Data-Driven Discovery")
            
            benchmarker = AutomatedAlgorithmBenchmarker('anomaly_detection')
            
            if st.button("Analyze Outlier Potential", key="analyze_outliers"):
                st.session_state.anomaly_insights = benchmarker.analyze_dataset_stats(X)
                if not st.session_state.anomaly_insights:
                    st.session_state.anomaly_insights = ["No extreme statistical anomalies detected at first glance. Proceeding with detailed algorithmic detection."]
                
            if st.session_state.get('anomaly_insights'):
                with st.expander("Anomaly Insights", expanded=True):
                    for insight in st.session_state.anomaly_insights:
                        st.write(f"• {insight}")
            
            # GPT Suggestions
            if st.session_state.gpt_analysis:
                gpt_suggestions = st.session_state.gpt_analysis.get('algorithm_recommendations', {}).get('anomaly_detection', [])
                if gpt_suggestions:
                    with st.expander("GPT Recommendations (Calibrated)", expanded=True):
                        for suggestion in gpt_suggestions[:3]:
                            st.write(f"**{suggestion['algorithm']}** - Estimated Precision: {suggestion.get('estimated_precision', 'N/A')}")
            
            # Model selection
            anomaly_models = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope"]
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
                        elif model_choice == "One-Class SVM":
                            anomaly_labels, anomaly_count, model = perform_anomaly_detection(X, 'One-Class SVM', contamination)
                        elif model_choice == "Local Outlier Factor":
                            anomaly_labels, anomaly_count, model = perform_anomaly_detection(X, 'Local Outlier Factor', contamination)
                        elif model_choice == "Elliptic Envelope":
                            anomaly_labels, anomaly_count, model = perform_anomaly_detection(X, 'Elliptic Envelope', contamination)
                        
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