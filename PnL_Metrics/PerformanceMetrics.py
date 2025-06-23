import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from scipy.stats import pearsonr, spearmanr


class ModelMetrics:
    def __init__(self, y_true, y_pred, y_prob=None, rank_preds=None):
        """
        Initialize the model‐level metrics calculator.

        Parameters:
        - y_true: pd.DataFrame or pd.Series containing true signals (1, 0, -1) OR true continuous returns if regression
        - y_pred: pd.DataFrame or pd.Series containing predicted signals (1, 0, -1) OR predicted continuous returns if regression
        - y_prob: (Optional) pd.Series or array‐like of predicted probabilities of the positive class if regression model used
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob) if y_prob is not None else None

    # Classification Metrics

    def accuracy(self):
        """
        Calculate accuracy aka hit rate.
        """
        return metrics.accuracy_score(self.y_true, self.y_pred)

    def precision(self, average='weighted'):
        """
        Calculate precision.

        Parameters:
        - average: 'binary' (0/1), 'macro'/'micro' for multi‐class, 'weighted' for imbalanced signal predictions
        """
        return metrics.precision_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )

    def recall(self, average='weighted'):
        """
        Calculate recall.

        Parameters:
        - average: 'binary' (0/1), 'macro'/'micro' for multi‐class, 'weighted' for imbalanced signal predictions
        """
        return metrics.recall_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )

    def f1_score(self, average='weighted'):
        """
        Calculate F1‐score.

        Parameters:
        - average: 'binary' (0/1), 'macro'/'micro' for multi‐class, 'weighted' for imbalanced signal predictions
        """
        return metrics.f1_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )

    # Probabilistic Model Metrics

    def roc_auc(self):
        """
        Calculate ROC AUC. Requires y_prob.
        """
        if self.y_prob is None:
            return None
        unique_labels = np.unique(self.y_true)
        if set(unique_labels).issubset({0,1}):
            return metrics.roc_auc_score(self.y_true, self.y_prob)
        else:
            return None

    def log_loss(self):
        """
        Return log‐loss aka entropy. Requires y_prob.
        """
        if self.y_prob is None:
            return None
        try:
            return metrics.log_loss(self.y_true, self.y_prob)
        except Exception:
            return None

    def brier_score(self):
        """
        Return Brier score. Requires y-prob.
        """
        if self.y_prob is None:
            return None
        try:
            return metrics.brier_score_loss(self.y_true, self.y_prob)
        except Exception:
            return None

    # Regression and Directional Metrics

    def mse(self):
        """
        Return Mean Squared Error.
        """
        return metrics.mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):
        """
        Return Root Mean Squared Error.
        """
        return np.sqrt(self.mse())

    def mae(self):
        """
        Return Mean Absolute Error.
        """
        return metrics.mean_absolute_error(self.y_true, self.y_pred)

    def r2(self):
        """
        Return R squared.
        """
        return metrics.r2_score(self.y_true, self.y_pred)

    def directional_accuracy(self):
        """
        Return directional accuracy.
        """
        # Define sign(0) = 0
        y_true_sign = np.sign(self.y_true)
        y_pred_sign = np.sign(self.y_pred)

        nonzero_idx = (y_true_sign != 0)
        if nonzero_idx.sum() == 0:
            return np.nan

        return np.mean(y_true_sign[nonzero_idx] == y_pred_sign[nonzero_idx])
    
    def summary(self, type='classification'):
        """
        Generate key metrics based on signal model type.

        Parameters:
        - type: 'classification' or 'regression'.
        """
        if type == 'classification':
            all_metrics = {
                'accuracy': self.accuracy(),
                'precision': self.precision(),
                'recall': self.recall(),
                'f1_score': self.f1_score(),
                'roc_auc': self.roc_auc(),
                'log_loss': self.log_loss(),
                'brier_score': self.brier_score(),
            }

        elif type == 'regression':
            all_metrics = {
                'mse': self.mse(),
                'rmse': self.rmse(),
                'mae': self.mae(),
                'r2': self.r2(),
                'directional_accuracy': self.directional_accuracy(),
            }

        else:
            raise ValueError("problem_type must be 'classification' or 'regression'.")
        
        summary_df = pd.DataFrame(all_metrics, index=['PerformanceMetrics'])
        return summary_df

