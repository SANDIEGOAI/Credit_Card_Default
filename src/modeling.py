"""
modeling.py
Reusable functions for model training, evaluation, and selection.
"""
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score, precision_recall_curve
import numpy as np

def train_model(model, X_train, y_train):
    """Fit a scikit-learn model to the training data."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return metrics as a dictionary."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics


def evaluate_model_with_threshold(model, X_test, y_test, model_name, default_threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]

    # Use precision-recall curve to find optimal threshold for recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    # Find threshold that maximizes recall while keeping precision reasonable
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    optimal_idx = np.argmax(f1_scores)  # Maximize F1-score as a balance
    optimal_threshold = thresholds[optimal_idx]
    print(f"\n{model_name} Optimal Threshold: {optimal_threshold:.3f}")

    # Evaluate with default threshold (0.5)
    y_pred_default = (y_prob >= default_threshold).astype(int)
    print(f"\n{model_name} Performance (Default Threshold = {default_threshold}):")
    print('Accuracy:', np.ceil(accuracy_score(y_test, y_pred_default) * 100) / 100)
    print('Precision:', np.ceil(precision_score(y_test, y_pred_default) * 100) / 100)
    print('Recall:', np.ceil(recall_score(y_test, y_pred_default) * 100) / 100)
    print('F1-Score:', np.ceil(f1_score(y_test, y_pred_default) * 100) / 100)
    print('AUC-ROC:', np.ceil(roc_auc_score(y_test, y_prob) * 100) / 100)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_default))

    # Evaluate with optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    print(
        f"\n{model_name} Performance (Optimal Threshold = {optimal_threshold:.2f}):")
    print('Accuracy:', np.ceil(accuracy_score(y_test, y_pred_optimal) * 100) / 100)
    print('Precision:', np.ceil(precision_score(y_test, y_pred_optimal) * 100) / 100)
    print('Recall:', np.ceil(recall_score(y_test, y_pred_optimal) * 100) / 100)
    print('F1-Score:', np.ceil(f1_score(y_test, y_pred_optimal) * 100) / 100)
    print('AUC-ROC:', np.ceil(roc_auc_score(y_test, y_prob) * 100) / 100)
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_optimal))


def cross_validate_model(model, X, y, cv=5, scoring='roc_auc'):
    """Perform cross-validation and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores


def grid_search(model, param_grid, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1):
    """Perform grid search for hyperparameter tuning."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_
