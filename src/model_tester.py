import sys
from pathlib import Path

# add root so config is discoverable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer
from typing import Tuple

import config.mlp_config as mlp_config

class ModelTester:
    
    # initialize model with training configuration
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=mlp_config.HIDDEN_LAYERS,
            activation=mlp_config.ACTIVATION,
            solver=mlp_config.SOLVER,
            learning_rate_init=mlp_config.LEARNING_RATE_INIT,
            learning_rate=mlp_config.LEARNING_RATE,
            alpha=mlp_config.ALPHA,
            batch_size=mlp_config.BATCH_SIZE,
            max_iter=mlp_config.MAX_ITER,
            early_stopping=mlp_config.EARLY_STOPPING,
            validation_fraction=mlp_config.VALIDATION_FRACTION,
            n_iter_no_change=mlp_config.N_ITER_NO_CHANGE,
            tol=mlp_config.TOL,
            random_state=mlp_config.RANDOM_STATE,
            verbose=False
        )
    
    # load weights into model
    def load_weights(self, weights_file: str, num_classes: int = None) -> None:        
        print(f"Loading weights from: {weights_file}")
        
        # load weights from file
        data = np.load(weights_file)
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        
        # set model weights and biases
        self.model.coefs_ = [weights[i] for i in range(0, len(weights), 2)]
        self.model.intercepts_ = [weights[i] for i in range(1, len(weights), 2)]
        
        # set other necessary attributes for the model to work
        
        # model with n hidden layers has n+1 layers total
        self.model.n_layers_ = len(self.model.coefs_) + 1 
        
        # number of neurons in output layer
        # in this case, we are predicting binary outcome (readmitted or not)
        # scikit learn uses 1 neuron for binary classification, >1 for multi-class
        self.model.n_outputs_ = self.model.intercepts_[-1].shape[0] 

        # set output activation based on number of classes
        # binary classification uses 'logistic', multi-class uses 'softmax'
        # in this case, predicting whether patient is readmitted or not (binary)
        self.model.out_activation_ = 'softmax' if self.model.n_outputs_ > 1 else 'logistic'
        
        print(f"Weights loaded successfully!")
        print(f"Model has {self.model.n_layers_} layers")
        print(f"Output layer has {self.model.n_outputs_} neurons\n")
    
    # load test data from CSV test file
    def load_data(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"Loading test data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # separate features and target (last col is target)
        features = df.iloc[:, :-1].values
        target = df.iloc[:, -1].values
        
        # set model classes based on unique labels in target column
        unique_classes = np.unique(target)
        self.model.classes_ = unique_classes
        self.model._label_binarizer = LabelBinarizer()
        self.model._label_binarizer.fit(self.model.classes_)
        
        print(f"Loaded {len(features)} test samples with {features.shape[1]} features")
        print(f"Detected classes in test data: {unique_classes}\n")
        
        return features, target
    
    # evaluate model on test data and compute metrics
    def evaluate_model(self, test_csv: str) -> dict:
        # load test data
        test_features, test_target = self.load_data(test_csv)
        
        # make predictions
        print("Making predictions on test data...")
        target_prediction = self.model.predict(test_features)

        # 2) get probabilities for positive class (needed for AUC)
        # for binary clf, predict_proba gives shape (n_samples, 2)
        try:
            y_proba = self.model.predict_proba(test_features)[:, 1]
        except AttributeError:
            # if for some reason predict_proba is not available
            y_proba = None

        # do a threshold sweep
        best = {}
        for t in [0.25, 0.3, 0.35, 0.4, 0.5]:
            y_pred_t = (y_proba >= t).astype(int)
            accuracy_t = accuracy_score(test_target, y_pred_t)
            f1_t = f1_score(test_target, y_pred_t, zero_division=0)
            rec_t = recall_score(test_target, y_pred_t, zero_division=0)
            prec_t = precision_score(test_target, y_pred_t, zero_division=0)
            best[t] = (accuracy_t, prec_t, rec_t, f1_t)

        print("\nThreshold sweep:")
        for t, (a, p, r, f) in best.items():
            print(f"t={t:.2f} → Accuracy={a:.3f} Precision={p:.3f} Recall={r:.3f} F1 Score={f:.3f}")
        
        # determine if binary or multiclass classification
        unique_classes = np.unique(test_target)
        is_binary = len(unique_classes) == 2
        average_method = 'binary' if is_binary else 'weighted'
        
        # calculate individual metrics
        accuracy = accuracy_score(test_target, target_prediction)
        precision = precision_score(test_target, target_prediction, average=average_method, zero_division=0)
        recall = recall_score(test_target, target_prediction, average=average_method, zero_division=0)
        f1 = f1_score(test_target, target_prediction, average=average_method, zero_division=0)

        roc_auc = None
        pr_auc = None
        if is_binary and y_proba is not None:
            roc_auc = roc_auc_score(test_target, y_proba)
            pr_auc = average_precision_score(test_target, y_proba)

        cm = confusion_matrix(test_target, target_prediction)

        # print metrics
        print("\nEvaluation Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")
        print(f"ROC AUC:    {roc_auc:.4f}")
        print(f"PR AUC:     {pr_auc:.4f}")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(test_target, target_prediction, zero_division=0))
        
        # store metrics in dictionary
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,    
            "pr_auc": pr_auc, 
            "num_samples": len(test_target),
            "num_classes": len(unique_classes),
        }
        
        return metrics


def main():
    # check command line arguments
    if len(sys.argv) < 2:
        print("Error: Weights file required")
        print("Usage: python inference.py <weights_file>")
        sys.exit(1)
    
    # get file paths
    weights_file = sys.argv[1]
    test_csv = "data/split/test_data.csv"
    
    # initialize inference object
    print("Initializing model tester...")
    tester = ModelTester()
    
    # load model weights
    tester.load_weights(weights_file)
    
    # evaluate model on test data
    tester.evaluate_model(test_csv)


if __name__ == "__main__":
    main()
