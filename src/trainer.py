import sys
from pathlib import Path

# add root so config is discoverable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from typing import List, Tuple, Optional

import config.mlp_config as mlp_config
import config.dp_config as dp_config
from model_tester import ModelTester

class FederatedTrainer:
    # full configuration guide
    # https://sklearn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    # === architecture parameters ===
    # hidden_layers: 
    # tuple, defines number of neurons in each hidden layer
    # e.g., (64, 32) for two hidden layers with 64 and 32 neurons respectively
    # more layers can capture more complex patterns but can also risk overfitting
    
    # activation: 
    # defines activation function for the hidden layers
    # options: 'identity', 'logistic', 'tanh', 'relu' (default relu)
    # ===============================
    
    # === optimization parameters ===
    # solver:
    # defines the algo used to optimize weights
    # options: 'adam', 'sgd', 'lbfgs' (default adam)
    
    # learning_rate_init:
    # initial learning rate for weight updates
    # float (usually 0.0001 < x < 0.01), default 0.001
    
    # learning_rate:
    # defines the step size for weight updates
    # options: 'constant', 'invscaling', 'adaptive'
    # ===============================
    
    # === regularization parameters ===
    # alpha:
    # L2 penalty (regularization term) to prevent overfitting
    # float (usually 0.00001 < x < 0.01), default 0.0001
    
    # batch_size:
    # number of samples per gradient update
    # integer, default 'auto' (min(200, n_samples))
    # ================================
    
    # === training parameters ===
    # max_iter:
    # maximum number of training iterations
    # integer, default 200
    
    # early_stopping:
    # whether to stop training when validation score is not improving
    # boolean, default False
    
    # validation_fraction:
    # proportion of training data to set aside for validation when early stopping is used
    # float (0.0 < x < 1.0), default 0.1
    
    # n_iter_no_change:
    # number of iterations with no improvement to wait before stopping
    # integer, default 10
    
    # tol:
    # tolerance for optimization, training stops when loss does not improve by at least tol
    # float, default 0.0001
    # ============================
    
    # random_state:
    # random seed for reproducibility, each seed gives different weight initialization
    # use same seed to reproduce results
    
    # initialize multi layer perceptron model with configuration
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
        
    # load data from CSV file
    def load_data(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_file)
        
        # separate features and target (last col is target)
        features = df.iloc[:, :-1].values
        target = df.iloc[:, -1].values

        return features, target

    # train the model and return weights
    def train(self, csv_file: str) -> List[np.ndarray]:
        # backward compatible single-call train (no initial weights)
        return self.train_from(csv_file)

    def train_from(self, csv_file: str, initial_weights: Optional[List[np.ndarray]] = None, epochs: int = 1) -> List[np.ndarray]:
        """
        Train the model on the given CSV. If `initial_weights` is provided (interleaved list: W1,b1,W2,b2,...)
        the model will continue training from those weights using `partial_fit`. `epochs` controls how many
        partial_fit passes are made over the local data (default 1 per round).
        Returns the trained weights in the interleaved format.
        """
        features, target = self.load_data(csv_file)

        print(f"Training on {len(features)} samples... (epochs={epochs})")

        classes = np.unique(target)

        # prepare label binarizer required by sklearn partial_fit
        lb = LabelBinarizer()
        lb.fit(classes)
        # ensure classes_ and _label_binarizer are available for partial_fit
        self.model.classes_ = classes
        self.model._label_binarizer = lb

        # if initial weights provided, set model weights to them
        if initial_weights is not None:
            # convert interleaved list to coefs_ and intercepts_
            coefs = [initial_weights[i] for i in range(0, len(initial_weights), 2)]
            intercepts = [initial_weights[i] for i in range(1, len(initial_weights), 2)]

            # ensure copies so we don't mutate caller arrays
            self.model.coefs_ = [c.copy() for c in coefs]
            self.model.intercepts_ = [b.copy() for b in intercepts]

            # set minimal attributes required by sklearn downstream code
            self.model.n_layers_ = len(self.model.coefs_) + 1
            self.model.n_outputs_ = self.model.intercepts_[-1].shape[0]
            self.model.out_activation_ = 'softmax' if self.model.n_outputs_ > 1 else 'logistic'

        # perform `epochs` rounds of partial_fit over local data
        for e in range(epochs):
            # partial_fit updates weights in-place and preserves previous weights
            self.model.partial_fit(features, target, classes=classes)

        print(f"Training completed. Score (train): {self.model.score(features, target):.4f}\n")

        # extract weights and biases
        weights: List[np.ndarray] = []
        for coef, intercept in zip(self.model.coefs_, self.model.intercepts_):
            weights.append(coef.copy())
            weights.append(intercept.copy())

        return weights
    
    # extract weights and biases from the trained model
    def get_weights(self) -> List[np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        weights: List[np.ndarray] = []
        print("Extracting model weights and biases...")
        for coef, intercept in zip(self.model.coefs_, self.model.intercepts_):
            weights.append(coef.copy())
            weights.append(intercept.copy())

        return weights

    # compute weight updates without DP noise
    def compute_updates(self, weights: List[np.ndarray], original_weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute weight updates as (weights - original_weights).
        Returns the raw updates without any noise or clipping.
        """
        return [w - w_old for w, w_old in zip(weights, original_weights)]

def test_model(output_file: str) -> None:
    # initialize tester
    tester = ModelTester()
    
    # load weights
    tester.load_weights(output_file)
    
    # evaluate model and print results
    tester.evaluate_model("data/split/test_data.csv")
    

def main():
    # check command line arguments
    if len(sys.argv) < 2:
        print("Error: CSV file path is required")
        print("Example: python trainer.py <path_to_csv>")
        sys.exit(1)
    
    # get CSV file 
    csv_file = sys.argv[1]
    print(f"\nTraining federated learning model on: {csv_file}")
    
    # initialize trainer and train model
    trainer = FederatedTrainer()
    weights = trainer.train(csv_file)
    
    print("\nModel weights generated:")
    print(f"Total number of weight arrays: {len(weights)}")
    for i, w in enumerate(weights):
        weight_type = "Weights" if i % 2 == 0 else "Biases"
        print(f"  Layer {i//2 + 1} {weight_type}: shape {w.shape}")
    
    # save weights to numpy format
    csv_path = Path(csv_file)
    hospital_name = csv_path.stem
    output_file = f"weights/{hospital_name}_weights.npz"
    np.savez(output_file, *weights)
    print(f"\nWeights saved to: {output_file}")
    
    # evaluate trained model
    print("\nEvaluating trained model on test data...")
    test_model(output_file)

if __name__ == "__main__":
    main()