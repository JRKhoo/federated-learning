# === architecture parameters ===
# hidden_layers: 
# tuple, defines number of neurons in each hidden layer
# e.g., (64, 32) for two hidden layers with 64 and 32 neurons respectively
# more layers can capture more complex patterns but can also risk overfitting
HIDDEN_LAYERS = (64, 32)

# activation: 
# defines activation function for the hidden layers
# options: 'identity', 'logistic', 'tanh', 'relu' (default relu)
ACTIVATION = 'relu'
# ===============================

# === optimization parameters ===
# solver:
# defines the algo used to optimize weights
# options: 'adam', 'sgd', 'lbfgs' (default adam)
SOLVER = 'adam'

# learning_rate_init:
# initial learning rate for weight updates
# float (usually 0.0001 < x < 0.01), default 0.001
LEARNING_RATE_INIT = 0.0007

# learning_rate:
# defines the step size for weight updates
# options: 'constant', 'invscaling', 'adaptive'
LEARNING_RATE = 'adaptive'
# ===============================

# === regularization parameters ===
# alpha:
# L2 penalty (regularization term) to prevent overfitting
# float (usually 0.00001 < x < 0.01), default 0.0001
ALPHA = 0.001

# batch_size:
# number of samples per gradient update
# integer, default 'auto' (min(200, n_samples))
BATCH_SIZE = 'auto'
# ================================

# === training parameters ===
# max_iter:
# maximum number of training iterations
# integer, default 200
MAX_ITER = 500

# early_stopping:
# whether to stop training when validation score is not improving
# boolean, default False
EARLY_STOPPING = True

# validation_fraction:
# proportion of training data to set aside for validation when early stopping is used
# float (0.0 < x < 1.0), default 0.1
VALIDATION_FRACTION = 0.15

# n_iter_no_change:
# number of iterations with no improvement to wait before stopping
# only used when solver='adam' or 'sgd' and early_stopping=True
# integer, default 10
N_ITER_NO_CHANGE = 20

# tol:
# tolerance for optimization, training stops when loss does not improve by at least tol
# float, default 0.0001
TOL = 0.00005
# ============================

# random_state:
# random seed for reproducibility, each seed gives different weight initialization
# keep constant to reproduce results
RANDOM_STATE = 67