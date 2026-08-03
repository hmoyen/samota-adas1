"""
RBF (Radial Basis Function) Surrogate Model

Uses scipy's Rbf for interpolation based on training data.
"""

import numpy as np
from scipy.interpolate import Rbf as ScipyRbf
from sklearn.preprocessing import StandardScaler


class Model:
    """RBF surrogate model."""

    def __init__(self, n_neurons=10, train_data=None, normalize=False):
        """
        Initialize RBF model.

        Args:
            n_neurons: Number of neurons (RBF kernels)
            train_data: List of (X, y) tuples for training
            normalize: Standardize input features before fitting. The RBF kernel
                uses a single global epsilon over all dimensions, so on raw
                (unnormalized) inputs the fit is dominated by whichever variable
                has the largest numeric range.
        """
        self.n_neurons = n_neurons
        self.rbf_model = None
        self.X_train = None
        self.y_train = None
        self.normalize = normalize
        self.X_scaler = StandardScaler() if normalize else None

        if train_data is not None:
            self.train(train_data)

    def train(self, train_data):
        """
        Train RBF model.

        Args:
            train_data: List of (X, y) tuples
        """
        if len(train_data) == 0:
            raise ValueError("No training data provided")

        X_list, y_list = zip(*train_data)
        self.X_train = np.array(X_list)
        self.y_train = np.array(y_list)

        # Prepare data for RBF
        # RBF expects: rbf(*coordinates, z)
        X_fit = self.X_scaler.fit_transform(self.X_train) if self.normalize else self.X_train
        X_T = X_fit.T  # Shape: (n_features, n_samples)

        # Use multiquadric RBF
        try:
            self.rbf_model = ScipyRbf(*X_T, self.y_train, function='multiquadric', epsilon=1.0)
        except Exception as e:
            # Fallback to thin-plate if multiquadric fails (e.g. singular matrix
            # from collinear/duplicate points)
            print(f"RBF multiquadric training failed ({type(e).__name__}: {e}), "
                  f"falling back to thin_plate")
            self.rbf_model = ScipyRbf(*X_T, self.y_train, function='thin_plate')

    def predict(self, X):
        """
        Predict value for input X.

        Args:
            X: Input vector (1D array of parameters)

        Returns:
            Predicted value
        """
        if self.rbf_model is None:
            raise RuntimeError("Model not trained yet")

        # Ensure X is a numpy array
        X = np.array(X).flatten()
        if self.normalize:
            X = self.X_scaler.transform(X.reshape(1, -1)).flatten()

        # RBF expects individual coordinates as separate arguments
        try:
            pred = self.rbf_model(*X)
        except Exception as e:
            print(f"RBF prediction error: {e}")
            # Return mean of training data as fallback
            return float(np.mean(self.y_train))

        return float(pred)

    def predict_batch(self, X_batch):
        """
        Predict values for batch of inputs.
        
        Args:
            X_batch: Array of inputs (shape: n_samples × n_features)
            
        Returns:
            Array of predictions
        """
        return np.array([self.predict(X) for X in X_batch])

    def get_model_error(self):
        """Get training error (MAE)."""
        if self.X_train is None:
            return float('inf')

        y_pred = self.predict_batch(self.X_train)
        mae = np.mean(np.abs(y_pred - self.y_train))
        return mae
