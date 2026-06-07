"""
SAMOTA Per-Objective Ensemble Surrogate
For global and local search phases, each objective gets its own ensemble.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings('ignore')


class SAMOTAPerObjectiveEnsemble:
    """
    SAMOTA-style ensemble: ONE ensemble per objective (V0-V4).
    Each objective gets Gaussian Process, Polynomial, and RBF network.
    Uses Goel weighting based on training accuracy.
    """

    def __init__(self, X_train, y_train_single_obj, normalize=True, obj_name="V0"):
        """
        Initialize ensemble for ONE objective.

        Args:
            X_train: Parameters (N, 6) - [car_speed, p_x, p_y, orientation, weather, road_shape]
            y_train_single_obj: Violation scores for ONE objective (N,) - just V0, V1, V2, V3, or V4
            normalize: Normalize inputs and outputs
            obj_name: Name of objective (e.g., "V0") for tracking
        """
        self.X_train = X_train
        self.y_train = y_train_single_obj
        self.normalize = normalize
        self.obj_name = obj_name

        # Scalers
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        if normalize:
            X_train_norm = self.X_scaler.fit_transform(X_train)
            y_train_norm = self.y_scaler.fit_transform(y_train_single_obj.reshape(-1, 1)).ravel()
        else:
            X_train_norm = X_train
            y_train_norm = y_train_single_obj

        # Train 3 models
        self.gp = GaussianProcessRegressor(
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=42
        )
        self.gp.fit(X_train_norm, y_train_norm)

        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly_features.fit_transform(X_train_norm)
        self.poly = LinearRegression()
        self.poly.fit(X_poly, y_train_norm)

        self.rbf = KernelRidge(kernel='rbf', gamma=0.1, alpha=1e-6)
        self.rbf.fit(X_train_norm, y_train_norm)

        # Compute training errors for Goel weighting
        gp_pred = self.gp.predict(X_train_norm)
        poly_pred = self.poly.predict(X_poly)
        rbf_pred = self.rbf.predict(X_train_norm)

        e_gp = np.mean(np.abs(gp_pred - y_train_norm))
        e_poly = np.mean(np.abs(poly_pred - y_train_norm))
        e_rbf = np.mean(np.abs(rbf_pred - y_train_norm))

        # Goel formula: weight inversely by error
        e_sum = e_gp + e_poly + e_rbf
        w_gp = (e_sum - e_gp) / (2 * e_sum) if e_sum > 0 else 1/3
        w_poly = (e_sum - e_poly) / (2 * e_sum) if e_sum > 0 else 1/3
        w_rbf = (e_sum - e_rbf) / (2 * e_sum) if e_sum > 0 else 1/3

        # Normalize weights to sum to 1
        w_sum = w_gp + w_poly + w_rbf
        self.w_gp = w_gp / w_sum
        self.w_poly = w_poly / w_sum
        self.w_rbf = w_rbf / w_sum

        # Store errors for reference
        self.e_gp = e_gp
        self.e_poly = e_poly
        self.e_rbf = e_rbf
        self.mae = np.mean([e_gp, e_poly, e_rbf])

    def predict(self, X):
        """
        Predict violations with uncertainty for ONE objective.

        Args:
            X: Parameters (N, 6) or (6,)

        Returns:
            predictions: Predicted violation scores (N,) or scalar
            uncertainties: Model disagreement (N,) or scalar
        """
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # Normalize
        if self.normalize:
            X_norm = self.X_scaler.transform(X)
        else:
            X_norm = X

        # Get predictions from 3 models
        gp_pred = self.gp.predict(X_norm)
        poly_pred = self.poly.predict(self.poly_features.transform(X_norm))
        rbf_pred = self.rbf.predict(X_norm)

        # Stack predictions [N, 3]
        all_preds = np.column_stack([gp_pred, poly_pred, rbf_pred])

        # Weighted ensemble
        ensemble_pred = (
            self.w_gp * gp_pred +
            self.w_poly * poly_pred +
            self.w_rbf * rbf_pred
        )

        # Uncertainty = std dev of model predictions
        uncertainty = np.std(all_preds, axis=1)

        # Denormalize
        if self.normalize:
            ensemble_pred = self.y_scaler.inverse_transform(
                ensemble_pred.reshape(-1, 1)
            ).ravel()
            uncertainty = uncertainty * self.y_scaler.scale_[0]

        if single_sample:
            ensemble_pred = ensemble_pred[0]
            uncertainty = uncertainty[0]

        return ensemble_pred, uncertainty

    def retrain(self, X_new, y_new):
        """
        Retrain with additional data (online learning).

        Args:
            X_new: New parameters (M, 6)
            y_new: New violation scores for this objective (M,)
        """
        # Combine old and new
        X_combined = np.vstack([self.X_train, X_new])
        y_combined = np.concatenate([self.y_train, y_new])

        # Retrain (reinitialize)
        self.__init__(X_combined, y_combined, self.normalize, self.obj_name)


class SAMOTAGlobalSurrogates:
    """
    Collection of per-objective ensembles for GLOBAL SEARCH phase.
    One ensemble per violation objective (V0-V4).

    ENHANCED: Supports training ONLY uncovered objectives for efficiency.
    """

    def __init__(self, X_train, F_train, normalize=True, objective_indices=None):
        """
        Initialize surrogates for specified objectives (or all if None).

        Args:
            X_train: Parameters (N, 6)
            F_train: Violation scores (N, 5) - [V0, V1, V2, V3, V4]
            normalize: Normalize each objective independently
            objective_indices: List of objective indices to train (e.g., [2,3,4])
                             If None, trains all 5 objectives
        """
        self.X_train = X_train
        self.F_train = F_train
        self.normalize = normalize
        self.all_obj_names = ["V0", "V1", "V2", "V3", "V4"]

        # ENHANCED: Only train specified objectives
        if objective_indices is None:
            objective_indices = list(range(5))

        self.objective_indices = sorted(objective_indices)
        self.obj_names = [self.all_obj_names[i] for i in self.objective_indices]

        # Train one ensemble per UNCOVERED objective only
        self.ensembles = {}
        for obj_idx in self.objective_indices:
            obj_name = self.all_obj_names[obj_idx]
            self.ensembles[obj_name] = SAMOTAPerObjectiveEnsemble(
                X_train,
                F_train[:, obj_idx],
                normalize=normalize,
                obj_name=obj_name
            )

    def predict_all(self, X):
        """
        Predict selected objectives with uncertainties.

        ENHANCED: Only predicts objectives in self.objective_indices
        Returns full 5D array with NaN for objectives not trained.

        Args:
            X: Parameters (N, 6) or (6,)

        Returns:
            predictions: (N, 5) or (5,) array of predictions
            uncertainties: (N, 5) or (5,) array of uncertainties
        """
        # Initialize full 5D arrays with NaN (for untrained objectives)
        scalar_input = len(X.shape) == 1
        if scalar_input:
            full_predictions = np.full(5, np.nan)
            full_uncertainties = np.full(5, np.nan)
        else:
            full_predictions = np.full((X.shape[0], 5), np.nan)
            full_uncertainties = np.full((X.shape[0], 5), np.nan)

        # Predict ONLY trained objectives
        for obj_idx in self.objective_indices:
            obj_name = self.all_obj_names[obj_idx]
            pred, unc = self.ensembles[obj_name].predict(X)

            if scalar_input:
                full_predictions[obj_idx] = pred
                full_uncertainties[obj_idx] = unc
            else:
                full_predictions[:, obj_idx] = pred
                full_uncertainties[:, obj_idx] = unc

        # Return full 5D arrays (with NaN for untrained objectives)
        return full_predictions, full_uncertainties

    def predict_single_objective(self, X, obj_name):
        """
        Predict single objective (for local search focus).

        Args:
            X: Parameters (N, 6) or (6,)
            obj_name: "V0", "V1", "V2", "V3", or "V4"

        Returns:
            predictions, uncertainties for that objective
        """
        return self.ensembles[obj_name].predict(X)

    def retrain_all(self, X_new, F_new):
        """
        Retrain all 5 surrogates with new data.

        Args:
            X_new: New parameters (M, 6)
            F_new: New violation scores (M, 5)
        """
        for obj_idx, obj_name in enumerate(self.obj_names):
            self.ensembles[obj_name].retrain(X_new, F_new[:, obj_idx])

    def get_uncertainties_threshold(self):
        """
        Get uncertainty thresholds per objective (for validation decision).
        Threshold = 1.0 × MAE (validate when uncertainty exceeds this).

        Returns:
            dict: {obj_name: threshold_value}
        """
        thresholds = {}
        for obj_name in self.obj_names:
            thresholds[obj_name] = self.ensembles[obj_name].mae
        return thresholds

    def get_model_errors(self):
        """
        Get training errors per objective (for diagnostics).

        Returns:
            dict: {obj_name: {"gp": error, "poly": error, "rbf": error, "mae": error}}
        """
        errors = {}
        for obj_name in self.obj_names:
            ensemble = self.ensembles[obj_name]
            errors[obj_name] = {
                "gp": ensemble.e_gp,
                "poly": ensemble.e_poly,
                "rbf": ensemble.e_rbf,
                "mae": ensemble.mae,
                "weights": {
                    "gp": ensemble.w_gp,
                    "poly": ensemble.w_poly,
                    "rbf": ensemble.w_rbf
                }
            }
        return errors


