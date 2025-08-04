import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error as MSE
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from pandas import ExcelWriter
import warnings

warnings.filterwarnings('ignore')

"""
Take the prediction of LFL using the SVR model on the associative dataset as an example. 
Other statistical machine learning models can be implemented similarly by substituting the model type, 
adjusting hyperparameters, and/or using a different dataset as needed.
"""

# ===================== DATA LOADING AND PREPROCESSING =====================

def load_and_prepare_data():
    """
    Load and prepare mixture and pure substance datasets.
    

    Returns:
        pd.DataFrame: Combined dataset with mixture and pure substance data
        list: List of feature column names
        list: List of all column names including target
    """
    # Load datasets
    mixture_data = pd.read_excel(r"Refrigerant mixture dataset.xlsx", sheet_name='LFL')
    pure_data = pd.read_excel(r"Pure substance dataset.xlsx", sheet_name='LFL')

    # Define feature columns (molecular descriptors)
    feature_columns = [
        'Chi1n',  # Connectivity index
        'fr_halogen',  # Number of halogen atoms
        'MolMR',  # Molecular refractivity
        'BCUT2D_MWLOW',  # BCUT descriptor
        'VSA_EState8',  # VSA EState descriptor
        'MolLogP',  # Molecular LogP
        'MinEStateIndex',  # Minimum EState index
        'SlogP_VSA10',  # SlogP VSA descriptor
        'fr_alkyl_halide',  # Number of alkyl halides
        'NumHeteroatoms'  # Number of heteroatoms
    ]

    # All columns including target
    all_columns = feature_columns + ['LFL(vol%)']

    # Prepare mixture data
    data_mixture = mixture_data[all_columns].copy()
    # Load the mixture class to assign labels to mixtures sharing identical components.
    data_mixture['Class'] = mixture_data['Class']
    data_mixture['Type'] = mixture_data['Type']

    print(max(data_mixture['Class'])+1)

    # Prepare pure substance data
    data_pure = pure_data[all_columns].copy()
    # Each pure organic substance is assigned to its own distinct class.
    data_pure['Class'] = range(max(data_mixture['Class'])+1, data_pure.shape[0] + max(data_mixture['Class'])+1)  # Assign unique IDs
    data_pure['Type'] = pure_data['Type']

    # Combine datasets
    combined_data = pd.concat([data_mixture, data_pure], axis=0)

    return combined_data, data_mixture, data_pure, feature_columns, all_columns


# ===================== HYPERPARAMETER OPTIMIZATION =====================

def define_parameter_grid():
    """
    Define hyperparameter grid for SVR model optimization.

    Returns:
        dict: Parameter grid for grid search
    """
    param_grid = {
        'kernel': ['rbf'],  # Radial basis function kernel
        'C': list(np.arange(0.2, 1.1, 0.2)) + list(np.arange(2, 11, 2)),  # Regularization parameter
        'epsilon': (list(np.arange(0.0002, 0.001, 0.0002)) +
                    list(np.arange(0.002, 0.01, 0.002)) +
                    list(np.arange(0.02, 0.1, 0.02)))  # Epsilon parameter for SVR
    }
    return param_grid


def optimize_hyperparameters(full_train_data, feature_columns, param_grid):
    """
    Optimize hyperparameters using nested cross-validation.

    Args:
        full_train_data (pd.DataFrame): Training data
        feature_columns (list): List of feature column names
        param_grid (dict): Parameter grid for optimization

    Returns:
        tuple: Best parameters and best weight
    """
    best_score = -np.inf
    best_params = {}
    best_weight = 1

    # Grid search over all parameter combinations
    for params in ParameterGrid(param_grid):
        fold_scores = []

        # Test different sample weights (currently only weight=1)
        for weight in [1]:
            # Inner cross-validation for parameter evaluation
            inner_kf = KFold(n_splits=10, shuffle=True, random_state=42)

            for inner_train_idx, inner_valid_idx in inner_kf.split(full_train_data):
                # Split inner training and validation sets
                inner_train = full_train_data.iloc[inner_train_idx]
                inner_valid = full_train_data.iloc[inner_valid_idx]

                # Prepare features and targets
                X_train = inner_train[feature_columns]
                y_train = np.log10(inner_train['LFL(vol%)'])  # Log transformation
                X_valid = inner_valid[feature_columns]
                y_valid = np.log10(inner_valid['LFL(vol%)'])

                # Standardize features
                scaler = StandardScaler()
                X_train_std = scaler.fit_transform(X_train)
                X_valid_std = scaler.transform(X_valid)

                # Set sample weights (can be used to weight mixture vs pure samples differently)
                sample_weight = np.ones(len(inner_train))
                sample_weight[inner_train['Type'] == 'Mixture'] = weight

                # Train and evaluate model
                model = SVR(**params)
                model.fit(X_train_std, y_train, sample_weight=sample_weight)
                score = model.score(X_valid_std, y_valid)
                fold_scores.append(score)

            # Update best parameters if current combination is better
            avg_score = np.mean(fold_scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_weight = weight

    return best_params, best_weight


# ===================== MODEL TRAINING AND EVALUATION =====================

def train_and_evaluate_fold(full_train_data, test_mix_data, test_pure_data,
                            feature_columns, best_params, best_weight):
    """
    Train final model and evaluate on test sets.

    Args:
        full_train_data (pd.DataFrame): Full training data
        test_mix_data (pd.DataFrame): Mixture test data
        test_pure_data (pd.DataFrame): Pure substance test data
        feature_columns (list): List of feature column names
        best_params (dict): Best hyperparameters
        best_weight (float): Best sample weight

    Returns:
        tuple: Predictions and true values for mixture, pure, and combined test sets
    """
    # Prepare training data
    X_train = full_train_data[feature_columns]
    y_train = np.log10(full_train_data['LFL(vol%)'])

    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    # Set sample weights
    sample_weight = np.ones(len(full_train_data))
    sample_weight[full_train_data['Type'] == 'Mixture'] = best_weight

    # Train final model
    final_model = SVR(**best_params)
    final_model.fit(X_train_std, y_train, sample_weight=sample_weight)

    # Evaluate on mixture test set
    X_test_mix = test_mix_data[feature_columns]
    y_test_mix = np.log10(test_mix_data['LFL(vol%)'])
    X_test_mix_std = scaler.transform(X_test_mix)
    pred_mix = final_model.predict(X_test_mix_std)

    # Evaluate on pure substance test set
    X_test_pure = test_pure_data[feature_columns]
    y_test_pure = np.log10(test_pure_data['LFL(vol%)'])
    X_test_pure_std = scaler.transform(X_test_pure)
    pred_pure = final_model.predict(X_test_pure_std)

    # Evaluate on combined test set
    combined_test = pd.concat([test_mix_data, test_pure_data])
    X_test_all = combined_test[feature_columns]
    y_test_all = np.log10(combined_test['LFL(vol%)'])
    X_test_all_std = scaler.transform(X_test_all)
    pred_all = final_model.predict(X_test_all_std)

    return (pred_mix, y_test_mix, pred_pure, y_test_pure, pred_all, y_test_all)


# ===================== CROSS-VALIDATION FRAMEWORK =====================

def run_cross_validation(data_mixture, data_pure, feature_columns, param_grid):
    """
    Run K-fold cross-validation for model evaluation.

    Args:
        data_mixture (pd.DataFrame): Mixture dataset
        data_pure (pd.DataFrame): Pure substance dataset
        feature_columns (list): List of feature column names
        param_grid (dict): Parameter grid for optimization

    Returns:
        tuple: Lists of predictions and true values, best parameters and weights
    """
    # Initialize result storage
    predictions_mix_all = []
    true_mix_all = []
    predictions_pure_all = []
    true_pure_all = []
    predictions_all_all = []
    true_all_all = []
    best_params_list = []
    best_weights_list = []

    # Set up K-fold cross-validation
    kf_mix = KFold(n_splits=10, shuffle=True, random_state=0)
    kf_pure = KFold(n_splits=10, shuffle=True, random_state=0)
    split_indices = pd.DataFrame(range(max(data_mixture['Class'])))  # Assuming 76 mixture samples

    # Perform cross-validation
    fold_iterator = zip(kf_mix.split(split_indices), kf_pure.split(data_pure))
    
    # Divide the dataset based on class.
    for fold, ((train_mix_idx, test_mix_idx), (train_pure_idx, test_pure_idx)) in enumerate(fold_iterator):
        print(f"Processing fold {fold + 1}/10...")

        # Split mixture data
        train_mix = data_mixture[data_mixture['Class'].isin(train_mix_idx + 1)]
        test_mix = data_mixture[data_mixture['Class'].isin(test_mix_idx + 1)]

        # Split pure substance data
        train_pure = data_pure.iloc[train_pure_idx]
        test_pure = data_pure.iloc[test_pure_idx]

        # Combine training data
        full_train = pd.concat([train_mix, train_pure])

        # Optimize hyperparameters for current fold
        best_params, best_weight = optimize_hyperparameters(full_train, feature_columns, param_grid)
        best_params_list.append(best_params)
        best_weights_list.append(best_weight)

        # Train and evaluate model
        results = train_and_evaluate_fold(full_train, test_mix, test_pure,
                                          feature_columns, best_params, best_weight)
        pred_mix, true_mix, pred_pure, true_pure, pred_all, true_all = results

        # Store results
        predictions_mix_all.extend(pred_mix.tolist())
        true_mix_all.extend(true_mix.tolist())
        predictions_pure_all.extend(pred_pure.tolist())
        true_pure_all.extend(true_pure.tolist())
        predictions_all_all.extend(pred_all.tolist())
        true_all_all.extend(true_all.tolist())

    return (predictions_mix_all, true_mix_all, predictions_pure_all, true_pure_all,
            predictions_all_all, true_all_all, best_params_list, best_weights_list)


# ===================== EVALUATION METRICS =====================

def calculate_metrics(y_true, y_pred, dataset_name):
    """
    Calculate and print evaluation metrics.

    Args:
        y_true (list): True values
        y_pred (list): Predicted values
        dataset_name (str): Name of the dataset for printing
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(MSE(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{dataset_name} Results:")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return r2, rmse, mae


def save_results(y_true, y_pred, filename):
    """
    Save prediction results to Excel file.

    Args:
        y_true (list): True values
        y_pred (list): Predicted values
        filename (str): Output filename
    """
    results_df = pd.DataFrame({
        'Log_LFL_True': y_true,
        'Log_LFL_Predicted': y_pred
    })

    with ExcelWriter(filename) as writer:
        results_df.to_excel(writer, sheet_name='SVR_Results', index=False)

    print(f"\nResults saved to {filename}")


# ===================== MAIN EXECUTION =====================

def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("SVR Model for LFL Prediction")
    print("=" * 60)

    # Load and prepare data
    print("\nLoading and preparing data...")
    combined_data, data_mixture, data_pure, feature_columns, all_columns = load_and_prepare_data()

    print(f"Mixture samples: {len(data_mixture)}")
    print(f"Pure substance samples: {len(data_pure)}")
    print(f"Total samples: {len(combined_data)}")
    print(f"Number of features: {len(feature_columns)}")

    # Define parameter grid
    param_grid = define_parameter_grid()
    print(f"\nHyperparameter search space:")
    for param, values in param_grid.items():
        print(f"  {param}: {len(values)} values")

    # Run cross-validation
    print("\nStarting 10-fold cross-validation...")
    results = run_cross_validation(data_mixture, data_pure, feature_columns, param_grid)
    (pred_mix_all, true_mix_all, pred_pure_all, true_pure_all,
     pred_all_all, true_all_all, best_params_list, best_weights_list) = results

    # Print best parameters
    print("\nBest parameters for each fold:")
    for i, (params, weight) in enumerate(zip(best_params_list, best_weights_list)):
        print(f"Fold {i + 1}: {params}, Weight: {weight}")

    # Calculate and display metrics
    print("\n" + "=" * 40)
    print("FINAL EVALUATION RESULTS")
    print("=" * 40)

    calculate_metrics(true_mix_all, pred_mix_all, "Mixture Test Set")
    calculate_metrics(true_pure_all, pred_pure_all, "Pure Substance Test Set")
    calculate_metrics(true_all_all, pred_all_all, "All Test Set")

    # Save results to Excel
    save_results(true_all_all, pred_all_all, '../SVR-LFL-mixture-based—10特征.xlsx')


if __name__ == "__main__":
    main()

