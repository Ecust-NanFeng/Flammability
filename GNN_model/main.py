import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from dataset import MixtureDataset, collate_fn
from CV import EarlyStopping, train_epoch, evaluate, grid_search_cv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model import MixtureGNN
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')


def run_final_training(df, best_params_list, label, k=10, epochs=100):
    """
    Run final training using best parameters with K-fold cross-validation.

    Args:
        df (pd.DataFrame): DataFrame containing mixture data
        best_params_list (list): List of best parameters from grid search
        label (str): Target property column name
        k (int): Number of folds for cross-validation
        epochs (int): Number of training epochs

    Returns:
        pd.DataFrame: DataFrame containing training results
        pd.DataFrame: DataFrame containing test results
    """
    all_results = defaultdict(list)
    test_all_df = pd.DataFrame()

    """
    The preset data partitioning scheme from SVR_model was directly adopted, 
    with the training set and test set loaded accordingly. For specific partitioning logic, 
    refer to the implementation in the SVR_model.py file.
    """
    for fold in range(1, k + 1):
        best_params = best_params_list[fold - 1]
        print(f"\n========== Fold {fold}/{k} ==========")

        # Split data into training and testing sets
        test_df = df[df['Kold'] == fold]
        train_df = df[df['Kold'] != fold]
        print(best_params)
        train_dataset, val_dataset = train_test_split(train_df, test_size=0.1, random_state=42, shuffle=True)
        print(
            f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_df)}")

        scaler = StandardScaler()
        train_dataset[label] = scaler.fit_transform(train_dataset[label].values.reshape(-1, 1)).flatten()
        val_dataset[label] = scaler.transform(val_dataset[label].values.reshape(-1, 1)).flatten()
        test_df[label] = scaler.transform(test_df[label].values.reshape(-1, 1)).flatten()

        train_dataset = MixtureDataset(train_dataset, label)
        val_dataset = MixtureDataset(val_dataset, label)
        test_dataset = MixtureDataset(test_df, label)

        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=best_params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=best_params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MixtureGNN(
            num_atom_features=25,
            hidden_dim=best_params['hidden_dim'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Early stopping mechanism
        early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

        # Training history
        train_history = []
        val_history = []
        best_val_mae = float('inf')

        # Training loop
        for epoch in range(epochs):
            train_loss, train_rmse, train_r2, train_mae, _, _ = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_rmse, val_r2, val_mae, _, _ = evaluate(
                model, val_loader, criterion, device, scaler
            )

            # Adjust learning rate
            scheduler.step(val_mae)

            # Record history
            train_history.append(train_mae)
            val_history.append(val_mae)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train MAE={train_mae:.4f}, "
                      f"Val MAE={val_mae:.4f}, Val R2={val_r2:.4f}")

            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae

            # Early stopping check
            if early_stopping(val_mae):
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model for final evaluation
        _, final_train_rmse, final_train_r2, final_train_mae, _, _ = evaluate(
            model, train_loader, criterion, device, scaler
        )
        _, final_val_rmse, final_val_r2, final_val_mae, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device, scaler
        )
        test_df["test_preds"] = test_preds
        test_df["test_targets"] = test_targets
        test_all_df = pd.concat([test_all_df, test_df])
        # Record results
        all_results['fold'].append(fold)
        all_results['train_rmse'].append(final_train_rmse)
        all_results['val_rmse'].append(final_val_rmse)
        all_results['train_r2'].append(final_train_r2)
        all_results['val_r2'].append(final_val_r2)
        all_results['train_mae'].append(final_train_mae)
        all_results['val_mae'].append(final_val_mae)
        all_results['best_epoch'].append(len(train_history))

        print(f"\nFold {fold} final results:")
        print(f"Training set - RMSE: {final_train_rmse:.4f}, R2: {final_train_r2:.4f}, MAE: {final_train_mae:.4f}")
        print(f"Test set - RMSE: {final_val_rmse:.4f}, R2: {final_val_r2:.4f}, MAE: {final_val_mae:.4f}")

    return pd.DataFrame(all_results), test_all_df


# ===================== MAIN PROGRAM =====================

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    mixture_data = pd.read_excel(r"..\Refrigerant mixture dataset.xlsx", sheet_name='LFL')
    pure_data = pd.read_excel(r"..\Pure substance dataset.xlsx", sheet_name='LFL')

    all_data = pd.concat([mixture_data, pure_data])

    all_data['SMILES_B'] = all_data['SMILES_B'].fillna('')
    all_data['SMILES_C'] = all_data['SMILES_C'].fillna('')
    all_data['Ratio_B'] = all_data['Ratio_B'].fillna(0)
    all_data['Ratio_C'] = all_data['Ratio_C'].fillna(0)

    all_data["LFL(vol%)"] = np.log10(all_data["LFL(vol%)"])

    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Define hyperparameter search space
    param_grid = {
        'batch_size': [64],
        'hidden_dim': [64, 128, 256],
        'learning_rate': [0.0005, 0.001, 0.005, 0.01],
        'dropout_rate': [0, 0.1, 0.2, 0.3]
    }

    print("\nHyperparameter search space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    # Start grid search
    start_time = time.time()
    best_params_list, grid_results = grid_search_cv(all_data, param_grid, label="LFL(vol%)", k=10, epochs=200, start=1)

    grid_search_time = time.time() - start_time
    print(f"\nGrid search completed! Time taken: {grid_search_time / 60:.2f} minutes")
    print(f"\nBest parameter combinations:")
    # Save grid search results
    grid_results.to_csv('grid_search_results.csv', index=False)
    best_params_list.to_csv('best_params_list.csv', index=False)

    print(best_params_list)

    final_results, test_df = run_final_training(all_data, best_params_list, label="LFL(vol%)", k=10, epochs=200)

    print("\n" + "=" * 60)

    final_results.to_csv('final_cv_results.csv', index=False)
    test_df.to_excel('final_pre_results.xlsx', index=False)
    print("\n交叉验证结果已保存至 'final_cv_results.csv'")

    print("\n10折交叉验证汇总统计:")
    print("-" * 50)

    metrics = ['rmse', 'r2', 'mae']
    datasets = ['train']
    for dataset in datasets:
        print(f"\n{dataset.upper()}集结果:")
        for metric in metrics:
            col_name = f'{dataset}_{metric}'
            mean_val = final_results[col_name].mean()
            std_val = final_results[col_name].std()
            min_val = final_results[col_name].min()
            max_val = final_results[col_name].max()

            print(f"  {metric.upper()}:")
            print(f"    平均值: {mean_val:.4f} ± {std_val:.4f}")
            print(f"    范围: [{min_val:.4f}, {max_val:.4f}]")
    Types = ["all", "Pure", "Mixture"]
    for Type in Types:
        if Type == "all":
            all_df = test_df
        else:
            all_df = test_df[test_df['Type'] == Type]
        rmse = np.sqrt(mean_squared_error(all_df['test_targets'], all_df['test_preds']))
        r2 = r2_score(all_df['test_targets'], all_df['test_preds'])
        mae = mean_absolute_error(all_df['test_targets'], all_df['test_preds'])
        print(f"{Type}数据集大小：{len(all_df['test_targets'])}")
        print(f"  R2:{r2:.4f}")
        print(f"  RMSE:{rmse:.4f}")
        print(f"  MAE:{mae:.4f}")