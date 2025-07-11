import os
hostname = os.uname()[1]
if hostname == 'GLaDOS':
    import sys
    sys.path.append('/home/wfd/WFP')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.DatabaseGetInfo import DatabaseAnalyzer
from src.ThesisProject.CF_table.CF_interval_detect import detect_and_plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Reshape, MultiHeadAttention, GlobalAveragePooling1D, Add, LayerNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.backend import clear_session


if hostname == 'penguin':
    os.environ['KMP_AFFINITY'] = 'disabled'
if hostname == 'GLaDOS':
    import sys
    sys.path.append('home/wfd/WFP')

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize
hostname = os.uname()[1]
if hostname == 'penguin':
    path = os.path.abspath('/home/wheatley/WFD/wfd.db')
elif hostname == 'GLaDOS':
    path = os.path.abspath('/home/wfd/WFD/wfd.db')
analyzer = DatabaseAnalyzer.WindFarmAnalyzer(path)


def evaluate_and_plot(y_true, y_pred, production_lag, time_steps, forecast_horizon, prediction_method, installed_capacity=None):
    """
    Calculates error metrics and generates plots for evaluating wind power forecasts.

    Args:
        y_true (np.ndarray): Array of true production values.
        y_pred (np.ndarray): Array of predicted production values.
        production_lag (int): Number of past production lags used as input.
        time_steps (int): Number of time steps used for each feature.
        forecast_horizon (int): Forecast horizon (number of steps ahead).
        prediction_method (str): Prediction method ('single_shot' or 'autoregressive').
        installed_capacity (float, optional): Installed capacity of the wind farm (for normalization). Defaults to None.
    """

    # --- Error Calculation ---
    def calculate_errors(y_true, y_pred, installed_capacity, forecast_horizon):
        errors = {}

        if forecast_horizon == 1:
            errors['R^2'] = r2_score(y_true, y_pred)
            errors['MAE'] = mean_absolute_error(y_true, y_pred)
            errors['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            errors['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            if installed_capacity:
                errors['nMAE_capacity'] = errors['MAE'] / installed_capacity
                errors['nRMSE_capacity'] = errors['RMSE'] / installed_capacity

            # Normalize by the diff of highest and lowest values in y_true
            if np.mean(y_true) != 0:
                errors['nMAE'] = errors['MAE'] / (np.max(y_true) - np.min(y_true))
                errors['nRMSE'] = errors['RMSE'] / (np.max(y_true) - np.min(y_true))
            else:
                errors['nMAE'] = np.nan
                errors['nRMSE'] = np.nan

        elif forecast_horizon > 1:

            print(y_true.shape)
            print(y_pred.shape)

            # Per-horizon errors
            r2_values = []
            mae_values = []
            rmse_values = []
            mape_values = []
            nrmse_values = []
            nmae_values = []

            for h in range(forecast_horizon):
                errors[f'R^2_t+{h + 1}'] = r2_score(y_true[:, h], y_pred[:, h])
                errors[f'MAE_t+{h + 1}'] = mean_absolute_error(y_true[:, h], y_pred[:, h])
                errors[f'RMSE_t+{h + 1}'] = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
                errors[f'MAPE_t+{h + 1}'] = np.mean(np.abs((y_true[:, h] - y_pred[:, h]) / y_true[:, h])) * 100

                if installed_capacity:
                    errors[f'nMAE_capacity_t+{h + 1}'] = errors[f'MAE_t+{h + 1}'] / installed_capacity
                    errors[f'nRMSE_capacity_t+{h + 1}'] = errors[f'RMSE_t+{h + 1}'] / installed_capacity
                else:
                    # Normalize by the diff of highest and lowest values in y_true
                    if np.mean(y_true[:, h]) != 0:
                        errors[f'nMAE_t+{h + 1}'] = errors[f'MAE_t+{h + 1}'] / (np.max(y_true[:, h]) - np.min(y_true[:, h]))
                        errors[f'nRMSE_t+{h + 1}'] = errors[f'RMSE_t+{h + 1}'] / (np.max(y_true[:, h]) - np.min(y_true[:, h]))
                    else:
                        errors[f'nMAE_t+{h + 1}'] = np.nan
                        errors[f'nRMSE_t+{h + 1}'] = np.nan

                r2_values.append(errors[f'R^2_t+{h + 1}'])
                mae_values.append(errors[f'MAE_t+{h + 1}'])
                rmse_values.append(errors[f'RMSE_t+{h + 1}'])
                mape_values.append(errors[f'MAPE_t+{h + 1}'])
                if installed_capacity:
                    nmae_values.append(errors[f'nMAE_capacity_t+{h + 1}'])
                    nrmse_values.append(errors[f'nRMSE_capacity_t+{h + 1}'])
                else:
                    nmae_values.append(errors[f'nMAE_t+{h + 1}'])
                    nrmse_values.append(errors[f'nRMSE_t+{h + 1}'])

            # Aggregate errors (example: averaging)
            errors['R^2_avg'] = np.mean(r2_values)
            errors['MAE_avg'] = np.mean(mae_values)
            errors['RMSE_avg'] = np.mean(rmse_values)
            errors['MAPE_avg'] = np.mean(mape_values)
            errors['nMAE_avg'] = np.mean(nmae_values)
            errors['nRMSE_avg'] = np.mean(nrmse_values)

            # Alternatively, you could flatten and calculate:
            errors['R^2_overall'] = r2_score(y_true.flatten(), y_pred.flatten())
            errors['MAE_overall'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            errors['RMSE_overall'] = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
            errors['MAPE_overall'] = np.mean(np.abs((y_true.flatten() - y_pred.flatten()) / y_true.flatten())) * 100
            if installed_capacity:
                errors['nMAE_overall'] = errors['MAE_overall'] / installed_capacity
                errors['nRMSE_overall'] = errors['RMSE_overall'] / installed_capacity
            else:
                # Normalize by the diff of highest and lowest values in y_true
                if np.mean(y_true) != 0:
                    errors['nMAE_overall'] = errors['MAE_overall'] / (np.max(y_true) - np.min(y_true))
                    errors['nRMSE_overall'] = errors['RMSE_overall'] / (np.max(y_true) - np.min(y_true))
                else:
                    errors['nMAE_overall'] = np.nan
                    errors['nRMSE_overall'] = np.nan

        # write the errors to a file
        with open(f'model_attention/{folder_name}/errors_{prediction_method}_{forecast_horizon}.txt'.format(
                folder_name=folder_name,
                prediction_method = prediction_method,
                forecast_horizon = forecast_horizon), 'w') as f:
            for metric, value in errors.items():
                f.write(f"{metric}: {value}\n")

        return errors

    # --- Plotting ---
    def plot_time_series(y_true, y_pred, forecast_horizon, time_steps):
        num_samples = y_true.shape[0]
        time_index = np.arange(num_samples) + time_steps

        if forecast_horizon == 1:
            plt.figure(figsize=(12, 6))
            plt.plot(time_index, y_true, label='Actual', color='blue')
            plt.plot(time_index, y_pred, label='Predicted', color='red', linewidth=0.6)
            plt.xlabel('Time Steps (from the start of test period)')
            plt.ylabel('Production')
            plt.title('Time Series of Actual vs. Predicted Production')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'model_attention/{folder_name}/time_series_{prediction_method}_{forecast_horizon}.png')
            plt.close()

        else:
            if forecast_horizon <= 8:
                rows = int(np.ceil(forecast_horizon / 2))
                fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), sharex=True)
                axes = axes.ravel()
            if forecast_horizon > 8:
                rows = int(np.ceil(forecast_horizon / 4))
                fig, axes = plt.subplots(rows, 4, figsize=(30, 5 * rows), sharex=True)
                axes = axes.ravel()

            for h in range(forecast_horizon):
                # Select actual and predicted values for the current horizon
                y_true_h = y_true[:, h]
                y_pred_h = y_pred[:, h]

                axes[h].plot(time_index, y_true_h, label='Actual', color='blue')
                axes[h].plot(time_index, y_pred_h, label='Predicted', color='red', linewidth=0.6)
                axes[h].set_title(f'Forecast Horizon: t+{h + 1}')
                axes[h].set_ylabel('Production')
                axes[h].legend()
                axes[h].grid(True)

            # Remove any unused subplots
            for h in range(forecast_horizon, rows * 2):
                fig.delaxes(axes[h])

            plt.xlabel('Time Steps (from the start of test period)')
            plt.tight_layout()
            plt.savefig(f'model_attention/{folder_name}/time_series_{prediction_method}_{forecast_horizon}.png')
            plt.close()


    def plot_correlation(y_true, y_pred, forecast_horizon):
        if forecast_horizon == 1:
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, alpha=0.5, color='green')
            plt.xlabel('Actual Production')
            plt.ylabel('Predicted Production')
            plt.title('Correlation between Actual and Predicted Production')

            # Add y=x line
            #lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
            lims = [np.min(y_true), np.max(y_true)]
            plt.plot(lims, lims, linestyle='--', color='black', alpha=0.75, label='y=x')

            plt.legend()
            plt.grid(True)
            plt.savefig(f'model_attention/{folder_name}/correlation_{prediction_method}_{forecast_horizon}.png')
            plt.close()

        else:
            if forecast_horizon <= 8:
                rows = int(np.ceil(forecast_horizon / 2))
                fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows), sharex=True, sharey=True)
                axes = axes.ravel()
            else:
                rows = int(np.ceil(forecast_horizon / 4))
                fig, axes = plt.subplots(rows, 4, figsize=(30, 5 * rows), sharex=True, sharey=True)
                axes = axes.ravel()

            for h in range(forecast_horizon):
                # Select actual and predicted values for the current horizon
                y_true_h = y_true[:, h]
                y_pred_h = y_pred[:, h]

                axes[h].scatter(y_true_h, y_pred_h, alpha=0.5, color='green')
                axes[h].set_xlabel('Actual Production')
                axes[h].set_ylabel('Predicted Production')
                axes[h].set_title(f'Correlation (t+{h + 1})')

                # Add y=x line
                #lims = [np.min([axes[h].get_xlim(), axes[h].get_ylim()]), np.max([axes[h].get_xlim(), axes[h].get_ylim()])]
                lims = [np.min(y_true_h), np.max(y_true_h)]
                axes[h].plot(lims, lims, linestyle='--', color='black', alpha=0.75, label='y=x')

                axes[h].legend()
                axes[h].grid(True)

            # Remove any unused subplots
            for h in range(forecast_horizon, rows * (2 if forecast_horizon <= 8 else 4)):
                fig.delaxes(axes[h])

            plt.tight_layout()
            plt.savefig(f'model_attention/{folder_name}/correlation_{prediction_method}_{forecast_horizon}.png')
            plt.close()


    def plot_metrics_vs_horizon(errors, forecast_horizon):
        r2_values = []
        nrmse_values = []
        nmae_values = []

        # Extract relevant metrics from the errors dictionary
        for h in range(forecast_horizon):
            r2_values.append(errors[f'R^2_t+{h + 1}'])
            if installed_capacity:
                nrmse_values.append(errors[f'nRMSE_capacity_t+{h + 1}'])
                nmae_values.append(errors[f'nMAE_capacity_t+{h + 1}'])
            else:
                nrmse_values.append(errors[f'nRMSE_t+{h + 1}'])
                nmae_values.append(errors[f'nMAE_t+{h + 1}'])

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot nRMSE and nMAE on the left y-axis
        ax1.plot(np.arange(1, forecast_horizon + 1), nrmse_values, marker='s', color='red', label='nRMSE')
        ax1.plot(np.arange(1, forecast_horizon + 1), nmae_values, marker='^', color='green', label='nMAE')
        ax1.set_xlabel('Forecast Horizon (t+n)')
        ax1.set_ylabel('nRMSE & nMAE', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        #ax1.set_xticks(np.arange(0, forecast_horizon + 1, 1))  # Set x-axis ticks

        ax1.grid(True)

        # Customize x-axis tick labels for better visibility
        if forecast_horizon > 10:
            ax1.xaxis.set_major_locator(plt.MaxNLocator(10))  # Reduce the number of ticks if too many

        # Rotate x-axis labels if needed
        # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Create a second y-axis for R^2
        ax2 = ax1.twinx()
        ax2.plot(np.arange(1, forecast_horizon + 1), r2_values, marker='o', color='blue', label='R^2')
        ax2.set_ylabel('R^2', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Add a legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        plt.title('R^2, nRMSE, and nMAE vs. Forecast Horizon')
        plt.tight_layout()
        plt.savefig(f'model_attention/{folder_name}/metrics_vs_horizon_{prediction_method}_{forecast_horizon}.png')
        plt.close()


    def plot_multiple_predictions_subplots(y_true, y_pred, forecast_horizon, folder_name, prediction_method):

        prediction_indices = [1, 4 * forecast_horizon, 8 * forecast_horizon, 16 * forecast_horizon]

        # Create a figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))  # 2 rows, 2 columns
        fig.suptitle('Time Series of Actual vs. Predicted Production at Different Lead Times')
        axes = axes.flatten()  # Flatten the axes array for easier indexing

        for i, pred_index in enumerate(prediction_indices):
            if pred_index < y_true.shape[0]:
                y_true_slice = y_true[pred_index, :, 0]
                y_pred_slice = y_pred[pred_index, :, 0]

                # Plot on the corresponding subplot
                ax = axes[i]
                ax.plot(y_true_slice, label=f'Actual', color='blue')
                ax.plot(y_pred_slice, label=f'Predicted', color='red', linewidth=0.6)
                ax.set_ylim(0, np.max(y_true))
                ax.set_xlim(0, forecast_horizon)
                #ax.set_xticks(np.arange(1, forecast_horizon+1, 1))
                ax.set_xlabel('Time Steps (from start of test period)')
                ax.set_ylabel('Production')
                ax.set_title(f'Prediction at t+{pred_index}')
                ax.legend()
                ax.grid(True)

        plt.tight_layout()  # Adjust spacing between subplots
        plt.savefig(f'model_attention/{folder_name}/time_series_4subplots_{prediction_method}_{forecast_horizon}.png')
        plt.close()

    # --- Main part of the function ---
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Calculate errors
    errors = calculate_errors(y_true, y_pred, installed_capacity, forecast_horizon)

    # Print errors
    print("Error Metrics:")
    for metric, value in errors.items():
        if dense == 1:
            if metric in ['nMAE_capacity', 'nRMSE_capacity', 'nMAE', 'nRMSE']:
                # print percentage
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")

        elif dense > 1:
            if 'avg' in metric:
                print(f"{metric}: {value:.4f}")
            elif 'nMAE_capacity' in metric or 'nRMSE_capacity' in metric or 'nMAE' in metric or 'nRMSE' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")

    # Generate plots
    plot_time_series(y_true, y_pred, forecast_horizon, time_steps)
    plot_correlation(y_true, y_pred, forecast_horizon)
    if forecast_horizon > 1:
        plot_metrics_vs_horizon(errors, forecast_horizon)
        plot_multiple_predictions_subplots(y_true, y_pred, forecast_horizon, folder_name, prediction_method)

    return None



def lstm_wind_power_forecast(wf_id, start_date, end_date, time_steps=1, batch_size=128, epochs=100, dropout_rate=0.25,
                             learning_rate=0.001, number_of_cells=[200, 100, 50], early_stopping_patience=3,
                             variables=['temperature', 'pressure', 'dew_point', 'ws100'], production_lag=0,
                             model_name='noname', scaler_name='noname', filter_data=False, save_model=False,
                             forecast_horizon=24, folder_name='', prediction_method='single_shot', dense=1):

    X = (analyzer.get_era5_data(wf_id, start_date, end_date, grid_number=0, variables_to_plot=variables))

    # Get production data
    if filter_data:
        y = detect_and_plot(wf_id, start_date, end_date, plot=False)
        X = X.set_index('timestamp')
        y = y.set_index('timestamp')
        # Get the timestamps present in y
        y_timestamps = y.index
        # Filter X to keep only the timestamps present in y
        X_filtered = X[X.index.isin(y_timestamps)]
        # Reset index if needed
        X = X_filtered.reset_index().drop(columns=['timestamp'])
        y = y.reset_index()['production']
    else:
        y = analyzer.get_wind_production_data(wf_id, start_date, end_date, CF=False)['production']
        X = X.drop(columns=['timestamp'])

    if production_lag > 0:
        # Create lagged copies of y for each lag up to production_lag
        for lag in reversed(range(1, production_lag + 1)):
            y_lagged = y.shift(lag)
            y_lagged[:lag] = 0  # Fill initial NaN values with 0
            X[f'production_lagged_{lag}'] = y_lagged

    y = np.asarray(y).reshape(-1, 1)

    # Create time steps
    features = X.copy()

    for i in range(1, time_steps):
        shifted_df = X.shift(i)
        shifted_df.columns = [f"{col}_t-{i}" for col in X.columns]
        features = pd.concat([features, shifted_df], axis=1)  # axis=1 = (sample, *time_steps+1*, features)

    features.dropna(inplace=True)

    X = features.values
    y = y[time_steps - 1:]  # Adjust y to align with the shifted X

    def create_multi_step_target(y, steps=forecast_horizon):
        """Creates a multi-step target variable."""
        if prediction_method == 'single_shot':
            y_multistep = []
            for i in range(len(y) - steps + 1):
                y_multistep.append(y[i:i + steps])
        else:
            y_multistep = y[:-steps + 1]
        return np.array(y_multistep)

    if forecast_horizon > 1:
        y = create_multi_step_target(y, steps=forecast_horizon)
        if prediction_method == 'single_shot':
            X = X[:-forecast_horizon + 1]  # Adjust X to align with the new y
        else:
            X = X[:-forecast_horizon + 1]


    print(X.shape, y.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, shuffle=False)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape input data
    X_train = X_train.reshape((X_train.shape[0], time_steps, -1))  # Reshape with time_steps
    X_test = X_test.reshape((X_test.shape[0], time_steps, -1))  # Reshape with time_steps

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#################################################################################################
    def build_lstm_with_attention(time_steps, num_features_input, number_of_cells, dropout_rate, forecast_horizon):
        """Builds an LSTM model with Multi-Head Attention using Functional API."""

        inputs = Input(shape=(time_steps, num_features_input))
        x = inputs

        # LSTM Layers - All must return sequences for attention
        for i, units in enumerate(number_of_cells):
            # Ensure all LSTM layers return sequences
            x = LSTM(units=units, return_sequences=True)(x)
            # Optional dropout between LSTM layers (can sometimes be helpful)
            # if dropout_rate > 0 and i < len(number_of_cells) - 1:
            #     x = Dropout(dropout_rate)(x)

        # --- Attention Mechanism ---
        # Using MultiHeadAttention. Output of last LSTM is used for query, key, and value.
        # num_heads and key_dim are hyperparameters to tune.
        num_heads = 4  # Example value
        key_dim = number_of_cells[-1] // num_heads  # Example: Ensure key_dim * num_heads is reasonable

        # Calculate attention weights and apply them
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(query=x, value=x, key=x)

        # --- Optional: Add & Norm (Common practice after attention) ---
        # Add the attention output back to the input of the attention layer (residual connection)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        # --- End Optional Add & Norm ---

        # Dropout after attention/norm block
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

        # Reduce the sequence dimension to a single vector
        # Global Average Pooling is a common choice
        context_vector = GlobalAveragePooling1D()(x)  # Shape: (batch, last_lstm_units)

        # Final Dense layers to predict the horizon
        # You might experiment with adding another Dense layer here
        # x = Dense(64, activation='relu')(context_vector)
        outputs_flat = Dense(forecast_horizon, name='output_dense')(context_vector)  # Shape: (batch, forecast_horizon)

        # Reshape to the target format [batch, horizon, 1 feature]
        outputs = Reshape([forecast_horizon, 1], name='output_reshape')(outputs_flat)

        model = Model(inputs=inputs, outputs=outputs)
        return model

#################################################################################################

    # 1. Build the attention model by CALLING the function
    #    (REMOVE the old Sequential model building loop entirely)
    print("Building LSTM model with Attention...")
    model = build_lstm_with_attention(time_steps, X_train.shape[2], number_of_cells, dropout_rate, forecast_horizon)

    # 2. Define the optimizer (Use the recommended lower learning rate)
    #    Make sure 'learning_rate' variable holds the desired INITIAL learning rate (e.g., 0.001)
    print(f"Using initial learning rate: {learning_rate}")
    optimizer = Adam(learning_rate=learning_rate)

    # 3. Compile the model
    print("Compiling model...")
    # Use 'mae' or 'mse' as needed
    model.compile(loss='mae', optimizer=optimizer)

    # 4. Print model summary
    model.summary()

    # 5. Print parameters to file (This part looks fine)
    print("Saving model summary and metadata...")
    with open(f'model_attention/{folder_name}/model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        # Ensure 'metadata' dictionary is defined correctly before this point
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    # 6. Define Callbacks
    print("Defining callbacks...")
    # Ensure 'early_stopping_patience' is defined
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=early_stopping_patience,
                                   restore_best_weights=True)

    # Calculate patience for LR scheduler, ensuring it's at least 1
    lr_reduce_patience = max(1, int(early_stopping_patience / 4))
    print(f"LR Scheduler patience: {lr_reduce_patience}")
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.2,
                                     patience=lr_reduce_patience,
                                     verbose=1,
                                     min_lr=1e-6)

    # 7. Train or Load Model
    #    (Ensure all variables like folder_name, model_name, epochs, batch_size etc. are defined)
    if not os.path.exists(f'model_attention/{folder_name}/{model_name}'):
        print("Training model...")
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=2, shuffle=False,
                            callbacks=[early_stopping, lr_scheduler])  # Pass the callbacks

        # Plot training history (ensure plt is imported and works)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.savefig(f'model_attention/{folder_name}/training_history.png')
            plt.close()  # Close plot to free memory
        except Exception as e:
            print(f"Could not plot training history: {e}")

        # Save model and scaler (ensure save_model is defined)
        if save_model:
            print("Saving model...")
            model.save(f'model_attention/{folder_name}/{model_name}')
            # Ensure 'scaler' object and 'scaler_name' are defined
            try:
                with open(f'model_attention/{folder_name}/{scaler_name}', 'wb') as f:
                    pickle.dump(scaler, f)
                print("Model and scaler saved.")
            except Exception as e:
                print(f"Could not save scaler: {e}")

    else:
        print(f"Loading existing model from: model_attention/{folder_name}/{model_name}")
        # Loading models with custom layers like Attention might need custom_objects in some TF versions
        # Try without first, if it fails, you might need:
        # custom_objects = {'MultiHeadAttention': MultiHeadAttention, 'LayerNormalization': LayerNormalization}
        # model = tf.keras.models.load_model(f'model_attention/{folder_name}/{model_name}', custom_objects=custom_objects)
        try:
            model = tf.keras.models.load_model(f'model_attention/{folder_name}/{model_name}')
            print("Model loaded successfully.")
            # Ensure 'scaler_name' is defined
            print(f"Loading existing scaler from: model_attention/{folder_name}/{scaler_name}")
            with open(f'model_attention/{folder_name}/{scaler_name}', 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            # Decide how to handle error - maybe exit or retrain?
            # return None # Or raise the exception

    # 8. Continue with predictions if model loaded/trained successfully
    # Ensure 'prediction_method' is defined
    if os.path.exists(f'model_attention/{folder_name}/errors_{prediction_method}_{forecast_horizon}.txt'):
        print("Error file exists, skipping prediction and evaluation.")
        # Depending on your workflow, you might want to remove the 'return None'
        # if you want the function to proceed even if the error file exists
        return None  # Original behavior

    # Make predictions
    print("Making predictions...")

    if prediction_method == 'single_shot':
        yhat = model.predict(X_test)
        print('\n')
        print(yhat.shape)
        print(y_test.shape)
        print(X_test.shape)
        print(X_test[:10])
        print(yhat[:10])
        print(y_test[:10])
        print('\n')
        yhat[yhat < 0] = 0
        if dense > 1:
            yhat = yhat.reshape(-1, dense, 1)
            print(yhat.shape)

    elif prediction_method == 'autoregressive':
        if production_lag > 0:
            if forecast_horizon == 1:
                yhat = []
                X_test_temp = X_test.copy()
                # print(X_test_temp[:10])
                # Inverse transform the entire X_test_temp
                X_test_temp = scaler.inverse_transform(X_test_temp.reshape(X_test_temp.shape[0], -1)).reshape(X_test.shape)
                # print(X_test_temp[:10])
                for i in range(len(X_test)):
                    # Reshape the current time steps for scaling
                    current_time_steps = X_test_temp[i].reshape(1, -1)
                    # print('current_input', current_time_steps)
                    # Scale the current time steps
                    current_time_steps_scaled = scaler.transform(current_time_steps)

                    # Reshape for model prediction
                    current_time_steps_scaled = current_time_steps_scaled.reshape((1, time_steps, -1))

                    # Predict using the scaled time steps
                    current_pred = model.predict(current_time_steps_scaled, verbose=0)
                    # set the values below 0 to 0 (array is 2D [[]])
                    current_pred[current_pred < 0] = 0
                    # print('current_pred', current_pred)

                    yhat.append(current_pred)

                    # Update the next sample's last production values
                    if i < len(X_test) - 1:
                        if production_lag == 1:
                            X_test_temp[i + 1, -production_lag:, -1] = np.roll(X_test_temp[i + 1, -production_lag:, -1], -1)
                            X_test_temp[i + 1, -production_lag:, -1][-1] = current_pred
                        elif production_lag > 1:
                            # print('X_test_temp (i+1) before', X_test_temp[i + 1])
                            for lag in range(1, production_lag+1):
                                if i + lag < len(X_test):
                                    for j in range(0, time_steps):
                                        X_test_temp[i + lag][j][-lag] = current_pred
                            # print('X_test_temp (i+1) after', X_test_temp[i + 1])
                            # print('\n')

                yhat = np.array(yhat).reshape(-1, 1)  # Reshape to (num_samples, 1)

            elif forecast_horizon > 1:
                yhat = []
                X_test_temp = X_test.copy()  # Start with scaled X_test

                # --- Adjust y_test to match the multi-step target structure ---
                def multi_step_target(y, steps=forecast_horizon):
                    """Creates a multi-step target variable."""
                    y_multistep = []
                    for i in range(len(y) - steps + 1):
                        y_multistep.append(y[i:i + steps])
                    return np.array(y_multistep)

                # Ensure y_test is 2D before applying multi_step_target
                if y_test.ndim == 1:
                    y_test = y_test.reshape(-1, 1)
                # Apply multi_step_target only if y_test is not already multi-step
                if y_test.shape[1] != forecast_horizon:
                    y_test = multi_step_target(y_test, steps=forecast_horizon)
                # Adjust X_test length to match y_test after multi_step_target
                X_test_temp = X_test_temp[:len(y_test)]
                # --- End of y_test adjustment ---

                # Inverse transform the entire adjusted X_test_temp ONCE at the beginning
                X_test_original_scale = scaler.inverse_transform(X_test_temp.reshape(X_test_temp.shape[0], -1)).reshape(
                    X_test_temp.shape)

                # print("First 10 steps for X test")
                # print(X_test_original_scale[10:])
                # print("First 10 steps for y test")
                # print(y_test[10:])
                # print("First 10 steps for yhat")
                # print(yhat[10:])
                # print('\n')

                for i in range(len(X_test_original_scale)):  # Iterate through each starting point
                    temp_pred_original_scale = []  # Store predictions in original scale for this sequence
                    # Initialize the input window (ORIGINAL SCALE) for the current prediction sequence
                    current_input_window_original = X_test_original_scale[i].copy()  # Shape: (time_steps, features)

                    for j in range(forecast_horizon):  # Iterate for each step in the forecast horizon

                        # if i==344 or i == 345:
                        #     print('i=' + str(i) + ', j=' + str(j))
                        #     print('current_X_test', current_input_window_original)


                        # --- Prepare input for prediction ---
                        # Reshape the current original scale window for scaling
                        current_window_to_scale = current_input_window_original.reshape(1, -1)
                        # Scale the current window
                        current_window_scaled = scaler.transform(current_window_to_scale)
                        # Reshape for model prediction
                        current_input_for_model = current_window_scaled.reshape((1, time_steps, -1))

                        # --- Predict ---
                        # Predict the next step (output is scaled, shape (1, 1))
                        current_pred_scaled = model.predict(current_input_for_model, verbose=0)
                        current_pred_scaled[current_pred_scaled < 0] = 0 # Optional clipping

                        current_pred_original_scale = current_pred_scaled[0, 0]

                        temp_pred_original_scale.append(current_pred_original_scale)

                        # if i==344 or i == 345:
                        #     print('current_pred', temp_pred_original_scale)
                        #     print('\n')

                        # --- Update the ORIGINAL SCALE input window for the *next* prediction step ---
                        if j < forecast_horizon - 1:
                            # Shift the entire time_steps window forward by one step (discard oldest)
                            current_input_window_original = np.roll(current_input_window_original, 1, axis=0)

                            # Update the last time step's production_lag features with the ORIGINAL SCALE prediction
                            # put the newest values for the new (top) row's weather values before putting the predicted values at the last column of the same
                            # this makes the last samples' latest time step wrong, the weather data not exists for the
                            # last hour and is a replicated one from the 'time_steps' amount of time.
                            # But it's okay since the error caused is minimal and the last sample will be discarded anyway somewhere below.
                            if i+j+1 < len(X_test_original_scale):
                                current_input_window_original[0, :] = X_test_original_scale[i+j+1, 0, :]

                            current_input_window_original[0,
                            -production_lag:] = current_pred_original_scale  # Use original scale prediction

                    yhat.append(temp_pred_original_scale)  # Append the sequence of original scale predictions

                yhat = np.array(yhat)
                # Reshape yhat to match the desired output structure: (samples, horizon, 1)
                yhat = yhat.reshape(len(yhat), forecast_horizon, 1)
                yhat[yhat < 0] = 0  # Ensure non-negative values in the final output

                # Ensure y_test also has the 3rd dimension if needed for evaluation
                if y_test.ndim == 2:
                    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

                # print(yhat.shape)
                # print(y_test.shape)
                # print(y_test[10:])
                # print(yhat[10:])
                # print('\n')


            else:
                raise ValueError("Autoregressive prediction requires forecast_horizon > 0")
        else:
            raise ValueError("Autoregressive prediction requires production_lag > 0")


    installed_capacity = float(analyzer.get_moe_data(wf_id)['additional_unit_power_electrical'].sum())
    # Example usage (assuming you have y_true and y_pred from your model):
    evaluate_and_plot(y_test, yhat, production_lag, time_steps, forecast_horizon, prediction_method, installed_capacity=installed_capacity)

    return None



### Input options
start_date = '2020-01-01'
end_date = '2023-12-31'
variables = ['u100', 'v100', 'air_density']
filter_data = False
production_lag = 1 # 0 == no lag
time_steps_list = [4, 8] # 1 == no time steps

### Forecasting options
# When using autoregressive it does not make sense to have forecast_horizon = 1,
# since there is nothing to predict ahead, and we can't use the prediction as input for the next prediction.
# Therefore, if forecast_horizon = 1, the autoregressive method will predict the whole data autoregressively
prediction_method = 'autoregressive' # single_shot or autoregressive
forecast_horizon = 24 # How many hours to predict ahead

wf_id_list = [9, 11, 13,14,  29, 37, 39, 45, 47, 48, 49, 52, 55, 56, 58, 89, 122, 151, 154, 155, 164, 169, 196, 212]

### Hyperparameters
# number_of_cells_list = [[64, 32, 16], [128, 64, 32], [256, 128, 64], [512, 256, 128],
#                         [64, 32], [128, 64], [256, 128], [512, 256],
#                         [64], [128], [256], [512]]
# batch_size_list = [16, 32, 64, 128, 256, 512]
# epochs_list = [10, 25, 50, 100]a
# learning_rate_list = [0.0001, 0.001, 0.01]
# dropout_rate_list = [0, 0.1, 0.2, 0.4]

number_of_cells_list = [[128, 64, 32], [256, 128, 64], [512, 256, 128]]
batch_size_list = [16, 32, 64]
epochs_list = [100]
learning_rate_list = [0.001, 0.01]
dropout_rate_list = [0, 0.2]

for wf_id in wf_id_list:
    for time_steps in time_steps_list:
        for number_of_cells in number_of_cells_list:
            for batch_size in batch_size_list:
                for epochs in epochs_list:
                    for learning_rate in learning_rate_list:
                        for dropout_rate in dropout_rate_list:

                            if prediction_method == 'single_shot':
                                dense = forecast_horizon
                            elif prediction_method == 'autoregressive':
                                dense = 1
                            else:
                                raise ValueError("Invalid prediction method")

                            if epochs < 50 and learning_rate == 0.0001:
                                continue
                            elif len(number_of_cells) == 3 and epochs == 10 and learning_rate == 0.0001:
                                continue
                            elif epochs >= 50 and learning_rate == 0.01 and batch_size >= 64:
                                continue

                            early_stopping_patience = int(20)

                            metadata = {
                                "wf_id": wf_id,
                                "start_date": start_date,
                                "end_date": end_date,
                                "filter_data": filter_data,
                                "production_lag": production_lag,
                                "time_steps": time_steps,
                                "forecast_horizon": forecast_horizon,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "dropout_rate": dropout_rate,
                                "learning_rate": learning_rate,
                                "number_of_cells": number_of_cells,
                                "variables": variables,
                                "prediction_method": prediction_method,
                                "dense": dense
                            }

                            folder_name = f'LSTM_wf{metadata["wf_id"]}_start{metadata["start_date"]}_end{metadata["end_date"]}_filt{metadata["filter_data"]}_lag{metadata["production_lag"]}_steps{metadata["time_steps"]}_bs{metadata["batch_size"]}_ep{metadata["epochs"]}_do{metadata["dropout_rate"]}_lr{metadata["learning_rate"]}_cells[{"_".join(map(str, metadata["number_of_cells"]))}]_vars[{"-".join(metadata["variables"])}]_dense{metadata["dense"]}'
                            model_name = f'model.keras'
                            scaler_name = f'scaler.pkl'
                            print(folder_name)

                            # Create model directory
                            os.makedirs(f'model_attention/{folder_name}', exist_ok=True)

                            if os.path.exists(f'model_attention/{folder_name}/errors_{prediction_method}_{forecast_horizon}.txt') == True:
                                #os.remove(f'model_attention/{folder_name}/errors_{prediction_method}_{forecast_horizon}.txt')
                                clear_session()
                                continue

                            lstm_wind_power_forecast(wf_id, start_date, end_date, time_steps, batch_size, epochs, dropout_rate, learning_rate,
                                                     number_of_cells, early_stopping_patience, variables, production_lag, model_name, scaler_name,
                                                     filter_data, save_model=True, forecast_horizon=forecast_horizon, folder_name=folder_name, prediction_method=prediction_method, dense=dense)
                            gc.collect()
                            clear_session()