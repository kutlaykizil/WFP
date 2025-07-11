import os
import re
import pandas as pd
import numpy as np


def extract_metadata_from_foldername(folder_name):
    """Extracts metadata from the folder name using regular expressions."""
    metadata = {}

    # Match wf_id
    match = re.search(r'wf(\d+)', folder_name)
    if match:
        metadata['wf_id'] = int(match.group(1))

    # Match start_date and end_date
    match = re.search(r'start(\d{4}-\d{2}-\d{2})_end(\d{4}-\d{2}-\d{2})', folder_name)
    if match:
        metadata['start_date'] = match.group(1)
        metadata['end_date'] = match.group(2)

    # Match filter_data
    match = re.search(r'filt(True|False)', folder_name)
    if match:
        metadata['filter_data'] = match.group(1) == 'True'

    # Match production_lag
    match = re.search(r'lag(\d+)', folder_name)
    if match:
        metadata['production_lag'] = int(match.group(1))

    # Match time_steps
    match = re.search(r'steps(\d+)', folder_name)
    if match:
        metadata['time_steps'] = int(match.group(1))

    # Match batch size
    match = re.search(r'bs(\d+)', folder_name)
    if match:
        metadata['batch_size'] = int(match.group(1))
    # Match epochs
    match = re.search(r'ep(\d+)', folder_name)
    if match:
        metadata['epochs'] = int(match.group(1))
    # Match dropout_rate
    match = re.search(r'do(0\.\d+|\d+)', folder_name)
    if match:
        metadata['dropout_rate'] = float(match.group(1))
    # Match learning_rate
    match = re.search(r'lr(0\.\d+|\d+(\.\d+)?(?:e[-+]?\d+)?)', folder_name)
    if match:
        metadata['learning_rate'] = float(match.group(1))

    # Match number_of_cells
    match = re.search(r'cells\[(.*?)\]', folder_name)
    if match:
        cells_str = match.group(1)
        metadata['number_of_cells'] = [int(c) for c in cells_str.replace(" ", "").split('_') if c.isdigit()]

    # Match variables
    match = re.search(r'vars\[(.*?)\]', folder_name)
    if match:
        vars_str = match.group(1)
        metadata['variables'] = vars_str.split('-')

    # match dense
    match = re.search(r'dense(\d+)', folder_name)
    if match:
        metadata['dense'] = int(match.group(1))

    return metadata



def analyze_error_metrics(model_dir):
    """Analyzes error metric files within the model directory."""

    all_results = []

    for folder_name in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder_name)

        if os.path.isdir(folder_path):
            metadata = extract_metadata_from_foldername(folder_name)

            for prediction_method in ['single_shot', 'autoregressive']:
                for file_name in os.listdir(folder_path):
                    if file_name.startswith(f'errors_{prediction_method}_') and file_name.endswith('.txt'):
                        file_path = os.path.join(folder_path, file_name)

                        match = re.search(r'errors_{}_(\d+)\.txt'.format(prediction_method), file_name)
                        if match:
                            forecast_horizon = int(match.group(1))
                        else:
                            print(f"Warning: Could not extract forecast horizon from {file_name}")
                            continue

                        try:
                            with open(file_path, 'r') as f:
                                error_data = {}
                                for line in f:
                                    metric, value = line.strip().split(': ')
                                    if re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', value):
                                        error_data[metric] = float(value)
                                    else:
                                        error_data[metric] = value

                                combined_data = metadata.copy()
                                combined_data['prediction_method'] = prediction_method
                                combined_data['forecast_horizon'] = forecast_horizon
                                combined_data.update(error_data)

                                all_results.append(combined_data)

                        except Exception as e:
                            print(f"Error reading or processing file {file_path}: {e}")
                            continue

    if not all_results:
        print("No error files found.")
        return None

    return pd.DataFrame(all_results)


model_dir = '../model'
results_df = analyze_error_metrics(model_dir)
if results_df is None:
    print("No results to process. Exiting.")  # More descriptive exit
    exit()
os.makedirs(f'errors/{model_dir}', exist_ok=True)
results_df.to_csv(f'errors/{model_dir}/all_errors.csv', index=False)

model_dir = '../model_attention'
results_df = analyze_error_metrics(model_dir)
if results_df is None:
    print("No results to process. Exiting.")  # More descriptive exit
    exit()
os.makedirs(f'errors/{model_dir}', exist_ok=True)
results_df.to_csv(f'errors/{model_dir}/all_errors.csv', index=False)





#
#
#
# results_df['wf_id'] = results_df['wf_id'].astype(int)
#
# # --- Modified Grouping and Aggregation ---
# # Dynamically determine which metrics to aggregate based on their presence in the DataFrame
# metrics_to_aggregate = ['MAE', 'RMSE', 'nMAE', 'nRMSE', 'R^2']  # Base metrics
# avg_metrics = [metric + '_avg' for metric in metrics_to_aggregate if metric + '_avg' in results_df.columns]
# overall_metrics = [metric + '_overall' for metric in metrics_to_aggregate if metric + '_overall' in results_df.columns]
#
# # Combine all metrics to aggregate
# all_metrics_to_aggregate = avg_metrics + overall_metrics
# # Add single-step metrics if they exist, handle cases where forecast_horizon is always > 1.
# for metric in metrics_to_aggregate:
#     if metric in results_df.columns:
#         all_metrics_to_aggregate.append(metric)
#
#
# # Perform the groupby and aggregation, only if there are metrics to aggregate.
# if all_metrics_to_aggregate:
#     grouped_df = results_df.groupby(['wf_id', 'prediction_method', 'forecast_horizon'])[
#         all_metrics_to_aggregate
#     ].mean().reset_index()
#     print("\nGrouped Results:")
#     print(grouped_df)
#     grouped_df.to_csv('errors/grouped_errors.csv', index=False)
# else:
#     print("No relevant metrics found for aggregation.")
#     grouped_df = None # set to None, so the next part doesn't fail.
#
#
# # --- Best Hyperparameter Selection (Conditional) ---
# # Proceed only if grouped_df exists and is not empty
# if grouped_df is not None and not grouped_df.empty :
#     best_hyperparameters = []
#     for wf_id in grouped_df['wf_id'].unique():
#         for method in grouped_df['prediction_method'].unique():
#             for horizon in grouped_df['forecast_horizon'].unique():
#                 subset = results_df[(results_df['wf_id'] == wf_id) &
#                                     (results_df['prediction_method'] == method) &
#                                     (results_df['forecast_horizon'] == horizon)]
#
#                 if not subset.empty:
#                     # Determine the best metric to use (nMAE_avg if it exists, otherwise nMAE)
#                     best_metric = 'nMAE_avg' if 'nMAE_avg' in subset.columns else 'nMAE' if 'nMAE' in subset.columns else None
#
#                     if best_metric: #proceed only if best_metric is not None
#                         #  Check for all NaN values *before* calling idxmin
#                         if subset[best_metric].isnull().all():
#                             print(f"All values for {best_metric} are NaN for wf_id={wf_id}, method={method}, horizon={horizon}. Skipping.")
#                             continue #skip this
#
#                         min_index = subset[best_metric].idxmin()
#                         min_value = subset[best_metric][min_index]
#                         candidates = subset[subset[best_metric] == min_value]
#                         # Use the corresponding RMSE metric as a tie-breaker
#                         tie_breaker_metric = 'nRMSE_avg' if 'nRMSE_avg' in candidates.columns else 'nRMSE' if 'nRMSE' in candidates.columns else None
#
#                         if tie_breaker_metric:
#                             best_row = candidates.loc[candidates[tie_breaker_metric].idxmin()].to_dict()
#                             best_hyperparameters.append(best_row)
#                         else: #if there is no tie_breaker_metric, just take the first one
#                             best_hyperparameters.append(candidates.iloc[0].to_dict())
#
#     if best_hyperparameters: #check if list is not empty
#         best_hyperparameters_df = pd.DataFrame(best_hyperparameters)
#         print("\nBest Hyperparameters:")
#         print(best_hyperparameters_df)
#         best_hyperparameters_df.to_csv('errors/best_hyperparameters.csv', index=False)
#     else:
#         print("No best hyperparameters found.")
#         best_hyperparameters_df = None #set to none for next part.
#
# else:
#     print("No grouped data available to determine best hyperparameters.")
#     best_hyperparameters_df = None #set to none for next part
#
#     # --- Summary Table (Conditional) ---
#     #Proceed only if best_hyperparameters_df exists
# if best_hyperparameters_df is not None and not best_hyperparameters_df.empty:
#     # Determine which metrics to include in the summary (check if _avg exists)
#     metrics_for_summary = ['nMAE', 'nRMSE', 'R^2']
#     avg_metrics_summary = [metric + '_avg' for metric in metrics_for_summary if metric + '_avg' in best_hyperparameters_df.columns]
#     #add the non avg metrics if they exist
#     for metric in metrics_for_summary:
#         if metric in best_hyperparameters_df.columns:
#             avg_metrics_summary.append(metric)
#
#
#     summary_table = best_hyperparameters_df.groupby(['wf_id', 'prediction_method', 'forecast_horizon'])[
#         avg_metrics_summary].first().reset_index()
#     print("\nSummary Table:")
#     print(summary_table)
#     summary_table.to_csv('errors/summary_table.csv', index=False)
# else:
#     print("No best hyperparameter data available to create a summary table.")