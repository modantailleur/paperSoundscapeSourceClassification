import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import main_doce
import pandas as pd
from textwrap import wrap
import numpy as np
import argparse

np.random.seed(0)

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def get_correlations(dataset, classifier, idx_tvb=[327, 0, 112]):

    deep = "False"
    plan = "reference"

    selector_pred = {"deep":deep, "step":"compute", "classifier":classifier, "dataset":dataset}
    selector_gt = {"dataset":dataset}

    (data_pred, settings_pred_total, header_pred_total) = main_doce.experiment.get_output(
    output = 'detection_mean',
    selector = selector_pred,
    path = "detection_mean",
    plan = plan
    )

    (data_gt, settings_gt_total, header_gt_total) = main_doce.experiment.get_output(
    output = 'groundtruth',
    selector = selector_gt,
    path = "groundtruth",
    plan = "groundtruth"
    )

    (data_fname_pred, settings_fname_pred_total, header_fname_pred_total) = main_doce.experiment.get_output(
    output = 'fname',
    selector = selector_pred,
    path = "detection_mean",
    plan = plan
    )

    (data_fname_gt, settings_fname_gt_total, header_fname_gt_total) = main_doce.experiment.get_output(
    output = 'fname',
    selector = selector_gt,
    path = "groundtruth",
    plan = "groundtruth"
    )

    for idx, (pred, gt, fname_pred, fname_gt) in enumerate(zip(data_pred, data_gt, data_fname_pred, data_fname_gt)):

    # ax.fill_between(y1='min_eval', y2='max_eval', data=df,
    #               color=mpl.colors.to_rgba('brown', 0.15))
        pred_list = pred
        gt_list = gt
        fname_scores = fname_pred
        # fname_scores = [item.split("_detection_")[1] for item in fname_scores]
        fname = fname_gt

        df_gt = pd.DataFrame(gt, columns=['t_gt', 'v_gt', 'b_gt'], index=fname_gt)
        df_pred = pd.DataFrame(pred, index=fname_scores)
        df_gt = df_gt.sort_index()
        df_pred = df_pred.sort_index()
        common_indices = df_gt.index.intersection(df_pred.index)
        df_pred = df_pred.loc[common_indices]
        df_pred_arr = df_pred.to_numpy()

        correlation_tables = []
        for k in range(1000):
            #noisy predictions
            df_gt_temp = df_gt.copy()
            std_dev = 0.1
            data = df_gt_temp.to_numpy()
            noise = np.random.normal(loc=0, scale=std_dev, size=data.shape)
            noisy_data = data + noise
            noisy_data = np.clip(noisy_data, 0, 1)
            df_gt_temp[['t_gt', 'v_gt', 'b_gt']] = noisy_data

            df_gt_arr = df_gt_temp.to_numpy()

            correlation_table = corr2_coeff(df_pred_arr.T, df_gt_arr.T)
            correlation_tables.append(correlation_table)

        correlation_tables = np.array(correlation_tables)
        std_result = np.std(correlation_tables, axis=0)
        df_gt_arr = df_gt.to_numpy()
        mean_result = corr2_coeff(df_pred_arr.T, df_gt_arr.T)

        # Create a dictionary to store the data
        data_dict = {
            'mean_t': mean_result[:, 0],
            'std_t': std_result[:, 0],
            'mean_v': mean_result[:, 1],
            'std_v': std_result[:, 1],
            'mean_b': mean_result[:, 2],
            'std_b': std_result[:, 2]
        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data_dict)

        if 'PANN' in classifier:
            # Load the Excel file 'sub_classes.xlsx' into a DataFrame
            file_path = './utils/sub_classes.xlsx'
            df_index = pd.read_excel(file_path, usecols=[0])  # Use the first column as the index
            df.set_index(df_index.iloc[:, 0], inplace=True)
        if classifier == 'TFSD':
            df.index = ['laeq', 'tfsd_mid', 'tfsd_high']

        ##################
        #### add corr with std to df

        df['corr_t'] = df.apply(lambda row: f"{row['mean_t']:.2f} +/- {2*row['std_t']:.2f}", axis=1)
        df['corr_v'] = df.apply(lambda row: f"{row['mean_v']:.2f} +/- {2*row['std_v']:.2f}", axis=1)
        df['corr_b'] = df.apply(lambda row: f"{row['mean_b']:.2f} +/- {2*row['std_b']:.2f}", axis=1)

        if classifier == 'TFSD':
            t = df.at['laeq', 'corr_t']
            v = df.at['tfsd_mid', 'corr_v']
            b = df.at['tfsd_high', 'corr_b']
        elif "PANN" in classifier:
            t = df.at[df_index.iloc[idx_tvb[0], 0], 'corr_t']
            v = df.at[df_index.iloc[idx_tvb[1], 0], 'corr_v']
            b = df.at[df_index.iloc[idx_tvb[2], 0], 'corr_b']
        elif classifier == "felix":
            t = df.at[0, 'corr_t']
            v = df.at[1, 'corr_v']
            b = df.at[2, 'corr_b']

        return([t,v,b])

def main(config):
    dataset = config.dataset
    felix_corr = get_correlations(dataset, 'felix')
    transcoder_corr = get_correlations(dataset, 'CNN-PINV-PANN')
    pann_corr = get_correlations(dataset, 'PANN')
    # Create a new DataFrame with the desired structure
    result_df = pd.DataFrame({
        'Annotation' : ['Traffic', 'Voices', 'Birds'],
        'CNN-train-synth' : felix_corr,
        'PANN-1/3oct' : transcoder_corr,
        'PANN' : pann_corr

    })

    latex_table = result_df.style.hide(axis="index").to_latex()
    
    tex_file_path = './figures/correlation_table_' + dataset + '.tex'

    # You can save the LaTeX table to a file
    with open(tex_file_path, 'w') as f:
        f.write(latex_table)

    print(result_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the result tables for GRAFIC and Lorient1k datasets')
    parser.add_argument('--dataset', type=str, default="Grafic",
                        help="The dataset on which to generate the result table")
    config = parser.parse_args()
    main(config)