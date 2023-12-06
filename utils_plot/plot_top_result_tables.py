import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import main_doce
import pandas as pd
import numpy as np

np.random.seed(0)

dataset = 'Lorient1k'
classifier = 'CNN-PINV-PANN'

coef_t = 1
coef_v = 1
coef_b = 1
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
  plan = "deep"
  )

(data_fname_gt, settings_fname_gt_total, header_fname_gt_total) = main_doce.experiment.get_output(
  output = 'fname',
  selector = selector_gt,
  path = "groundtruth",
  plan = "groundtruth"
  )

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

for idx, (pred, gt, fname_pred, fname_gt) in enumerate(zip(data_pred, data_gt, data_fname_pred, data_fname_gt)):

  # ax.fill_between(y1='min_eval', y2='max_eval', data=df,
  #               color=mpl.colors.to_rgba('brown', 0.15))
    pred_list = pred
    gt_list = gt
    fname_scores = fname_pred
    fname_scores = [item.split("_detection_")[1] for item in fname_scores]
    fname = fname_gt

    df_gt = pd.DataFrame(gt, columns=['t_gt', 'v_gt', 'b_gt'], index=fname_gt)
    df_pred = pd.DataFrame(pred, index=fname_scores)
    df_gt = df_gt.sort_index()
    df_pred = df_pred.sort_index()
    common_indices = df_gt.index.intersection(df_pred.index)
    df_pred = df_pred.loc[common_indices]
    df_pred_arr = df_pred.to_numpy()

    correlation_tables = []
    for k in range(100):
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

    #trying t-test here: only problem is that it depends on the number of correlation tables that we generate. If we generate a 1000, there is always a significant 
    #difference, so not sure it's the right thing to do
    # a1 = correlation_tables[:, 111, 2]
    # a2 = correlation_tables[:, 112, 2]
    # t_stat, p_value = stats.ttest_ind(a1, a2)
    # print(t_stat)
    # # Check the p-value to determine significance
    # if p_value < 0.05:  # Set your significance level (e.g., 0.05)
    #     print("There is a significant difference between the classifiers.")
    # else:
    #     print("There is no significant difference between the classifiers.")

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
        file_path = './utils/sub_classes.xlsx'
        df_index = pd.read_excel(file_path, usecols=[0])  # Use the first column as the index
        df.set_index(df_index.iloc[:, 0], inplace=True)
    if classifier == 'TFSD':
        df.index = ['laeq', 'tfsd_mid', 'tfsd_high']
    
    # csv_file_path = 'correlation_' + classifier + '_' + dataset + '.csv'
    # df.to_csv(csv_file_path)

    if 'PANN' in classifier:
        tvb = df.loc[['Vehicle', 'Speech', 'Bird vocalization, bird call, bird song']]
    else:
        tvb = df.copy()


    tvb['corr_t'] = tvb.apply(lambda row: f"{row['mean_t']:.2f} +/- {2*row['std_t']:.2f}", axis=1)
    tvb['corr_v'] = tvb.apply(lambda row: f"{row['mean_v']:.2f} +/- {2*row['std_v']:.2f}", axis=1)
    tvb['corr_b'] = tvb.apply(lambda row: f"{row['mean_b']:.2f} +/- {2*row['std_b']:.2f}", axis=1)
    tvb = tvb.iloc[:, 6:]

    print(tvb)

    ##################
    #### add corr with std to df

    df['corr_t'] = df.apply(lambda row: f"{row['mean_t']:.2f} +/- {2*row['std_t']:.2f}", axis=1)
    df['corr_v'] = df.apply(lambda row: f"{row['mean_v']:.2f} +/- {2*row['std_v']:.2f}", axis=1)
    df['corr_b'] = df.apply(lambda row: f"{row['mean_b']:.2f} +/- {2*row['std_b']:.2f}", axis=1)

    print(df)

    # ########
    ## TOP 10 FOR EACH T,V,B
    # # Sort the DataFrame by 'mean_t' and select the top 10 rows
    # top_df_t = df.sort_values(by='mean_t', ascending=False).head(15)
    # top_df_v = df.sort_values(by='mean_v', ascending=False).head(15)
    # top_df_b = df.sort_values(by='mean_b', ascending=False).head(15)


    # # Create a new DataFrame with the desired structure
    # result_df = pd.DataFrame({
    #     'Top 3 Traffic': top_df_t.index,
    #     'mean_t': top_df_t['corr_t'].to_numpy(),
    #     'Top 3 Voices': top_df_v.index,
    #     'mean_v': top_df_v['corr_v'].to_numpy(),
    #     'Top 3 Birds': top_df_b.index,
    #     'mean_b': top_df_b['corr_b'].to_numpy()
    # })

    # # Reset the index of the result DataFrame
    # result_df.reset_index(drop=True, inplace=True)
    # latex_table = result_df.to_latex(index=False)
    
    # tex_file_path = './figures/top_correlation_' + classifier + '_' + dataset + '.tex'

    # # # You can save the LaTeX table to a file
    # # with open(tex_file_path, 'w') as f:
    # #     f.write(latex_table)

    # # df.to_csv(csv_file_path)
    # print(result_df)

    #####################
    ## TOP x on same row for each tvb

    top = 10
    # Sort the DataFrame by 'mean_t' and select the top 10 rows
    top_df_t = df.sort_values(by='mean_t', ascending=False).head(top)
    top_df_v = df.sort_values(by='mean_v', ascending=False).head(top)
    top_df_b = df.sort_values(by='mean_b', ascending=False).head(top)


    # Create a new DataFrame with the desired structure
    result_df = pd.DataFrame({
        'Type' : ['Traffic' for k in range(top)] + ['Voices' for k in range(top)] + ['Birds' for k in range (top)],
        'Top 3': np.concatenate((top_df_t.index.to_numpy(), top_df_v.index.to_numpy(), top_df_b.index.to_numpy())),
        'mean': np.concatenate((top_df_t['corr_t'].to_numpy(), top_df_v['corr_v'].to_numpy(), top_df_b['corr_b'].to_numpy())),
    })

    # Reset the index of the result DataFrame
    result_df.reset_index(drop=True, inplace=True)
    latex_table = result_df.to_latex(index=False)
    
    tex_file_path = './figures/top_correlation_' + classifier + '_' + dataset + '.tex'

    # You can save the LaTeX table to a file
    with open(tex_file_path, 'w') as f:
        f.write(latex_table)

    # df.to_csv(csv_file_path)
    print(result_df)



    # slopes = []

    # for k in range(pred_list.shape[1]):
    #     pred_temp = pred_list[:, k]
    #     gt_list_temp = gt_list[:, k]
    #     pred_temp = pred_temp.reshape(-1, 1)
    #     gt_list_temp = gt_list_temp.reshape(-1, 1)
    #     model = LinearRegression(fit_intercept=False)
    #     model.fit(pred_temp, gt_list_temp)
    #     slope = model.coef_[0]
    #     slopes.append(slope)

    # correlation_table = corr2_coeff(pred_list.T, gt_list.T)


