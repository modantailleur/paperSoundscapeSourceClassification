import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import main_doce
import pandas as pd
from textwrap import wrap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

deep = "False"
plan = "reference"

selector_pred = {"deep":deep, "step":"compute"}

(data_pred, settings_pred_total, header_pred_total) = main_doce.experiment.get_output(
  output = 'detection_mean',
  selector = selector_pred,
  path = "detection_mean",
  plan = plan
  )

(data_fname_pred, settings_fname_pred_total, header_fname_pred_total) = main_doce.experiment.get_output(
  output = 'fname',
  selector = selector_pred,
  path = "detection_mean",
  plan = "reference"
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

for idx, (pred, setting_pred_total, fname_pred, setting_fname_pred_total) in enumerate(zip(data_pred, settings_pred_total, data_fname_pred, settings_fname_pred_total)):
    setting = [pair.split('=') for pair in setting_pred_total.split(', ')]
    setting = dict(setting)

    print('XXXXXXXXXXXX')
    print(setting)

    dataset = setting["dataset"]
    classifier = setting["classifier"]
    selector_gt = {"dataset":dataset}
    (data_gt, settings_gt_total, header_gt_total) = main_doce.experiment.get_output(
      output = 'groundtruth',
      selector = selector_gt,
      path = "groundtruth",
      plan = "groundtruth"
      )

    (data_fname_gt, settings_fname_gt_total, header_fname_gt_total) = main_doce.experiment.get_output(
      output = 'fname',
      selector = selector_gt,
      path = "groundtruth",
      plan = "groundtruth"
      )
    
    gt = data_gt[0]
    fname_gt = data_fname_gt[0]

    if 'PANN' in classifier:
      if dataset == 'Lorient1k':
          idx_classes = [327, 0, 112]
          # classes that are the best for CNN-PINV-PANN
          # idx_classes = [300, 0, 112]
      elif dataset == 'Grafic':
          idx_classes = [327, 0, 112]
          # classes that are the best for CNN-PINV-PANN
          # idx_classes = [300, 0, 112]
      else:
          idx_classes = [327, 0, 112]
          # classes that are the best for CNN-PINV-PANN
          # idx_classes = [300, 0, 112]
    elif classifier == 'TFSD':
        idx_classes = [0, 1, 2]
    elif classifier == 'felix':
        idx_classes = [0, 1, 2]
    
    pred = pred[:, idx_classes]

    df_gt = pd.DataFrame(gt, columns=['t_gt', 'v_gt', 'b_gt'], index=fname_gt)
    df_pred = pd.DataFrame(pred, columns=['t_pred', 'v_pred', 'b_pred'], index=fname_pred)

    if 'PANN' in classifier:
      df_pred = df_pred.clip(0, 1)

    result_df = pd.concat([df_gt, df_pred], axis=1)
    result_df = result_df.dropna()

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(result_df)

    gt = result_df[['t_gt', 'v_gt', 'b_gt']].to_numpy()
    pred = result_df[['t_pred', 'v_pred', 'b_pred']].to_numpy()

    correlation_table = corr2_coeff(pred.T, gt.T)

    print('REGULAR CORRELATION TABLE')
    print(correlation_table)

  
    # Alpha transparency for better visibility of overlapping points --> lower for larger datasets
    if not dataset in ['Grafic', 'Lorient1k', 'Aumilab-MT']: 
        alpha_value = 0.1
    else:
        alpha_value = 0.7

    label_dict = {
       't_gt': 'traffic annotation',
       'v_gt': 'voices annotation',
       'b_gt': 'birds annotation',
       't_pred': 'traffic prediction',
       'v_pred': 'voices prediction',
       'b_pred': 'birds prediction',
    }

    titles = ['traffic prediction vs annotation', 'voices prediction vs annotation ', 'birds prediction vs annotation']
    for y_column, x_column, title in [('t_gt', 't_pred', 'traffic'), ('v_gt', 'v_pred', 'voices'), ('b_gt', 'b_pred', 'birds')]:
        x = result_df[x_column].to_numpy()
        y = result_df[y_column].to_numpy()
        cor = pearsonr(x, y).statistic
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, marker='x', alpha=0.7, color='black', linewidths=3)

        # Add regression line
        # x_l = x.reshape(-1, 1)
        # y_l = y.reshape(-1, 1)
        # model = LinearRegression(fit_intercept=False)
        # model.fit(y_l, x_l)
        # slope = model.coef_[0]
        # x_p = np.linspace(0, 1, len(x_l))
        # y_p = (1/slope) * x_p
        # plt.plot(x_p, y_p, label=f'y = {slope}*x', color='blue')
        # plt.legend()

        # Remove ticks on both x and y axes
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks

        # Save the individual plots
        plt.savefig(f'figures/correlation_plot_{dataset}_{classifier}_{title}.png')
