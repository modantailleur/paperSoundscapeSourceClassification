import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

import main_doce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr

classifier = 'CNN-PINV-PANN'
idx_classes = [300, 0, 111]

if classifier == 'TFSD':
    idx_classes = [0, 1, 2]

deep = "False"
plan = "reference"

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def what_happens_with_zeros(x, y, threshold=None):
    if threshold is None:
      idx_zero = np.where(x == 0)[0]
    else:
      idx_zero = np.where(x <= threshold)[0]
    new_x = x[idx_zero]
    new_y = y[idx_zero]
    count_below_threshold = np.count_nonzero(y < 0.05)
    print('MEAN ZEROS')
    print(np.mean(new_y))
    print(np.mean(new_x))
    print('BELOW 0.05')
    print(f'{count_below_threshold} / {len(new_y)} / {len(y)}')


def get_df(data_pred, data_gt, data_fname_pred, data_fname_gt, classifier, dataset):
    for idx, (pred, gt, fname_pred, fname_gt) in enumerate(zip(data_pred, data_gt, data_fname_pred, data_fname_gt)):

    # ax.fill_between(y1='min_eval', y2='max_eval', data=df,
    #               color=mpl.colors.to_rgba('brown', 0.15))
        pred_list = pred
        gt_list = gt
        fname_scores = fname_pred
        # fname_scores = [item.split("_detection_")[1] for item in fname_scores]
        fname = fname_gt
        
        pred = pred[:, idx_classes]
        # pred = np.mean(pred, axis=0)

        #MT: TEST
        # pred = np.exp(pred)/np.e

        df_gt = pd.DataFrame(gt, columns=['t_gt', 'v_gt', 'b_gt'], index=fname_gt)
        df_pred = pd.DataFrame(pred, columns=['t_pred', 'v_pred', 'b_pred'], index=fname_scores)

        #MT: TEST
        df_pred['t_pred'] = df_pred['t_pred']
        df_pred['v_pred'] = df_pred['v_pred']
        df_pred['b_pred'] = df_pred['b_pred']

        if 'PANN' in classifier:
            df_pred = df_pred.clip(0, 1)

        result_df = pd.concat([df_gt, df_pred], axis=1)
        result_df = result_df.dropna()

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #     print(result_df)

        gt_list = result_df[['t_gt', 'v_gt', 'b_gt']].to_numpy()
        pred_list = result_df[['t_pred', 'v_pred', 'b_pred']].to_numpy()

        return(result_df)
    

dataset = 'Grafic'
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
  plan = "reference"
  )

(data_fname_gt, settings_fname_gt_total, header_fname_gt_total) = main_doce.experiment.get_output(
  output = 'fname',
  selector = selector_gt,
  path = "groundtruth",
  plan = "groundtruth"
  )

df_grafic = get_df(data_pred, data_gt, data_fname_pred, data_fname_gt, classifier, dataset)

dataset = 'Lorient1k'

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
  plan = "reference"
  )

(data_fname_gt, settings_fname_gt_total, header_fname_gt_total) = main_doce.experiment.get_output(
  output = 'fname',
  selector = selector_gt,
  path = "groundtruth",
  plan = "groundtruth"
  )

df_lorient1k = get_df(data_pred, data_gt, data_fname_pred, data_fname_gt, classifier, dataset)

df = pd.concat([df_grafic, df_lorient1k], axis=0)

gt_list = df[['t_gt', 'v_gt', 'b_gt']].to_numpy()
pred_list = df[['t_pred', 'v_pred', 'b_pred']].to_numpy()

correlation_table = corr2_coeff(pred_list.T, gt_list.T)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Alpha transparency for better visibility of overlapping points
alpha_value = 0.7

if not dataset in ['Grafic', 'Lorient1k', 'Aumilab-MT']: 
    alpha_value = 0.1
    # result_df = result_df.sample(n=2000, random_state=3)

label_dict = {
    't_gt': 'traffic annotation',
    'v_gt': 'voices annotation',
    'b_gt': 'birds annotation',
    't_pred': 'traffic prediction',
    'v_pred': 'voices prediction',
    'b_pred': 'birds prediction',
}

titles = ['traffic prediction vs annotation', 'voices prediction vs annotation ', 'birds prediction vs annotation']
# Plot with alpha transparency and crosses
for idx, (ax, tvb_class, x_column, y_column) in enumerate(zip(axes, ['traffic', 'voices', 'birds'], ['t_gt', 'v_gt', 'b_gt'], ['t_pred', 'v_pred', 'b_pred'])):
    x = df[x_column].to_numpy()
    y = df[y_column].to_numpy()
    cor = pearsonr(x, y).statistic
    ax.scatter(x, y, marker='x', alpha=alpha_value)


    ax.set_xlabel(label_dict[x_column])
    ax.set_ylabel(label_dict[y_column])
    ax.set_title(titles[idx])

    #line plot
    x_l = x.reshape(-1, 1)
    y_l = y.reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(y_l, x_l)
    slope = model.coef_[0]
    x_p = np.linspace(0, 1, len(x_l))  # Adjust the range and number of points as needed
    y_p = (1/slope) * x_p

    print('MULTIPLICATION FACTORS')
    print(tvb_class)
    print(slope)


# Adjust layout
plt.tight_layout()

# Show the plots
# plt.show()
