'''
visualization functions
'''

__author__ = 'Oguzhan Gencoglu'

import matplotlib.pyplot as plt
import numpy as np

from metrics import (accuracy, false_positive_rate, false_negative_rate,
                     group_false_positive_rates, group_false_negative_rates)

plt.rcParams['font.sans-serif'] = "Liberation Sans"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 13
cb_palette = ['#58D68D', '#C39BD3', '#F7DC6F', '#85C1E9',
              "#EC7063", '#ff9900', '#8c8c8c']


def plot_perf(labels, predictions, groups, group_names, title=''):
    '''
    [labels]      : true labels
    [predictions] : predicted labels
    [groups]      : 2D numpy array of binary values
    [group_names] : str
    [title]       : str
    '''

    num_groups = groups.shape[1]

    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    group_fprs = 100 * group_false_positive_rates(labels, predictions, groups)
    ax[0].bar(range(num_groups + 1),
              np.append(group_fprs, 100 * false_positive_rate(labels,
                                                              predictions)),
              color=cb_palette)
    ax[0].set_xticks(range(num_groups + 1))
    ax[0].set_xticklabels(group_names + ['overall'])
    ax[0].set_xlabel("False Positive Rate (%)")
    ax[0].set_ylim([0, max(np.append(group_fprs,
                                     100 * false_positive_rate(labels,
                                                               predictions)))
                    + 1])
    ax[0].grid(axis='y', linestyle='--')

    group_fnrs = 100 * group_false_negative_rates(labels, predictions, groups)
    ax[1].bar(range(num_groups + 1),
              np.append(group_fnrs, 100 * false_negative_rate(labels,
                                                              predictions)),
              color=cb_palette)
    ax[1].set_xticks(range(num_groups + 1))
    ax[1].set_xticklabels(group_names + ['overall'])
    ax[1].set_xlabel("False Negative Rate (%)")
    ax[1].set_ylim([0, max(np.append(group_fnrs,
                                     100 * false_negative_rate(labels,
                                                               predictions)))
                    + 1])
    ax[1].grid(axis='y', linestyle='--')

    fig.suptitle('{} accuracy = {}%'.format(title,
                                            np.round(100 *
                                                     accuracy(labels,
                                                              predictions),
                                                     2)), x=0.5, y=1)

    return None
