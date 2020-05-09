'''
evaluation related functions
'''

__author__ = 'Oguzhan Gencoglu'

from metrics import (accuracy, precision, recall, auc, f1, false_negative_rate,
                     false_positive_rate, group_false_negative_rates,
                     group_false_positive_rates, group_recalls, group_mccs,
                     group_precisions, false_positive_equality_diff,
                     false_negative_equality_diff, mcc)
import pandas as pd
from mlxtend.evaluate import mcnemar_table, mcnemar
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_sm


def eval_report(true_labels, predicted_labels,
                predicted_probabilities, groups):
    '''
    [true_labels]             : list or python 1D array of zeros or ones
    [predicted_labels]        : list or python 1D array of zeros or ones
    [predicted_probabilities] : list or python 1D array of floats - range [0,1]
    [groups]                  : 2D numpy array of binary values
    '''

    print('Evaluation on test data:')
    print('\tAUC = {:.4f}'.format(auc(true_labels, predicted_probabilities)))
    print('\tAccuracy = {:.2f} %'.format(
                                accuracy(true_labels, predicted_labels) * 100))
    print('\tMatthews correlation coefficient = {:.4f}'.format(
                                           mcc(true_labels, predicted_labels)))
    print('\t\tGroup-specific MCCs = {}'.format(
                            group_mccs(true_labels, predicted_labels, groups)))
    print('\tf1 score = {:.4f}'.format(
                                            f1(true_labels, predicted_labels)))
    print('\tPrecision = {:.4f}'.format(
                                     precision(true_labels, predicted_labels)))
    print('\t\tGroup-specific precisions = {}'.format(
                      group_precisions(true_labels, predicted_labels, groups)))
    print('\tRecall (sensitivity) = {:.4f}'.format(
                                        recall(true_labels, predicted_labels)))
    print('\t\tGroup-specific recalls = {}'.format(
                         group_recalls(true_labels, predicted_labels, groups)))
    print('\tFalse negative (miss) rate = {:.4f} %'.format(
                     100 * false_negative_rate(true_labels, predicted_labels)))
    print('\t\tGroup-specific false negative rates = {} %'.format(
      100 * group_false_negative_rates(true_labels, predicted_labels, groups)))
    print('\tFalse positive (false alarm) rate = {:.4f} %'.format(
                     100 * false_positive_rate(true_labels, predicted_labels)))
    print('\t\tGroup-specific false positive rates = {} %'.format(
      100 * group_false_positive_rates(true_labels, predicted_labels, groups)))
    fned = false_negative_equality_diff(true_labels, predicted_labels, groups)
    fped = false_positive_equality_diff(true_labels, predicted_labels, groups)
    print('\tFalse negative equality difference (per group) = {:.4f} | '
          'total FNED = {:.4f}'.format(
                                      fned / groups.shape[1], fned)
          )
    print('\tFalse positive equality difference (per group) = {:.4f} | '
          'total FPED = {:.4f}'.format(
                                      fped / groups.shape[1], fped)
          )
    print('\tTotal equality difference (bias) = {:.4f}'.format((fned + fped)))

    return None


def mcnemar_test(labels, model1_preds, model1_name, model2_preds, model2_name):
    '''
    Performs McNemar's test for paired nominal data
    Ref: McNemar, Quinn, 1947. "Note on the sampling error of the difference
         between correlated proportions or percentages".
         Psychometrika. 12 (2): 153–157.

    [labels]       : list or 1D numpy array, correct labels (0 or 1)
    [model1_preds] : list or 1D numpy array, predictions of the first model
    [model1_name]  : str, name of the first model
    [model2_preds] : list or 1D numpy array, predictions of the second model
    [model2_name]  : str, name of the second model
    '''

    contigency_table = mcnemar_table(y_target=labels.ravel(),
                                     y_model1=model1_preds.ravel(),
                                     y_model2=model2_preds.ravel())
    contigency_df = pd.DataFrame(contigency_table,
                                 columns=['{} correct'.format(model1_name),
                                          '{} wrong'.format(model1_name)],
                                 index=['{} correct'.format(model2_name),
                                        '{} wrong'.format(model2_name)])
    print(contigency_df)

    # 'mlxtend' library implementation
    print("\n'mlxtend' library implementation")
    chi2, p = mcnemar(ary=contigency_table, exact=True)
    print('\tchi-squared = {}'.format(chi2))
    print('\tp-value = {}'.format(p))

    # 'statsmodels' library implementation
    print("\n'statsmodels' library implementation")
    print(mcnemar_sm(contigency_table, exact=True))

    return None
