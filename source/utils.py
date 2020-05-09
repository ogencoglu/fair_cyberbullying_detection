'''
utility functions
'''

__author__ = 'Oguzhan Gencoglu'

import os
from os.path import join
from os.path import abspath
from copy import deepcopy
import itertools

import json
import numpy as np
import pandas as pd

from configs import config as cf


def is_available(filename):
    '''
    [filename] : str
    '''

    return os.path.isfile(filename)


def save_embeddings(embeddings, dataset):
    '''
    [embeddings] : 2D numpy array
    [dataset]    : str
    '''

    if dataset == 'wiki':
        save_path = cf.wiki_embeddings_path
    elif dataset == 'jigsaw':
        save_path = cf.jigsaw_embeddings_path
    elif dataset == 'twitter':
        save_path = cf.twitter_embeddings_path
    elif dataset == 'gab':
        save_path = cf.gab_embeddings_path
    else:
        raise ValueError(
                '"dataset" can be one of "wiki", "jigsaw", "twitter", "gab"')

    if not is_available(save_path):
        np.save(save_path, embeddings)
        print('\tEmbeddings saved successfully.')
    else:
        print('\tEmbeddings already exist.')

    return None


def load_embeddings(dataset):
    '''
    [dataset] : str
    returns a 2D numpy array of shape (n_observations, n_embedding_dims)
    '''

    if dataset == 'wiki':
        load_path = cf.wiki_embeddings_path
    elif dataset == 'jigsaw':
        load_path = cf.jigsaw_embeddings_path
    elif dataset == 'twitter':
        load_path = cf.twitter_embeddings_path
    elif dataset == 'gab':
        load_path = cf.gab_embeddings_path
    else:
        raise ValueError(
                '"dataset" can be one of "wiki", "jigsaw", "twitter", "gab"')

    if is_available(load_path):
        embeddings = np.load(load_path)
        print('\tEmbeddings (shape={}) loaded successfully.'.format(
                                                            embeddings.shape)
              )
    else:
        print('\tSaved embeddings are not available.')
        return None

    return embeddings


# _____________ Wiki dataset related _____________
def read_wiki_data(mode):
    '''
    [mode] : 'toxicity' , 'aggression' or 'attack'
    returns pandas dataframes
    '''

    assert mode in ['toxicity', 'aggression', 'attack']

    data_path = abspath(
                join(cf.DATA_DIR_WIKI,
                     '{}_annotated_comments.tsv'.format(mode))
                     )
    annots_path = abspath(
                          join(cf.DATA_DIR_WIKI,
                               '{}_annotations.tsv'.format(mode))
                               )
    data = pd.read_csv(data_path, sep='\t')
    data = data[data.comment != '']
    annots = pd.read_csv(annots_path, sep='\t')
    if mode == 'attack':
        annots['attack_score'] = [np.nan] * annots.shape[0]
    print('Data shape _{}_={}, annotations shape={}'.format(
                                        mode, data.shape, annots.shape))

    return (data, annots)


def clean_wiki_data(data_annots_pair, mode):
    '''
    [data_annots_pair] : tuple of pandas dataframes
    [mode] : 'toxicity' , 'aggression' or 'attack'
    '''

    def remove_nl_token(text):
        text = text.replace('NEWLINE_TOKEN', '')

        return text

    # seperate data and annotations
    data, annots = data_annots_pair

    # make a deepcopy
    data_cleaned, annots_cleaned = deepcopy(data), deepcopy(annots)
    data_cleaned['rev_id'] = data_cleaned['rev_id'].astype(int)

    # drop irrelevant columns
    data_cleaned.drop(['logged_in', 'ns', 'sample', 'split'],
                      axis=1, inplace=True)
    annots_cleaned.drop(['worker_id', '{}_score'.format(mode)],
                        axis=1, inplace=True)
    if mode == 'attack':
        annots_cleaned.drop(['quoting_attack', 'recipient_attack',
                            'third_party_attack', 'other_attack'],
                            axis=1, inplace=True)

    # clean comments
    data_cleaned['comment'] = data_cleaned['comment'].apply(remove_nl_token)

    # calculate average _mode_ from different annotators
    data_cleaned[mode] = list(annots_cleaned.groupby('rev_id').mean()[mode])

    # create target column
    data_cleaned['is_{}'.format(mode)] = data_cleaned[mode] == 0.0

    return data_cleaned


# _____________ Jigsaw dataset related _____________
def read_jigsaw_data():
    '''
    returns a pandas dataframe
    '''

    # read data and drop irrelevant attributes
    data_path = abspath(
                join(cf.DATA_DIR_JIGSAW, 'train.csv')
                )
    data = pd.read_csv(data_path,
                       usecols=[
                                'comment_text',
                                'target'
                                ] + cf.identity_keys_jigsaw)
    data.rename(columns={'comment_text': 'comment'}, inplace=True)
    data = data[data.comment != '']
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # create identity groups
    for k in cf.identity_keys_jigsaw:
        data[k] = data[k] >= cf.target_thres

    # create target column
    data['target'] = data['target'] >= cf.target_thres
    print('Data shape={}'.format(data.shape))

    return data


# _________________ Multilingual Twitter data related _________________
def read_twitter_data():
    '''
    returns a pandas dataframe
    '''
    def clean_stopwords(text, remove_strings=['rt user : ', 'lrt : ', 'rt : '],
                        stopwords=['user', 'url', 'hashtag', '…']):
        '''
        clean stopwords
        '''
        for r in remove_strings:
            text = text.replace(r, '')
        words = []
        stripped = text.split()
        for s in stripped:
            if s not in stopwords:
                words.append(s)
        return ' '.join(words)

    full_data = []
    for lang in cf.twitter_languages:  # loop through languages
        data_path = abspath(
                    join(cf.DATA_DIR_TWITTER, 'anonymize', lang, 'corpus.tsv')
                    )

        # read data line by line instead of pandas (see issue
        # https://github.com/xiaoleihuang/Multilingual_Fairness_LREC/issues/3)
        lines = []
        with open(data_path) as dfile:
            dfile.readline()
            for line in dfile:
                stripped = line.strip().split('\t')
                if len(stripped) != 11:
                    break
                else:
                    lines.append(stripped)
        data = pd.DataFrame(lines, columns=['tid', 'uid', 'comment', 'date',
                                            'gender', 'age', 'city', 'state',
                                            'country', 'ethnicity', 'target'])

        for k in cf.identity_dict_twitter.keys():  # create identity groups
            for j in cf.identity_dict_twitter[k]:
                data[j] = data[k] == j
        data.drop(['tid', 'uid', 'date', 'age', 'city', 'state',
                   'country', 'gender', 'ethnicity'],
                  axis=1,
                  inplace=True)
        data[lang] = True
        full_data.append(data)

    full_data = pd.concat(full_data)
    full_data.fillna(False, inplace=True)

    full_data['comment'] = full_data['comment'].apply(clean_stopwords)
    full_data = full_data[full_data.comment != '']
    full_data['target'] = full_data['target'].map(cf.twitter_label_mapping)
    full_data['target'] = full_data['target'].astype(int)

    # create combination of sex and race identities
    iden_list = list(cf.identity_dict_twitter.values())
    identity_combinations = list(itertools.product(iden_list[0], iden_list[1]))
    for i in identity_combinations:
        full_data['{}_{}'.format(i[1], i[0])] = np.logical_and(full_data[i[0]],
                                                               full_data[i[1]])
    print('Data shape={}'.format(full_data.shape))

    return full_data


# _________________ Gab data related _________________
def read_gab_data():
    '''
    returns a pandas dataframe
    '''

    data_path = abspath(
                join(cf.DATA_DIR_GAB, 'GabHateCorpus_annotations.tsv')
                )
    all_data = pd.read_csv(data_path, sep='\t')
    all_data.drop(['Hate', 'VO', 'EX', 'IM', 'Annotator'],
                  axis=1, inplace=True)
    all_data.fillna(0, inplace=True)
    comments = all_data.loc[all_data['ID'].
                            drop_duplicates().index][['ID', 'Text']]
    grouped = all_data.groupby(by='ID').mean().reset_index()
    data = pd.merge(comments, grouped, how='outer')
    data['target'] = np.logical_or(data['HD'] > cf.target_thres,
                                   data['CV'] > cf.target_thres)
    data[['REL', 'RAE', 'SXO', 'GEN',
          'IDL', 'NAT', 'POL', 'MPH']] = data[
                                              ['REL', 'RAE', 'SXO', 'GEN',
                                               'IDL', 'NAT', 'POL', 'MPH']
                                               ].astype(bool)
    data.drop(['ID', 'HD', 'CV'], axis=1, inplace=True)
    data.rename(columns={'Text': 'comment'}, inplace=True)
    data = data[data.comment != '']
    print('Data shape={}'.format(data.shape))

    return data


# _____________ Logging related functions _____________
def save_logs(errors, fped, fned, metrics, dict_name):
    '''
    [errors]    : list
    [fped]      : list
    [fned]      : list
    [metrics]   : list
    [dict_name] : str
    '''

    logs_dict = {'err': errors,
                 'fped': fped,
                 'fned': fned,
                 'met': metrics}
    logs_json = json.dumps(logs_dict)
    f = open('{}/{}.json'.format(cf.LOGS_DIR, dict_name), 'w')
    f.write(logs_json)
    f.close()

    return None


def load_logs(dict_name, return_dict=False):
    '''
    [dict_name]   : str
    [return_dict] : bool
    '''

    with open('{}/{}.json'.format(cf.LOGS_DIR, dict_name)) as logs_json:
        logs = json.load(logs_json)

    if return_dict:
        return logs
    else:
        errors = np.array(logs['err'])
        fped, fned = np.array(logs['fped']), np.array(logs['fned'])
        metrics = np.array(logs['met'])
        return errors, fped, fned, metrics
