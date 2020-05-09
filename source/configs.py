'''
configs & settings are defined in this file
'''

__author__ = 'Oguzhan Gencoglu'

from os.path import join
from os.path import abspath
from os.path import dirname
from os import pardir


class Config(object):

    # _________________ Common to all experiments _________________

    # directory paths
    CURRENT_DIR = abspath(dirname(__file__))
    ROOT_DIR = abspath(join(CURRENT_DIR, pardir))
    DATA_DIR = abspath(join(ROOT_DIR, 'data'))
    DATA_DIR_WIKI = abspath(join(DATA_DIR, 'wiki'))
    DATA_DIR_JIGSAW = abspath(join(DATA_DIR, 'jigsaw'))
    DATA_DIR_TWITTER = abspath(join(DATA_DIR, 'twitter'))
    DATA_DIR_GAB = abspath(join(DATA_DIR, 'gab'))
    MODELS_DIR = abspath(join(ROOT_DIR, 'models'))
    LOGS_DIR = abspath(join(ROOT_DIR, 'logs'))

    # pretrained model related params
    model_identifier = 'distiluse-base-multilingual-cased'
    bert_embed_dim = 512

    # determinism control
    random_state = 42
    train_size = 0.7
    val_test_ratio = 0.5  # 50% val and 50% test for non-training data

    # label related params
    target_thres = 0.5  # above this value is considered cyberbullying

    # model related params
    hyperparams = {
        'batch_size': 128,
        'patience': 20,
        'epochs': 75,
        'lr': 0.0005,
        'lr_constraints': 0.005,
    }

    # _________________ Wiki data experiment _________________

    # representation, model & log output names
    wiki_embeddings_name = 'wiki_sentence_embeddings.npy'
    wiki_embeddings_path = abspath(
                                join(DATA_DIR_WIKI, wiki_embeddings_name)
                                     )
    wiki_plain_model_name = 'wiki_plain'
    wiki_constrained_model_name = 'wiki_constrained'
    wiki_log_name = 'wiki_log'

    # wiki data related params
    wiki_modes = ['toxicity', 'aggression', 'attack']
    wiki_identities = ['recent', 'older']
    num_identities_wiki = len(wiki_identities)

    # label related params
    wiki_year_thres = 2015

    # constraints
    wiki_allowed_fnr_deviation = 0.005
    wiki_allowed_fpr_deviation = 0.005

    # _________________ Jigsaw data experiment _________________

    # representation, model & log output names
    jigsaw_embeddings_name = 'jigsaw_sentence_embeddings.npy'
    jigsaw_embeddings_path = abspath(
                                join(DATA_DIR_JIGSAW, jigsaw_embeddings_name)
                                     )
    jigsaw_plain_model_name = 'jigsaw_plain'
    jigsaw_constrained_model_name = 'jigsaw_constrained'
    jigsaw_log_name = 'jigsaw_log'

    # jigsaw data related params
    identity_keys_jigsaw = ['male', 'female']
    num_identities_jigsaw = len(identity_keys_jigsaw)

    # constraints
    jigsaw_allowed_fnr_deviation = 0.02
    jigsaw_allowed_fpr_deviation = 0.03

    # _________________ Multilingual Twitter data experiment _________________

    # representation, model & log output names
    twitter_embeddings_name = 'twitter_sentence_embeddings.npy'
    twitter_embeddings_path = abspath(
                                join(DATA_DIR_TWITTER, twitter_embeddings_name)
                                     )
    twitter_plain_model_name = 'twitter_plain'
    twitter_constrained_model_name = 'twitter_constrained'
    twitter_log_name = 'twitter_log'

    # twitter data related params
    twitter_languages = ['English', 'Italian', 'Polish',
                         'Portuguese', 'Spanish']

    twitter_label_mapping = {'neither': 0, 'normal': '0', 'no': 0, 'spam': 0,
                             'link': 0, 'abusive': 1, 'sexism': 1,
                             'hateful': 1, 'racism': 1, 'strong': 1, 'weak': 1,
                             0: 0, 1: 1, '0': 0, '1': 1}
    identity_dict_twitter = {
            'gender': ['male', 'female'],
            'ethnicity': ['black', 'white', 'asian', 'hispanic'],
                     }
    identity_keys_twitter = twitter_languages
    num_identities_twitter = len(identity_keys_twitter)

    # constraints
    twitter_allowed_fnr_deviation = 0.15
    twitter_allowed_fpr_deviation = 0.1

    # _________________ Gab data experiment _________________

    # representation, model & log output names
    gab_embeddings_name = 'gab_sentence_embeddings.npy'
    gab_embeddings_path = abspath(
                                join(DATA_DIR_GAB, gab_embeddings_name)
                                     )
    gab_plain_model_name = 'gab_plain'
    gab_constrained_model_name = 'gab_constrained'
    gab_log_name = 'gab_log'

    # gab data related params
    identity_keys_gab = ['REL', 'RAE', 'NAT']
    num_identities_gab = len(identity_keys_gab)

    # constraints
    gab_allowed_fnr_deviation = 0.1
    gab_allowed_fpr_deviation = 0.15


config = Config()
