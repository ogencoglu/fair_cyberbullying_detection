'''
model training
'''

__author__ = 'Oguzhan Gencoglu'

import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint

from configs import config as cf


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def train_model(model, train, train_labels, val,
                val_labels, model_name='model'):
    '''
    train a given model with a set of data
    '''

    optimizer = tf.keras.optimizers.Adam(cf.hyperparams['lr'])
    loss = lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true,
                                                                      y_pred,
                                                                      from_logits=True)

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', f1_m],
                  )
    model.fit(x=train,
              y=train_labels,
              shuffle=True,
              sample_weight=np.ones(
                                    train_labels.ravel().shape
                                    ) + train_labels.ravel()*np.sum(
                                    train_labels == 0) / np.sum(
                                    train_labels == 1),
              validation_data=(val, val_labels),
              callbacks=[
                ModelCheckpoint('{}/{}.h5'.format(cf.MODELS_DIR, model_name),
                                monitor='val_f1_m',
                                mode='max',
                                verbose=1,
                                save_best_only=True),
                EarlyStopping(monitor='val_f1_m',
                              mode='max',
                              verbose=1,
                              patience=cf.hyperparams['patience'])],
              batch_size=cf.hyperparams['batch_size'],
              epochs=cf.hyperparams['epochs'],
              )

    return model


# utils for constrained training
def create_tensors(num_groups):
    '''
    create empty tensors for constrained training
    '''

    # Create feature, label and group tensors
    feat_batch_shape = (cf.hyperparams['batch_size'],
                        cf.bert_embed_dim)
    feat_tensor = tf.Variable(
                              np.zeros(feat_batch_shape,
                                       dtype='float32'),
                              name='features')
    feat_tensor_group = tf.Variable(
                                    np.zeros(feat_batch_shape,
                                             dtype='float32'),
                                    name='features_group')

    label_batch_shape = (cf.hyperparams['batch_size'], 1)
    label_tensor = tf.Variable(
                               np.zeros(label_batch_shape,
                                        dtype='float32'),
                               name='labels')
    label_tensor_group = tf.Variable(
                                     np.zeros(label_batch_shape,
                                              dtype='float32'),
                                     name='labels_group')

    group_batch_shape = (cf.hyperparams['batch_size'], num_groups)
    group_tensor = tf.Variable(
                               np.zeros(group_batch_shape,
                                        dtype='float32'),
                               name='groups')

    return (feat_tensor, feat_tensor_group, label_tensor,
            label_tensor_group, group_tensor)
