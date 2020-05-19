Implementation of `Cyberbullying Detection with Fairness Constraints <https://arxiv.org/abs/2005.06625>`_ - Gencoglu O. (2020) 
====================
This repository provides the full implementation with released models. Requires *python 3.7* and *TensorFlow 2.0* (see *requirements.txt*). 

Main Idea
====================
**Can we mitigate the unintended bias of cyberbullying detection models by guiding the model training with fairness constraints?**

.. raw:: html

    <img src="https://github.com/ogencoglu/fair_cyberbullying_detection/blob/master/media/main_idea.png" height="300px">

Quick overview
--------------

.. code-block:: python

  # list group-specific FNRs/FPRs
  fnrs = []
  fprs = []
  constraints = []
  for iden in range(cf.num_identity_groups):
      context_group_subset = context_group.subset(lambda kk=iden: group_tensor[:, kk] > 0)
      fnrs.append(tfco.false_negative_rate(context_group_subset))
      fprs.append(tfco.false_positive_rate(context_group_subset))

  # define lower and upper bound constraints (see equation 3 in paper)
  constraints.append(tfco.upper_bound(fnrs) - tfco.false_negative_rate(context) <= allowed_fnr_deviation)
  constraints.append(tfco.upper_bound(fprs) - tfco.false_positive_rate(context) <= allowed_fpr_deviation)
  constraints.append(tfco.false_negative_rate(context) - tfco.lower_bound(fnrs) <= allowed_fnr_deviation)
  constraints.append(tfco.false_positive_rate(context) - tfco.lower_bound(fprs) <= allowed_fpr_deviation)

  # define problem, optimizer and variables to optimize
  problem = tfco.RateMinimizationProblem(objective, constraints)
  optimizer = tfco.ProxyLagrangianOptimizerV2(
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      constraint_optimizer=tf.keras.optimizers.Adam(learning_rate),
      num_constraints=problem.num_constraints)
  var_list = (constrained_model.trainable_weights + problem.trainable_variables + optimizer.trainable_variables())

Example Results
====================

.. image:: https://github.com/ogencoglu/fair_cyberbullying_detection/blob/master/media/result.png
   :width: 400

Quick Reproduction of Results
====================
1 - Get the data
--------------
See *directory_info* in the *data* directory for the expected directory structure.

  -Jigsaw            `Link <https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data>`_
  -Twitter           `Link <https://github.com/xiaoleihuang/Multilingual_Fairness_LREC/tree/master/data>`_
  -Wiki              `Link <https://figshare.com/projects/Wikipedia_Talk/16731>`_
  -Gab               `Link <https://osf.io/edua3/>`_
2 - Download *unconstrained* and *constrained* models
--------------
`Download released models <https://drive.google.com/file/d/13i2dPf5FWw-NjUupbTMqvIJtZVJo_dGM/view?usp=sharing>`_ to *models* directory. See *directory_info* in the *model* directory for the expected directory structure.

3 - Run *compare_models.ipynb*
-------------------------------
See *source* directory.

Training From Scratch
====================
Run the corresponding notebook (e.g. *gab_experiment.ipynb*) for each experiment in the *source* directory for reproducing the full results from scratch. Note that the algorithms are non-determinisitic due to random weight initialization of the models.

Relevant configurations are defined in *configs.py*, e.g.:

  --batch_size                       128
  --epochs                           75
  --gab_allowed_fnr_deviation        0.10
  --gab_allowed_fpr_deviation        0.15
  --random_state                     42
  
*source* directory tree:

.. code-block:: bash

    ├── compare_models.ipynb
    ├── configs.py
    ├── embeddings.py
    ├── evaluation.py
    ├── gab_experiment.ipynb
    ├── jigsaw_experiment.ipynb
    ├── metrics.py
    ├── model.py
    ├── plot.py
    ├── train.py
    ├── twitter_experiment.ipynb
    ├── utils.py
    └── wiki_experiment.ipynb
  
`Cite <https://scholar.google.fi/scholar?hl=en&as_sdt=0%2C5&q=Cyberbullying+Detection+with+Fairness+Constraints&btnG=>`_
====================
  
.. code-block::

    @article{gencoglu2020cyberbullying,
      title={Cyberbullying Detection with Fairness Constraints},
      author={Gencoglu, Oguzhan},
      journal={arXiv preprint arXiv:2005.06625},
      year={2020}
    }
    
Or

    Gencoglu, Oguzhan. "Cyberbullying Detection with Fairness Constraints." arXiv preprint arXiv:2005.06625 (2020).
