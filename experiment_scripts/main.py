'Generate results for paper experiments'

import argparse
import logging
import os
from os.path import join

import sys
sys.path.insert(1, os.getcwd())
sys.path.insert(1, join(os.getcwd(), 'launchfiles'))

import logging
import pickle
import time

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split

import algo_hyperparams
import data_proc
import models
import ref
import partition_environments

def get_fullproc_data(t, train_ind_ps, val_ind_ps, ood_p, explicit_sa, sens_att, t_orig=1000.0):
    '''Apply transformations to SA and toxicity labels, convert environments into Dataset objects
       :return train_ind_envs: A list of Dataset objects, each an IOD environment
       :return val_ind-envs: A list of Dataset objects, each an IOD environment
       :return ood_env: A Dataset object, OOD environment'''

    #Set up explicitly including SA in experiment
    if explicit_sa:
        sa_fnc = lambda x: x[sens_att].values.squeeze()
    else:
        sa_fnc = False

    rel_cols = {'data':'comment_text', 'labels':'toxicity', 'sens_att':sens_att}
    include_cols = ['id', 'toxicity', 'comment_text', sens_att]
    train_ind_envs = [data_proc.ToxicityDataset(e[include_cols], rel_cols=rel_cols, add_sa=sa_fnc, transform=t)[:] for e in train_ind_ps]
    val_ind_envs = [data_proc.ToxicityDataset(e[include_cols], rel_cols=rel_cols, add_sa=sa_fnc, transform=t)[:] for e in val_ind_ps][:]
    ood_env = data_proc.ToxicityDataset(ood_p[include_cols], rel_cols=rel_cols, add_sa=sa_fnc, transform=t)[:]
    logging.info('Data Loaded, train_len: {}, test_len: {}, time: {:+.2f}'.format(\
                             str(int(train_ind_envs[0]['x'].shape[0] * 3)), \
                             str(ood_env['x'].shape[0]), \
                             (time.time() - t_orig)))

    return train_ind_envs, val_ind_envs, ood_env 

def get_raw_partitions(full_data, args):
    '''Given preprocessed dataset, seperate into environments. 
       :return train_ind_partitions: A list of dataframes, each an IOD environment
       :return val_ind_partitions: A list of dataframes, each an IOD environment
       :return ood_partition: A dataframe, OOD environment'''
    
    #Get Environment Data
    fullenv_partitions = partition_environments.partition_envs_cmnist(full_data, args)  

    #Segment into train-val-test
    if args['validation'] > 0:  #Val Envs
        train_ind_partitions = []
        val_ind_partitions = []
        for p in fullenv_partitions[:-1]:
            t_p, v_p = train_test_split(p, test_size=args['validation'], random_state=args['seed'])
            train_ind_partitions.append(t_p); val_ind_partitions.append(v_p)
    else:
        train_ind_partitions = fullenv_partitions[:-1]
        val_ind_partitions = []

    ood_eval, ood_tune = train_test_split(fullenv_partitions[-1], test_size=0.2, random_state=args['seed'])
    if args['hyp_tune']:
        ood_partition = ood_tune
    else:
        ood_partition = ood_eval

    return train_ind_partitions, val_ind_partitions, ood_partition

def load_full_data(data_fname, sens_att, tox_thresh, preproc, testing=0):
    '''Load dataframe and do text preprocessing'''

    full_data = pd.read_csv(data_fname)
    if testing == 1:
        full_data = full_data.sample(n=50000) #Do not make this smaller - not enough data for environment construction

    full_data[sens_att] = ref.pred_binarize(partition_environments.get_sensatt_column(full_data, sens_att), b=0.0001)
    full_data['raw_toxicity'] = full_data['toxicity']
    full_data = data_proc.preprocess_data(full_data, {'data':'comment_text', 'labels':'toxicity'},\
                                     tox_thresh=tox_thresh, c_len=15, \
                                     text_clean=preproc, stopwords=STOPWORDS)
    return full_data

def get_word_embeddings(we_type, wvec_path, t_orig=1000.0, testing=0):
    #Get Data Embedding Function
    if 'embed' in we_type:
        t = data_proc.get_word_transform(we_type, wvec_path, testing=testing)
        logging.info('WEs Loaded, num_words: {}, time: {:+.2f}'.format(\
                                   len(t.model.vocab), (time.time() - t_orig)))

    elif we_type == 'sbert':
        t = data_proc.get_word_transform('sbert', 'NA', testing=testing)
        logging.info('Sbert loaded, time: {:+.2f}'.format((time.time() - t_orig)))

    return t

def create_results_datastruct(model, args, algo_args, to_save_model, train_ind_envs, val_ind_envs, ood_env):
    '''Evaluate model performance and construct results object 
       :param model: Trained IRM model object 
       :param args: general parameters for run 
       :param algo_args: algorithm hyperparameters for run
       :param to_save_model: Dictionary, with key-value pairs of ('model_base', untrained model object)
                             and ('model', parameter values of some trained model object)
       :param train_ind_envs: A list of Dataset objects, each an IOD environment
       :param val_ind-envs: A list of Dataset objects, each an IOD environment
       :param ood_env: A Dataset object, OOD environment
        '''
    
    #Generate summary results 
    res_dict = {}
    for ltype in ['ACC', 'BCE']:
        train_metric = []
        for p in train_ind_envs:
            train_metric.append(evaluate_model(p, model, \
                                             ltype=ltype, b=args['tox_thresh']))
        test_metric = evaluate_model(ood_env, model, \
                                              ltype=ltype, b=args['tox_thresh'])
        val_metric = []
        for p in val_ind_envs:
            val_metric.append(evaluate_model(p, model, \
                                             ltype=ltype, b=args['tox_thresh']))

        res_dict[ltype] = {'train':train_metric, 'val':val_metric, 'test':test_metric}

    #Store raw results as confusion matricies for different SA values
    raw_dict = {}
    for sa in [0, 1]:
        raw_dict['sa_{}'.format(str(sa))] = {}

        for n, envset in zip(['train_ind', 'val_ind', 'ood'], \
                                [train_ind_envs, val_ind_envs, [ood_env]]):
            e_list = []
            for p in envset:
                assert ((p['sens_att']==0) | (p['sens_att']==1)).all()
                sa_ind = p['sens_att'] == sa
                cmat = ref.compute_loss(model.predict(p['x'][sa_ind]), \
                            p['y'][sa_ind], ltype='CONF', b=args['tox_thresh'])
                e_list.append(cmat)

            raw_dict['sa_{}'.format(str(sa))][n] = e_list

    return {'id':{'params':args, 'algo_params':algo_args}, \
            'model':to_save_model, \
            'results':res_dict, \
            'raw':raw_dict, \
            }


def evaluate_model(data, model, ltype='ACC', b=0.5):
    '''Given a model and its data, evaluate
    :param data: dict of np arrays dataset for a test env  {'x':arr, 'y':arr}
    :param model: the 'base' object with model object stored inside
    :param base: the model needed for prediction method'''
    preds = model.predict(data['x'])
    labels = data['y']
    acc = ref.compute_loss(preds, labels, ltype=ltype, b=b)
    return acc

def main(id, expdir, data_fname, wordvec_fname, args, algo_args, testing=0):
    logger_fname = os.path.join(expdir, 'log_{}.txt'.format(id))
    logging.basicConfig(filename=logger_fname, level=logging.DEBUG)
    t_orig = time.time()
    t = get_word_embeddings(args['word_encoding'], wordvec_fname, t_orig=t_orig, testing=testing)

    #Get full data
    full_data = load_full_data(data_fname, args['sens_att'], \
                                args['tox_thresh'], args['preproc'], testing=testing)
    train_ind_envs, val_ind_envs, ood_env = get_raw_partitions(full_data, args)
    train_ind_envs, val_ind_envs, ood_env = get_fullproc_data(t, \
                                                train_ind_envs, val_ind_envs, ood_env, \
                                                args['explicit_sa'], args['sens_att'], \
                                                t_orig=time.time())
    
    #Train models
    for method, method_args in algo_args.items():
        m_model = models.LinearInvariantRiskMinimization()
        errors, penalties, losses = m_model.train(train_ind_envs, args['seed'], method_args)

        #Save model
        to_save_model = {}
        to_save_model['model_base'] = models.LinearInvariantRiskMinimization()
        to_save_model['model'] = m_model.model
        
        m_res = create_results_datastruct(m_model, args, method_args, to_save_model, \
                            train_ind_envs, val_ind_envs, ood_env)
        pickle.dump(m_res, open(join(expdir, '{}_{}.pkl'.format(id, method)), 'wb'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('id', type=str, \
                        help='unique id of expieriment run')
    parser.add_argument("expdir", type=str, default=None,
                        help="path to results directory")
    parser.add_argument("data_fname", type=str,
                        help="path to data.csv file")
    parser.add_argument("wordvec_fname", type=str,
                        help="path to wordvecs.vec file. Can use garbage input if using SBERT embeddings") 
    parser.add_argument("seed", type=str, default=None, 
                        help="random seed")
    parser.add_argument("env_id", type=str, default=None, 
                        help="string of n+1 decimal fractional numbers connected by dashes (i.e 0.9-0.8-0.1). \
                            The first n are the spurious correlations for the n IOD environments. The last is \
                            the spurious correlation for the OOD environment")
    parser.add_argument("sens_att", type=str, default=None, 
                        help="name of sensitive attribute - one of ['black', 'muslim', 'new_LGBTQ', 'NeuroDiv]")
    parser.add_argument("word_encoding", type=str, default=None, 
                        help="type of word embedding used: one of ['sbert', 'embed_mean', 'embed_sum']")
    parser.add_argument("tox_thresh", type=float, default=None, 
                        help="label binarization threshold (from 0-1) above which comment labelled as toxic")
    parser.add_argument("preproc", type=str, default=None, 
                        help="text preprocessing applied to comments: one of ['na', 'reg']")
    parser.add_argument("validation", type=float, default=None, 
                        help="proportion of each environment (IOD and OOD) used for hyperparameter tuning")
    parser.add_argument("explicit_sa", type=int, default=None, 
                        help="binary flag (0 or 1): whether the sensitive attribute is concatenated to \
                            each comment embedding")


    #Hyperparams
    parser.add_argument('-hyp_tune', type=int, default=0, 
                        help="binary flag (0 or 1): whether the run is being used to tune hyper-parameters.")
    parser.add_argument('-lr', type=float, default=None, 
                        help="learning rate")
    parser.add_argument('-niter', type=int, default=None, 
                        help="number of iterations")
    parser.add_argument('-l2', type=float, default=None, 
                        help="l2 regularization penalty")
    parser.add_argument('-penalty_weight', type=float, default=None,
                        help="irm penalty weight parameter")
    parser.add_argument('-penalty_anneal', type=float, default=None, 
                        help="irm penalty anneal iterations parameter")
    
    #Other
    parser.add_argument('-testing', type=int, default=0, 
                        help="binary flag: 0 for regular run, 1 for run with reduced dataset size and \
                            word embeddings used (non-sbert embeddings only)")
    args = parser.parse_args()

    params = {'seed':int(args.seed), \
              'env_id':args.env_id, \
              'sens_att':args.sens_att, \
              'word_encoding':args.word_encoding, \
              'tox_thresh':args.tox_thresh, \
              'preproc':args.preproc, \
              'validation':args.validation, \
              'explicit_sa':args.explicit_sa, \
              'hyp_tune':args.hyp_tune
              }

    #Set up each method's hyperparameters 
    algo_params = {}  
    for m in ['baseline', 'irm']:
        hparams =  {'lr': args.lr, \
                    'n_iterations':args.niter, \
                    'penalty_anneal_iters':args.penalty_anneal, \
                    'l2_reg':args.l2, \
                    'pen_wgt':args.penalty_weight \
                    }
        if m == 'baseline':
            hparams['pen_wgt'] = 0.0; hparams['penalty_anneal_iters'] = 0; 

        default_hparams = algo_hyperparams.get_hyperparams(m, args.word_encoding)
        for p, v in hparams.items():
            if v is None:
                hparams[p] = default_hparams[p]

        algo_params[m] = hparams

    #Launch
    main(args.id, args.expdir, args.data_fname, args.wordvec_fname, params, algo_params, args.testing)
