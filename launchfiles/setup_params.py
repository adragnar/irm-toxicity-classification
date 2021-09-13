import argparse
import itertools
import os
from os.path import join
import time
import sys
import numpy as np

def setup(expdir, data_fname, wordvec_fname):    
    #Parameters for each experiment 'run'
    seeds = [152, 62, 8873, 1009, 3338, 9258, 447, 7611, 9044, 1381]
    splits = ['0.9-0.8-0.1']
    label_noise = [0]
    sens_att = ['new_LGBTQ', 'muslim', 'mental', 'black']
    w_enc = ['sbert', 'embed_sum', 'embed_mean']    
    tox_thresh = [0.5]
    preproc = ['reg']
    validation = [0.2]
    explicit_sa = [0, 1]

    cmdfile = join(expdir, 'cmdfile.sh')
    with open(cmdfile, 'w') as cmdf:
        for id, combo in enumerate(itertools.product(seeds, splits, sens_att, w_enc, tox_thresh, preproc, validation, explicit_sa)):
            command_str = \
            '''python experiment_scripts/main.py {id} {expdir} {data_fname} {wordvec_fname} {seed} {env_split} {sens_att} {w_enc} {tox_thresh} {preproc} {val} {esa}\n'''
            command_str = command_str.format(
                id=id,
                expdir=expdir,
                data_fname=data_fname,
                wordvec_fname=wordvec_fname,
                seed=combo[0],
                env_split=combo[1],
                sens_att=combo[2],
                w_enc=combo[3],
                tox_thresh=combo[4],
                preproc=combo[5],
                val=combo[6],
                esa=combo[7]
            )
            cmdf.write(command_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    # parser.add_argument("etype", type=str, default=None)
    parser.add_argument("expdir", type=str, default=None, help="Absolute path to results directory")
    parser.add_argument("data_fname", type=str, default=None, help="Absolute path to dataset.csv file")
    parser.add_argument("wordvec_fname", type=str, default=None, help="Absolute path to word vector .vec file")
    args = parser.parse_args()
    
    setup(args.expdir, args.data_fname, args.wordvec_fname)
 
    