'Hyperparameter configurations used in the paper'

def get_hyperparams(method, we_type):
    if method == 'baseline':
        if we_type == 'sbert':
            return  {'lr': 0.009201, \
                        'n_iterations':60000, \
                        'penalty_anneal_iters':0, \
                        'l2_reg':0.000592, \
                        'pen_wgt':0, \
                        'hid_layers':1, \
                        'verbose':False}
        else:
            return  {'lr':0.008804, \
                        'n_iterations':52500, \
                        'penalty_anneal_iters':0, \
                        'l2_reg':0.004557, \
                        'pen_wgt':0, \
                        'hid_layers':1, \
                        'verbose':False}
    elif method == 'irm':
        return  {'lr': 0.006285, \
                    'n_iterations':70000, \
                    'penalty_anneal_iters':157, \
                    'l2_reg':0.054086, \
                    'pen_wgt':52094.021177, \
                    'hid_layers':1, \
                    'verbose':False}
       
