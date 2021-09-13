'Generate environments from a single data source'

import logging
import pandas as pd
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value

def get_sensatt_column(data, satt):
    '''Return comments of a given sensitive attribute from Civil Comments dataset.
    :param data: pd Dataframe of Civil COmments
    :param satt: sensitive attribute name
    '''

    if satt == 'new_LGBTQ':
        return data[['homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation', \
                                  'other_gender', 'transgender']].max(axis=1)
    elif satt == 'muslim':
        return data['muslim']
    elif satt == 'mental':
        return data[['psychiatric_or_mental_illness', \
                        'intellectual_or_learning_disability']].max(axis=1)
    elif satt == 'black':
        return data['black']
    else:
        raise Exception('satt not implemented')

def partition_envs_cmnist(data, args):
    '''Split Civil Comments pd Dataframe into environments
    :param args['seed']: random seed (int)
    :param args['env_id']: correlation structure of environments (see main.py for description)
    :params args['sens_att']: sensitve attribute
    :return list of n + 1 pandas dfs. First n are IOD envs, last is OOD env'''

    ##Compute Size of Envrionments
    env_probs = [float(p) for p in args['env_id'].split('-')]

    opt_vars = {'tr1':{'vars':[0]*4, 'p':env_probs[0]}, 
                'tr2':{'vars':[0]*4, 'p':env_probs[1]}, 
                'te':{'vars':[0]*4, 'p':env_probs[2]}
            }

    #Find maximum for each env part 
    total_y0z0 = len(data[(data[args['sens_att']] == 0) & (data['toxicity'] == 0)])
    total_y0z1= len(data[(data[args['sens_att']] == 0) & (data['toxicity'] == 1)])
    total_y1z0 = len(data[(data[args['sens_att']] == 1) & (data['toxicity'] == 0)])
    total_y1z1 = len(data[(data[args['sens_att']] == 1) & (data['toxicity'] == 1)])
    logging.info("Total_y0z0: {} -- Total_y0z1: {} -- Total_y1z0: {} -- Total_y1z1: {}".format(total_y0z0, total_y0z1, total_y1z0, total_y1z1))

    # Create the model
    prob = LpProblem(name="small-problem", sense=LpMaximize)

    # Initialize the decision variables
    for e in opt_vars.keys(): 
        opt_vars[e]['vars'][0] = LpVariable(name="{}_y0z0".format(e), lowBound=0)
        opt_vars[e]['vars'][1] = LpVariable(name="{}_y0z1".format(e), lowBound=0)
        opt_vars[e]['vars'][2] = LpVariable(name="{}_y1z0".format(e), lowBound=0)
        opt_vars[e]['vars'][3] = LpVariable(name="{}_y1z1".format(e), lowBound=0)
        
    #Set up objective function 
    prob += lpSum(s for e in opt_vars.keys() for s in opt_vars[e]['vars'])

    #Set up constraints 

    #Equal environment sizes 
    prob += lpSum(opt_vars['tr1']['vars']) == lpSum(opt_vars['tr2']['vars']) 
    prob += lpSum(opt_vars['tr2']['vars']) == lpSum(opt_vars['te']['vars']) 

    #Upper bounds on env compositions
    prob += total_y0z0 >= lpSum([e['vars'][0] for e in opt_vars.values()])
    prob += total_y0z1 >= lpSum([e['vars'][1] for e in opt_vars.values()])
    prob += total_y1z0 >= lpSum([e['vars'][2] for e in opt_vars.values()])
    prob += total_y1z1 >= lpSum([e['vars'][3] for e in opt_vars.values()])


    #Correct env compositions 
    for e in opt_vars: 
        prob += opt_vars[e]['vars'][0] == float(opt_vars[e]['p']/2) * lpSum(opt_vars[e]['vars']) 
        prob += opt_vars[e]['vars'][3] == float(opt_vars[e]['p']/2) * lpSum(opt_vars[e]['vars']) 
        prob += opt_vars[e]['vars'][1] == float((1-opt_vars[e]['p'])/2) * lpSum(opt_vars[e]['vars']) 
        prob += opt_vars[e]['vars'][2] == float((1-opt_vars[e]['p'])/2) * lpSum(opt_vars[e]['vars']) 

    prob.solve()
    
    logging.info("Status: {}".format(LpStatus[prob.status]))
    for v in prob.variables():
        logging.info("{} = {}".format(v.name, v.varValue)) 
    logging.info("Total samples in all envs = {}".format(value(prob.objective)))

    ##Construct environments
    pmap = {'tr1':0, 'tr2':1, 'te':2}
    env_partitions = [0,0,0]

    for e in opt_vars:
        print(len(data[(data['toxicity'] == 0) & (data[args['sens_att']] == 0)]))
        e_data = data[(data['toxicity'] == 0) & (data[args['sens_att']] == 0)].sample(\
                                                        n=int(opt_vars[e]['vars'][0].varValue), \
                                                        random_state=args['seed'])

        print(len(data[(data['toxicity'] == 0) & (data[args['sens_att']] == 1)]))
        e_data = pd.concat([e_data, data[(data['toxicity'] == 0) & (data[args['sens_att']] == 1)].sample(\
                                                        n=int(opt_vars[e]['vars'][1].varValue), \
                                                        random_state=args['seed'])])

        print(len(data[(data['toxicity'] == 1) & (data[args['sens_att']] == 0)]))
        e_data = pd.concat([e_data, data[(data['toxicity'] == 1) & (data[args['sens_att']] == 0)].sample(\
                                                        n=int(opt_vars[e]['vars'][2].varValue), \
                                                        random_state=args['seed'])])

        print(len(data[(data['toxicity'] == 1) & (data[args['sens_att']] == 1)]))
        e_data = pd.concat([e_data, data[(data['toxicity'] == 1) & (data[args['sens_att']] == 1)].sample(\
                                                        n=int(opt_vars[e]['vars'][3].varValue), \
                                                        random_state=args['seed'])]) 
    
        data = data.drop(e_data.index)
        
        #Put into final list
        e_data.reset_index(inplace=True, drop=True)   
        env_partitions[pmap[e]] = e_data
    return env_partitions 

if __name__ == '__main__':
    pass