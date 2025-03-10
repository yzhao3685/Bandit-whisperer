import numpy as np
import pandas as pd
import time

from itertools import product

from robust_rmab.environments.bandit_env import SISBanditEnv, RandomBanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv, ARMMANEnv
from robust_rmab.environments.bandit_env_robust import  CounterExampleRobustEnv, ARMMANRobustEnv, SISRobustEnv, FakeT, ContinuousStateExampleEnv

import os
import os.path as osp
import argparse
import tqdm
import itertools
import torch

import mdptoolbox


import matplotlib.pyplot as plt

index_policies = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
RL_policies = [101, 102, 103, 104, 105]


def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0])
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options


def barPlot(labels, values, errors, ylabel='Mean Discounted Reward',
            title='Simulated policy rewards', filename='image.png', root='.',
            bottom=0):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)       
    plt.tight_layout() 
    plt.savefig(fname)
    plt.show()


class RobustEnvWrapper():
    def __init__(self, env):
        # params is either nature_params or opt_in.
        # when we have changing # arms, params is opt_in
        self.env = env
        self.opt_in = np.ones(env.N)

        # loop over the attributes of the parent class and create those for the decorator
        env_dict = vars(env)
        for attr in [a for a in env_dict if '__' not in a]:
            self.__dict__[attr] = env_dict[attr]

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset_random()

    def step(self, actions):
        return self.env.step(actions, self.opt_in)

class RobustEnvWrapperArmman():
    def __init__(self, env, params):
        # params is either nature_params or opt_in.
        # when we have changing # arms, params is opt_in

        self.params = params
        self.env = env

        # loop over the attributes of the parent class and create those for the decorator
        env_dict = vars(env)
        for attr in [a for a in env_dict if '__' not in a]:
            self.__dict__[attr] = env_dict[attr]

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset_random()

    def step(self, actions):
        return self.env.step(actions, self.params)


def takeAction(current_states, T, actions, random_stream):

    N=len(current_states)

    ###### Get next state
    next_states=np.zeros(current_states.shape)
    for i in range(N):

        current_state=int(current_states[i])
        next_state=np.argmax(random_stream.multinomial(1, T[i, current_state, int(actions[i]), :]))
        next_states[i]=next_state

    return next_states


def getActions(N, T, R, C, B, t, policy_option, act_dim, rl_info=None,
                        current_state=None, data_dict=None, env=None,
                        valid_action_combinations=None):

    gamma = data_dict['gamma']

    # Pull no arms
    if policy_option==0:

        return np.zeros(N)

    # Pull all arms
    elif policy_option==1:

        return np.ones(N)

    # Random continuous
    elif policy_option==2:

        # actions = np.zeros(N,dtype=int)
        # arm_inds = np.arange(N)
        # choices = np.random.choice(arm_inds, B, replace=False)
        # actions[choices] = 1
        # return actions
        actions = np.random.dirichlet([1]*N*act_dim)*B
        return actions.reshape(N,-1)

    # Random discrete (binary action only)
    elif policy_option==3:

        actions = np.zeros(N,dtype=int)
        arm_inds = np.arange(N)
        choices = np.random.choice(arm_inds, int(B), replace=False)
        actions[choices] = 1
        return actions




    # Round robin 1
    elif policy_option==5:
        actions = np.zeros(N)
        num_feasible = int(B/C[1])
        last_proc_acted_on = data_dict['last_proc_acted_on_rr']
        ind = 0
        for i in range(last_proc_acted_on+1, last_proc_acted_on+1 + num_feasible):
            ind = i%N
            actions[ind] = 1

        data_dict['last_proc_acted_on_rr'] = ind
        return actions


    # Fast random, inverse weighted, works for multi-action
    elif policy_option==6:

        actions = np.zeros(N,dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(N), N, replace=False)
        for arm in process_order:
            
            # select an action at random
            num_valid_actions_left = len(C[C<=B-current_action_cost])
            p = 1/(C[C<=B-current_action_cost]+1)
            p = p/p.sum()
            p = None
            a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
            current_action_cost += C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > B:
                break

            actions[arm] = a


        return actions



    # Hawkins - must be discrete
    elif policy_option==21:

        actions = np.zeros(N)
        # if T is a FakeT, we can't actually run Hawkins
        # use to avoid headahces when running sis
        if isinstance(T, FakeT):
            return actions

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros((N, C.shape[0], T.shape[1]))
        current_state = current_state.reshape(-1)
        current_state = current_state.astype(int)
        L_vals, lambda_val, obj_val = lp_methods.hawkins(T, R, C, B, current_state, lambda_lim=lambda_lim, gamma=gamma)


        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] - lambda_val*C[a] + gamma*L_vals[i].dot(T[i,s,a])
        data_dict['hawkins_lambda'].append(lambda_val)
        if args.just_hawkins_lambda:
            print('state', current_state)
            print('L_vals', L_vals)
            print('lambda',lambda_val)
            print('obj_val',obj_val)
            1/0

        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            print(s)
            indexes_per_state[i] = indexes[i,:,s]

        # start = time.time()

        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B:
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")

        return actions


    # LP to compute the index policies (online vs. oracle version)
    elif policy_option==22:

        # print(policy_option)
        # print(T)

        actions = np.zeros(N,dtype=int)

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros(N)

        a_index = 1 # need this because below method is set up to compute multi-action indices
        _, indexes = lp_methods.lp_to_compute_index(T, R, C, B, current_state, a_index, lambda_lim=lambda_lim, gamma=gamma)


        # compute all indexes
        if t==1 and policy_option==22:
            all_indexes = np.zeros((N, T.shape[1]))
            for s in range(T.shape[1]):
                state_vec = np.ones(N,dtype=int)*s
                # print(state_vec)
                _, all_indexes[:,s] = lp_methods.lp_to_compute_index(T, R, C, B, state_vec, 1, lambda_lim=lambda_lim, gamma=gamma)


            data_dict['lp-oracle-index'] = all_indexes



        # print(actions)
        # print(C)

        # action selection
        print('states')
        print(current_state)
        print('indexes')
        print(indexes)
        sorted_inds = np.argsort(indexes)[::-1]
        num_selected_actions = 0
        ind_cursor = 0
        epsilon=1e-6
        while num_selected_actions < B:

            next_best_index_value = indexes[sorted_inds[ind_cursor]]
            inds_of_equal_value = np.argwhere(abs(indexes - next_best_index_value)<epsilon).reshape(-1)
            if len(inds_of_equal_value) + num_selected_actions <= B:
                actions[inds_of_equal_value] = 1
                num_selected_actions += len(inds_of_equal_value)
                ind_cursor += len(inds_of_equal_value)
            else:
                num_actions_remaining = B - num_selected_actions
                randomly_chosen_inds = np.random.choice(inds_of_equal_value, num_actions_remaining, replace=False)
                actions[randomly_chosen_inds] = 1
                num_selected_actions = B



        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment == B: raise ValueError("Wrong budget")


        return actions



    # No cost value funcs
    elif policy_option==24:

        indexes = data_dict['indexes']

        actions = np.zeros(N)

        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

        return actions



    # State-based random
    # note this is only implemented for single action right now
    # and only meant for the 2state data
    elif policy_option==25:

        actions = np.zeros(N,dtype=int)
        if current_state.all():
            arm_inds = np.arange(N)
            choices = np.random.choice(arm_inds, B, replace=False)
            actions[choices] = 1
        else:
            state0_inds = np.argwhere(current_state == 0.0).reshape(-1)

            choices = np.random.choice(state0_inds, B, replace=False)
            actions[choices] = 1

        return actions

    # with communication
    elif policy_option == 102:
        if rl_info['data_type'] == 'continuous':
            actions = get_action_rl(rl_info['model'], (current_state), comm_choice='comm')
            payment = C(actions).sum()
            EPS = 1e-6
            if payment - EPS > B: raise ValueError("Over budget",payment)
            return actions
        elif rl_info['data_type'] == 'discrete':
            actions = get_action_rl(rl_info['model'], current_state, comm_choice='comm')
            payment = 0
            for a in actions:
                payment+= C[a]
            EPS = 1e-6
            return actions

    # no communication
    elif policy_option == 103:
        if rl_info['data_type'] == 'continuous':
            actions = get_action_rl(rl_info['model'], (current_state), comm_choice='control')
            payment = C(actions).sum()
            EPS = 1e-6
            if payment - EPS > B: raise ValueError("Over budget",payment)
            return actions
        elif rl_info['data_type'] == 'discrete':
            actions = get_action_rl(rl_info['model'], current_state, comm_choice='control')
            payment = 0
            for a in actions:
                payment+= C[a]
            return actions

    # topline (no noise)
    elif policy_option == 104:
        if rl_info['data_type'] == 'continuous':
            actions = get_action_rl(rl_info['model'], (current_state), comm_choice='topline')
            payment = C(actions).sum()
            EPS = 1e-6
            if payment - EPS > B: raise ValueError("Over budget",payment)
            return actions
        elif rl_info['data_type'] == 'discrete':
            actions = get_action_rl(rl_info['model'], current_state, comm_choice='topline')
            payment = 0
            for a in actions:
                payment+= C[a]
            return actions

    # oracle (communication knowing which arms are noisy)
    elif policy_option == 105:
        if rl_info['data_type'] == 'continuous':
            actions = get_action_rl(rl_info['model'], (current_state), comm_choice='oracle')
            payment = C(actions).sum()
            EPS = 1e-6
            if payment - EPS > B: raise ValueError("Over budget",payment)
            return actions
        elif rl_info['data_type'] == 'discrete':
            actions = get_action_rl(rl_info['model'], current_state, comm_choice='oracle')
            payment = 0
            for a in actions:
                payment+= C[a]
            return actions

# make function for producing an action given a single state
def get_action_rl(model, x, comm_choice):
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        action = model.act_test(x, method=comm_choice)
    return action

def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    # print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    return model

def featurize_tp(transition_probs, transformation=None, out_dim=4):
    N = transition_probs.shape[0]
    output_features = np.zeros((N, out_dim))
    np.random.seed(0)  # Set random seed for reproducibility
    
    if transformation == "linear":
        transformation_matrix = np.random.rand(4, out_dim)
        output_features = np.dot(transition_probs, transformation_matrix)
    elif transformation == "nonlinear":
        transformation_matrix = np.random.rand(4, out_dim)
        output_features = 1 / (1 + np.exp(-np.dot(transition_probs, transformation_matrix)))
    else:
        output_features[:, :min(4, out_dim)] = transition_probs[:, :min(4, out_dim)]
    return output_features

def simulateAdherence(N, L, T, R, C, B, policy_option, start_state, seedbase=None, world_random_seed=None,
                        data_dict=None, file_root=None, env=None, rl_info=None, valid_action_combinations=None):

    gamma = data_dict['gamma']

    # env.seed(world_random_seed)
    start_state = env.reset()

    state_log=np.zeros((N, L, env.observation_dimension))
    action_log=np.zeros((N, L-1, env.action_dimension))


    # indexes = np.zeros((N,C.shape[0]))

    # Round robin setups
    if policy_option in [5]:
        data_dict['last_proc_acted_on_rr'] = N-1




    if policy_option in RL_policies:
            model = load_pytorch_policy(rl_info['model_file_path_rmab'], "")
            rl_info['model'] =  model

    # print('Running simulation w/ policy: %s'%policy_option)
    if policy_option in index_policies:

        lambdas = np.zeros((N,C.shape[0]))
        V = np.zeros((N,T.shape[1]))

        start = time.time()


        if policy_option == 21:
            pass


        # VfNc
        if policy_option == 24:

            start = time.time()
            indexes = np.zeros((N, C.shape[0], T.shape[1]))

            # time to: add variables, add constraints, optimize, extract variable values
            for i in range(N):
                # Go from S,A,S to A,S,S
                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] = R[i]

                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='full', epsilon=data_dict['mdp_epsilon'])
                mdp.run()

                V[i] = np.array(mdp.V)


            for i in range(N):
                for a in range(C.shape[0]):
                    for s in range(T.shape[1]):
                        indexes[i,a,s] = R[i,s] + gamma*V[i].dot(T[i,s,a])



    state_log[:,0] = start_state

    # data_dict['indexes'] = indexes


    #######  Run simulation #######
    # print('Running simulation w/ policy: %s'%policy_option)
    # print("Policy:", policy_option)

    ep_ret = 0
    for t in range(1,L):
        # print("Round %s"%t)

        actions=getActions(N, T, R, C, B, t, policy_option, env.action_space,
                            rl_info=rl_info, current_state=state_log[:,t-1],
                            data_dict=data_dict, env=env,
                            valid_action_combinations=valid_action_combinations)
        actions = actions.reshape(N,-1)

        action_log[:, t-1]=actions
        EPS = 1e-6
        if policy_option != 0 and actions.sum() - B > EPS:
            print(actions.sum())
            raise ValueError('bad num actions')
        # print(policy_option, state_log[:,t-1])
        # print(policy_option, actions)


        next_state, r, d, _ = env.step(actions)
        ep_ret += r.sum()
        state_log[:,t] = next_state
        # state_log[:,t] = takeAction(state_log[:,t-1].reshape(-1), T, actions, random_stream=world_random_stream)

    # if policy_option == 22:
    #     utils.plot_indexes(data_dict['lp-oracle-index'])

    # if policy_option == 24:
    #     utils.plot_vfnc_indexes(data_dict['indexes'])


    return state_log, action_log, ep_ret





if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run adherence simulations with various methods.')
    parser.add_argument('-N', '--num_arms', default=4, type=int, help='Number of Processes')
    parser.add_argument('-b', '--budget_frac', default=0.5, type=float, help='Budget per day as fraction of n')
    parser.add_argument(      '--budget', default=None, type=float, help='Total budget per day (trumps budget_frac)')
    parser.add_argument('-L', '--simulation_length', default=180, type=int, help='Number of days to run simulation')
    parser.add_argument('-n', '--num_trials', default=3, type=int, help='Number of trials to run')
    parser.add_argument('-S', '--num_states', default=2, type=int, help='Number of states per process')
    parser.add_argument('-A', '--num_actions', default=2, type=int, help='Number of actions per process') # Only two actions implemented for now
    parser.add_argument('-g', '--discount_factor', default=0.9, type=float, help='Discount factor for MDP solvers')
    parser.add_argument('-rb', '--REWARD_BOUND', default=1.0, type=float, help='Maximum reward')
    parser.add_argument('-opt', '--opt_in_rate', default=1.0, type=float, help='Maximum reward')

    parser.add_argument('-d', '--data', default='real', type=str,help='Method for generating transition probabilities',
                            choices=[   'SIS_old',
                                        'random',
                                        'random_reset',
                                        'circulant', 
                                        'armman',
                                        'counterexample',
                                        'sis',
                                        'continuous_state'
                                    ])

    parser.add_argument('-me', '--mdp_epsilon', default=1e-1, type=float, help='Tolerance for Value Iteration')

    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws','--world_seed_base', default=None, type=int, help='Base for the random seed')

    parser.add_argument('-f', '--file_root', default='./..', type=str,help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='policy to run, default is all policies')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str, help='special string to include in saved file name')

    parser.add_argument('-sid', '--slurm_array_id', default=None, type=int, help='Specify the index of the parameter combo')
    parser.add_argument('-sva', '--save_actions', default=False, type=bool, help='Whether or not to save action logs')

    parser.add_argument('--init_infection_size', type=int, default=5)
    parser.add_argument('--population_size', type=int, default=100)

    parser.add_argument('-rlmfc', '--rl_combinatorial_model_filepath', default=None, type=str, help='path to Combinatorial RL model file if using')
    parser.add_argument('-rlmfr', '--rl_rmab_model_filepath', default=None, type=str, help='path to RMAB RL model file if using')
    parser.add_argument('-dt', '--data_type', default='discrete', type=str, choices=['continuous','discrete'], help='Whether data is continuous or discrete')
    parser.add_argument('-jhl', '--just_hawkins_lambda', default=False, type=bool, help='Just output the Hawkins lambda value')

    parser.add_argument('-nh', '--no_hawkins', default=1, type=int, help='If set, will not run Hawkins')

    parser.add_argument('--robust_keyword', default='mid', type=str,help='Method for picking some T out of robust env',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt',
                                    'sample_random'
                                ])
    parser.add_argument('-ps', '--pop_size', default=10, type=int, help='Population size for SIS')

    args = parser.parse_args()
    print('file path ', args.rl_rmab_model_filepath)
    if not args.no_hawkins:
        from robust_rmab import lp_methods


    # CODE ONLY IMPLEMENTED FOR TWO ACTIONS FOR NOW
    args.num_actions = 2


    # policy names dict
    pname={
            0: 'No Calls',    2: 'RandomC', 3: 'RandomD',
            4: 'MDP Optimal',
            5: 'Round Robin',

            6:'RandomDS',

            21:'Hawkins',
            22:'Whittle index',

            24:r'$\lambda=0$',
            25:'S-Based Random',

            101:'RLcomb',
            102:'DistRL communication',
            103: 'DistRL no communication',
            104: 'DistRL topline',
            105: 'DistRL oracle'
    }


    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()

    ##### Save special name
    if args.save_string=='':
        args.save_string=str(time.ctime().replace(' ', '_').replace(':','_'))


    policies = [102, 103, 104, 105, 0, 6] # start with PreFeRMAB to initialize the transition dynamics.
    policies_to_plot = policies

    ################# This section only for setting up batched jobs to run on FASRC's cannon ########################

    # for i in 0 21 24; do sbatch --array=0-899 job.run_simulation_low_mem.sh $i; done

    NUM_TRIALS = 20
    trial_number_list = [i for i in range(NUM_TRIALS)]
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]
    master_combo_list = list(itertools.product(trial_number_list, n_list, budget_frac_list, state_size_list))

    # print(len(master_combo_list));1/0

    trial_percent_lam0 = 0
    if args.slurm_array_id is not None:
        combo = master_combo_list[args.slurm_array_id]

        args.trial_number = combo[0]
        args.num_arms = combo[1]
        args.budget_frac = combo[2]
        args.num_states = combo[3]

    # If we pass a trial number, that means we are running this as a job
    # and we want jobs/trials to run in parallel so this does some rigging to enable that,
    # while still synchronizing all the seeds
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0


    #################################################################################################################


    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial




    N=args.num_arms
    L=args.simulation_length
    savestring=args.save_string
    N_TRIALS=args.num_trials
    S = args.num_states
    A = args.num_actions
    B = int(N*args.budget_frac)
    opt_in_rate = args.opt_in_rate
    if args.budget is not None:
        B = args.budget
    rl_info = {
        'model_file_path_combinatorial':args.rl_combinatorial_model_filepath,
        'model_file_path_rmab':args.rl_rmab_model_filepath,
        'data_type':args.data_type,
        'compute_hawkins_lambda':False

    }



    size_limits={
                    0:None, 1:None, 2:1000, 3:1000, 4:8,
                    5:None, 6:None,
                    21:None, 22:None,
                    24:None, 25:None,
                    101:1000, 102:None,
                    103:None, 104:None,
                    105:None
                }





    # for rapid prototyping
    # use this to avoid updating all the function calls when you need to pass in new
    # algo-specific things or return new data
    data_dict = {}
    data_dict['hawkins_lambda'] = []
    data_dict['gamma'] = args.discount_factor
    data_dict['mdp_epsilon'] = args.mdp_epsilon

    data_dict['hawkins_lambdas_rl_states'] = []
    data_dict['rl_lambdas'] = []
    data_dict['dataset_name'] = args.data




    start=time.time()
    file_root=args.file_root

    # for post-computation
    runtimes = np.zeros((N_TRIALS, len(policies)))
    reward_log=dict([(key,[]) for key in pname.keys()])
    state_log=dict([(key,[]) for key in pname.keys()])




    T = None
    R = None
    C = None
    # B = None gets set above
    start_state = None
    env=None


    one_hot_encode = True
    non_ohe_obs_dim = None

    # use np global seed for rolling random data, then for random algorithmic choices
    seedbase = first_seedbase
    torch.manual_seed(seedbase)
    np.random.seed(seed=seedbase)

    # Use world seed only for evolving the world (If two algs
    # make the same choices, should create the same world for same seed)
    world_seed_base = first_world_seedbase

    if args.data =='SIS':
        REWARD_BOUND = 1
        population_sizes = np.array([args.population_size]*N)
        init_infection_size = args.init_infection_size
        env = SISBanditEnv(N, population_sizes, B, seedbase, init_infection_size, REWARD_BOUND)
        C = env.costs_all

    if args.data =='random':
        REWARD_BOUND = 1
        env = RandomBanditEnv(N, S, A, B, seedbase, REWARD_BOUND)
        T = env.T
        R = env.R
        C = env.C

    if args.data =='random_reset':
        REWARD_BOUND = 1
        env = RandomBanditResetEnv(N, S, A, B, seedbase, REWARD_BOUND)
        T = env.T
        R = env.R
        C = env.C

    if args.data =='circulant':

        env = CirculantDynamicsEnv(N, B, seedbase)
        T = env.T
        R = env.R
        C = env.C


    if args.data == 'continuous_state':
        env = ContinuousStateExampleEnv(N, B, seedbase, rl_info['data_type'])
        one_hot_encode = False
        non_ohe_obs_dim = 1

        T = env.T
        R = env.R
        C = env.C
        
        current_state = np.random.get_state()
        np.random.set_state(current_state)
        env = RobustEnvWrapper(env)

    if args.data == 'counterexample':
        from robust_rmab.baselines.nature_baselines_counterexample import   (
                    RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                    OptimisticNaturePolicy, DetermNaturePolicy, SampledRandomNaturePolicy
                )
        env_fn = lambda : CounterExampleRobustEnv(N,B,seedbase)

        model = load_pytorch_policy(rl_info['model_file_path_rmab'], "")
        env = model.env
        T = env.T
        R = env.R
        C = env.C

        current_state = np.random.get_state()
        np.random.set_state(current_state)
        env = RobustEnvWrapper(env)

    if args.data == 'armman':
        from robust_rmab.baselines.nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: ARMMANRobustEnv(N,B,seedbase)

        one_hot_encode = False
        non_ohe_obs_dim = 1

        model = load_pytorch_policy(rl_info['model_file_path_rmab'], "")
        env = model.env
        T = env.T
        R = env.R
        C = env.C

        current_state = np.random.get_state()
        np.random.set_state(current_state)
        env = RobustEnvWrapperArmman(env)

    if args.data == 'sis':
        from robust_rmab.baselines.nature_baselines_sis import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: SISRobustEnv(N,B,args.pop_size,seedbase)
        
        # don't one hot encode this state space...
        one_hot_encode = False
        non_ohe_obs_dim = 1
        POP_SIZE_LIM = 2000

        # load the same arms as in training
        model = load_pytorch_policy(rl_info['model_file_path_rmab'], "")
        env = model.env
        T = env.T
        R = env.R
        C = env.C

        current_state = np.random.get_state()
        np.random.set_state(current_state)
        env = RobustEnvWrapper(env)


    valid_action_combinations = None



    np_seed_states = []
    world_seed_states = []

    # create a bunch of random seed states that can be replicated 
    # if we need to add another policy later
    for i in range(N_TRIALS):
        
        # save the states
        np_seed_states.append(np.random.get_state())
        world_seed_states.append(env.random_stream.get_state())

        # evolve the states
        np.random.rand()
        env.random_stream.rand()
        
    for i in range(N_TRIALS):
        if valid_action_combinations is None:
            combinatorial_policies = set(policies) & set([4, 101])
            if len(combinatorial_policies) > 0:

                at_least_one_size_limit_satisfied = False
                for policy in combinatorial_policies:
                    at_least_one_size_limit_satisfied |= size_limits[policy]>N

                if at_least_one_size_limit_satisfied:
                    options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
                    valid_action_combinations = list_valid_action_combinations(N,C,B,options)


        


        for p,policy_option in enumerate(policies):


            if size_limits[policy_option]==None or size_limits[policy_option]>N:

                np_seed_state_for_trial = np_seed_states[i]
                world_seed_state_for_trial = world_seed_states[i]


                # reset the seed states for all policies to have same shot
                np.random.set_state(np_seed_state_for_trial)
                env.random_stream.set_state(world_seed_state_for_trial)

                # TODO - recover MDP optimal
                optimal_policy = None
                combined_state_dict = None

                data_dict['optimal_policy'] = optimal_policy
                data_dict['combined_state_dict'] = combined_state_dict


                policy_start_time=time.time()
                state_matrix, action_log, ep_ret = simulateAdherence(N, L, T, R, C, B, policy_option, start_state, seedbase=seedbase, world_random_seed=world_seed_base,
                                                   data_dict=data_dict, file_root=file_root, env=env, rl_info=rl_info, valid_action_combinations=valid_action_combinations)
                policy_end_time=time.time()


                ####### SAVE RELEVANT LOGS #########

                state_log[policy_option].append(state_matrix)

                policy_run_time=policy_end_time-policy_start_time
                np.save(file_root+'/logs/runtimes/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states), policy_run_time)

                runtimes[i,p] = policy_run_time

                reward_log[policy_option].append(ep_ret)



                # write out action logs
                if args.save_actions:
                    print(action_log.shape)
                    print(action_log)
                    fname = os.path.join(args.file_root,'logs/action_log/action_log_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s.csv'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states))
                    columns = list(map(str, np.arange(L-1)))
                    df = pd.DataFrame(action_log, columns=columns)
                    df.to_csv(fname, index=False)




    end=time.time()
    print ("Time taken: {:2f} s".format(end-start))

    # for i,p in enumerate(policies):
    #     print ('{}: {:.3f} +/- {:.3f}'.format(pname[p], runtimes[:,i].mean(), runtimes[1:,i].std()))

    # print("Max state:",max_state)

    # labels = [pname[i] for i in policies_to_plot]
    # values_for_df=np.array([reward_log[i] for i in policies_to_plot])
    # values_for_df=values_for_df.T
    #
    # df = pd.DataFrame(values_for_df, columns=labels)
    # fname = file_root+'/logs/results/rewards_%s_n%s_b%s_opt%s_tp-%s.csv'%(args.data, N, int(args.budget), args.opt_in_rate, args.data_type)
    # if os.path.exists(fname):
    #     df.to_csv(fname, mode='a', header=False, index=False)
    # else:
    #     df.to_csv(fname, index=False)

    # now write aggregate info in a separate csv
    fname = file_root+'/logs/results/rewards_%s_n%s_b%s_seed%s.csv'%(args.data, N, int(args.budget), args.seed_base)
    labels = [pname[i] for i in policies_to_plot]
    values = [round(np.mean(np.array(reward_log[i])), 4) for i in policies_to_plot]
    import csv
    if os.path.exists(fname):
        with open(fname, 'a') as aggregate_file:
            writer = csv.writer(aggregate_file)
            writer.writerow(values)
            aggregate_file.flush()
    else:
        with open(fname, 'w') as aggregate_file:
            writer = csv.writer(aggregate_file)
            writer.writerow(labels)
            writer.writerow(values)
            aggregate_file.flush()

    ##### do some basic plotting if running at the command line with more than one policy
    if args.policy<0:
        labels = [pname[i] for i in policies_to_plot]
        values=[round(np.mean(np.array(reward_log[i])), 4) for i in policies_to_plot]
        errors=[np.std(np.array(reward_log[i])) for i in policies_to_plot]
        vals = [values, errors]

        print('rewards')
        for i in range(len(labels)):
            print('{} {} +/- {:.3f}'.format(labels[i], values[i], errors[i]))
