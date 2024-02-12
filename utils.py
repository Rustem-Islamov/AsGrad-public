import numpy as np
import pickle
import os

SAVED_RUNS_PATH = 'saved_data/'
EXP_PATH = 'exps_setup/'

def save_run_func(suffix, run):
    if not os.path.isdir(SAVED_RUNS_PATH):
        os.mkdir(SAVED_RUNS_PATH)
    file = SAVED_RUNS_PATH + suffix + '.pickle'
    with open(file, 'wb') as f:
        pickle.dump(run, f)
        
def load_run_func(suffix=''):
    file = SAVED_RUNS_PATH + suffix + '.pickle'
    with open(file, 'rb') as f:
        run = pickle.load(f)
    return run

def nonconvex_reg(x):
    return 2*x/(1+x**2)**2

def l2_reg(x):
    return 2*x

# default data set parameters for each data set
default_dataset_parameters = {
    'a1a': {
        'N':1600,
        'n':16,
        'm':100,
        'd':123
    },
    'a9a': {
        'N':32560, 
        'n':80,
        'm':407,
        'd':123
    },
    'w7a': {
        'N':24600, 
        'n':50,
        'm':492,
        'd':300
    },
    'w8a': {
        'N':49700, 
        'n':142,
        'm':350,
        'd':300
    },
    'phishing': {
        'N':11000, 
        'n':100,
        'm':110,
        'd':68    
    },
    'madelon': {
        'N':2000, 
        'n':5,
        'm':400,
        'd':500
    },
    'gisette': {
        'N':6000, 
        'n':6,
        'm':1000,
        'd':5000
    },
    'colon-cancer': {
        'N':60, 
        'n':4,
        'm':15,
        'd':2000
    },
    'leukemia': { 
        'N':72, 
        'n':4,
        'm':18,
        'd':7129
    }
}


def run_optimizer(x_init, nodes, lr, num_iter=1000, assign_type='pure'):
    nodes.init(x_init)
    x = np.copy(x_init)
    delays, errors = [], [np.linalg.norm(nodes.oracle.full_gradient(x))]

    for cur_iter in range(0, num_iter):
        nodes.decrease_time()
        grad, delay = nodes.get_update()
        x = x - lr * grad
        nodes.assign_new_job(np.copy(x), assign_type=assign_type)

        delays += [delay]
        errors += [np.linalg.norm(nodes.oracle.full_gradient(x))]

    return errors, delays

def generate_synthetic(alpha, beta, iid, d, n, m):
    '''
    ----------------------------------
    synthetic data generation function
    ----------------------------------
    input:
    alpha, beta - parameters of data
    iid - if 1, then the data distribution over nodes is iid
        - if 0, then the data distribution over nodes is non-iid
    d - dimension of the problem
    n - number of nodes
    m - size of local data
    
    output:
    numpy arrays A (features) and b (labels)
    '''
    
    NUM_USER = n
    dimension = d
    NUM_CLASS = 1
    N = n*m
    
    samples_per_user = [m for i in range(NUM_USER)]
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    #print(mean_b)
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    if iid == 1:
        W_global = np.random.normal(0, 1, dimension)
        b_global = np.random.normal(0, 1, NUM_CLASS)


    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, dimension)
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)


        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            p = sigmoid(tmp[0])
            yy[j] = np.random.choice([-1,1], p=[p,1-p])

        X_split[i] = xx
        y_split[i] = yy

    A = np.zeros((N, d))
    b =  np.zeros(N)
    for j in range(NUM_USER):
        A[j*m:(j+1)*m] = X_split[j]
        b[j*m:(j+1)*m] = y_split[j]

    return A, b


    
    
def read_data(dataset_path, N, n, m, d, lmb,
             labels=['+1', '-1']):
    '''
    -------------------------
    Function for reading data
    -------------------------
    '''
    b = np.zeros(N)
    A = np.zeros((N, d))
    
    f = open(dataset_path, 'r')
    for i, line in enumerate(f):
        line = line.split()
        if i < N:
            for c in line:
                # labels of classes depend on the data set
                # look carefully what they exactly are
                # for a1a, a9a, w8a, w7a they are {+1, -1}
                # for phishing they are {1, 0}
                if c == labels[0]: 
                    b[i] = 1
                elif c == labels[1]:
                    b[i] = -1
                elif c == '\n':
                    continue
                else:
                    c = c.split(':')
                    A[i][int(c[0]) - 1] = float(c[1])     

    f.close()
    return A, b