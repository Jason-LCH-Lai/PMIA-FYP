import numpy as np
#from partial_mia import row0_dataset

rng = np.random.default_rng(seed=50)

def random_dataset(num_rows=5000, num_features=10):
    random_dataset = rng.random((num_rows, num_features))
    return random_dataset

def weighted_dataset(num_rows=5000, num_features=10, choices=[0,1], prob=[0.6,0.4]):
    x = rng.choice(choices, (num_rows, num_features), p=prob)
    #choice(a[, size, replace, p, axis, shuffe])
    return x

def binomial_dataset(num_rows=5000, num_features=10, prob=0.35, n=100):
    x = rng.binomial(n, p=prob, size=(num_rows, num_features))
    #(n,p[,size])
    return x

def poisson_dataset(num_rows=5000, num_features=10, lam=35):
    x = rng.poisson(lam, size=(num_rows, num_features))
    #(lam,size)
    return x

def mini_DDoS():
    x = [row0_dataset] #need to reference CSV file
    y = random.sample(x, (10,500),)

#- number of classes (+ proportion for each class)
#- number of features (+ range for each feature)
#- number of samples