import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.patches as patch

np.random.seed(64)

df = pd.read_csv('CSE575-HW03-Data.csv')
df_array = np.array(df)

class GMM:
    def __init__(self, k, max_it=25):
        self.k = k
        self.max_iteration = max_it

    def initialize(self, data):
        self.shape = data.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        row_data = np.random.randint(low=0, high=self.shape[0], size=self.k)
        self.mu = [data[row_id,:] for row_id in row_data]
        self.sigma = [np.cov(data.T) for _ in range(self.k)]

    def mu_sig_const(self, data):
        self.weights = self.predict_proba(data)
        self.phi = self.weights.mean(axis=0)
    
    def phi_w_const(self, data):
        for i in range(self.k):
            wt = self.weights[:, [i]]
            sum_weight = wt.sum()
            self.mu[i] = (data * wt).sum(axis=0) / sum_weight
            self.sigma[i] = np.cov(data.T, aweights=(wt/sum_weight).flatten(), bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for i in range(self.max_iteration):
            self.mu_sig_const(X)
            self.phi_w_const(X)
            
    def predict_proba(self, data):
        likelihood = np.zeros( (self.shape[0], self.k) )
        for i in range(self.k):
            dist = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = dist.pdf(data)
        
        num = likelihood * self.phi
        den = num.sum(axis=1)[:, np.newaxis]
        wts = num / den
        return wts
    
    def predict(self, data):
        wts = self.predict_proba(data)
        return np.argmax(wts, axis=1)

gmm_estimator = GMM(2, 10)
gmm_estimator.fit(df_array)
results=gmm_estimator.predict(df_array)

cluster_1_data = []
cluster_2_data = []
for p in range(len(df_array)):
    if  results[p] == 0:
        cluster_1_data.append(df_array[p])
    else:
        cluster_2_data.append(df_array[p])
cluster_1_data = np.array(cluster_1_data)
cluster_2_data = np.array(cluster_2_data)

plt.scatter(cluster_1_data[:,0],cluster_1_data[:,1],color='red')
plt.scatter(cluster_2_data[:,0],cluster_2_data[:,1],color='blue')

d1 = patch.Patch(color='red', label='Cluster1')
d2 = patch.Patch(color='blue', label='Cluster2')

plt.legend(handles=[d1, d2])
plt.title('Plot - GMM at k=2')
plt.savefig('GMM_K_equal_2.png')