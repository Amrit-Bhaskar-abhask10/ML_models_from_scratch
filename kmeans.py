import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
np.random.seed(64)
rd.seed(64)

df = pd.read_csv("CSE575-HW03-Data.csv",header=None)
df_array = np.array(df)


class Kmeans:
    def __init__(self,X,K):
        self.X=X
        self.shape = X.shape
        self.Output={}
        self.Centroids=np.array([]).reshape(self.X.shape[1],0)
        self.K=K
        self.m=self.X.shape[0]
        self.l,self.n=self.shape
        self.loss = 0
        
    def intial_centroid(self,X,K):
        rand_val=rd.randint(0,X.shape[0])
        temporary_centroid=np.array([X[rand_val]])

        for k in range(1,K):
            A=np.array([]) 
            for x in X:
                A=np.append(A,np.min(np.sum((x-temporary_centroid)**2)))
            prob=A/np.sum(A)
            cum_prob=np.cumsum(prob)
            random_y=rd.random()
            rand_val = 0
            for r,s in enumerate(cum_prob):
                if random_y < s:
                    rand_val = r
                    break
            temporary_centroid=np.append(temporary_centroid,[X[rand_val]],axis=0)
        return temporary_centroid.T
    
    def fit(self,n_iter):
        
        self.Centroids=self.intial_centroid(self.X,self.K)
        prev_ctr = np.zeros((self.m,self.K))
        prev_loss = 0
        count = 0
        
        for n in range(n_iter):
            Euc_Dist=np.array([]).reshape(self.m,0)
            for k in range(self.K):
                add_dist=np.sum((self.X-self.Centroids[:,k])**2,axis=1)
                Euc_Dist=np.c_[Euc_Dist,add_dist]
            C=np.argmin(Euc_Dist,axis=1)+1
            self.loss = 0
            for i in range(len(Euc_Dist)):
                self.loss += Euc_Dist[i][C[i]-1]
            store_cluster_data={}
            for k in range(self.K):
                store_cluster_data[k+1]=np.array([]).reshape(self.n,0)
            for i in range(self.m):
                store_cluster_data[C[i]]=np.c_[store_cluster_data[C[i]],self.X[i]]#store data to its nearest cluster centr
        
            for k in range(self.K):
                store_cluster_data[k+1]=store_cluster_data[k+1].T  #Transposing cluster data
            for k in range(self.K):
                self.Centroids[:,k]=np.mean(store_cluster_data[k+1],axis=0)
            
            if((np.array_equal(self.Centroids, prev_ctr) == True) & (prev_loss == self.loss)):
                # print(self.K,n)
                count+=1
            else:
                count=0
            
            if(count==5):
                break ## break when optimum cluster is found
                
            prev_ctr = self.Centroids.copy()
            prev_loss = self.loss.copy()
            self.Output = store_cluster_data
            
    
    def predict(self):
        return self.Output,self.loss
    
km = Kmeans(df_array,2)
ft = km.fit(50)
output, loss = km.predict()


cluster_1_data = output[1]
cluster_1_data = cluster_1_data[:,0:2]
cluster_2_data = output[2]
cluster_2_data = cluster_2_data[:,0:2]

plt.figure(0)
plt.scatter(cluster_1_data[:,0],cluster_1_data[:,1],color='red')
plt.scatter(cluster_2_data[:,0],cluster_2_data[:,1],color='blue')
plt.title('Plot - Cluster data for K = 2')
first = patch.Patch(color='red', label='Cluster_1')
second = patch.Patch(color='blue', label='Cluster_2')
plt.legend(handles=[first, second])
plt.savefig('kmeans_k_equals_2.png')


K = [2,3,4,5,6,7,8,9]
obj_func = []
for k in K:
    kms = Kmeans(df_array, k)
    result = kms.fit(50)
    output,loss = kms.predict()
    obj_func.append(loss)
    print(f"K-value: {k}\t,Loss value: {loss}")


plt.figure(1)
plt.plot(K, obj_func, marker = 'o')
plt.xlabel('K values')
plt.ylabel('Objective Function (J) - Loss param')
plt.title('Plot - Objective Function_vs_K')
plt.savefig('kmeans_elbow_curve.png')