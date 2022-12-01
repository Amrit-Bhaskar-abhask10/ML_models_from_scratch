import time
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import accuracy_score

imagefile = 'mnist/mnist_uncompressed/train-images-idx3-ubyte'
X_train_2d = idx2numpy.convert_from_file(imagefile)

imagefile = 'mnist/mnist_uncompressed/train-labels-idx1-ubyte'
y_train = idx2numpy.convert_from_file(imagefile)

imagefile = 'mnist/mnist_uncompressed/t10k-images-idx3-ubyte'
X_test_2d = idx2numpy.convert_from_file(imagefile)

imagefile = 'mnist/mnist_uncompressed/t10k-labels-idx1-ubyte'
y_test = idx2numpy.convert_from_file(imagefile)


X_train = X_train_2d.reshape((X_train_2d.shape[0], 
                                   X_train_2d.shape[1] * X_train_2d.shape[2]))
X_train = np.asarray(X_train).astype(np.float32)


X_test = X_test_2d.reshape((X_test_2d.shape[0], 
                                 X_test_2d.shape[1] * X_test_2d.shape[2]))
X_test = np.asarray(X_test).astype(np.float32)

Y_train = y_train.reshape(-1,1)
Y_train = np.asarray(Y_train).astype(np.int32)

Y_test = y_test.reshape(-1,1)
Y_test = np.asarray(Y_test).astype(np.int32)

Y_train.shape, X_train.shape, Y_test.shape, X_test.shape


class kNN():
    def _init_(self):
        pass

    #Train function
    def train_func(self, X, y):
        self.X_train = X
        self.Y_train = y
    
    #Calculate the euclidean distance
    def EuclideanDistance(self, X):
        sq_sum_test = np.square(X).sum(axis = 1)
        sq_sum_train = np.square(self.X_train).sum(axis = 1)
        dot_product = np.dot(X, self.X_train.T)
        euclid_dist = np.sqrt(-2 * dot_product + sq_sum_train + np.matrix(sq_sum_test).T)
        return(euclid_dist)

    #Calculate prediction
    def predict_func(self, X, k=1):
        dist = self.EuclideanDistance(X)
        num_test = dist.shape[0]
        Y_pred = np.zeros(num_test)

        for i in range(num_test):
            kClosestNeighbor = []
            neighbors = self.Y_train[np.argsort(dist[i,:])].flatten()
            kClosestNeighbor = neighbors[:k]
            c = Counter(kClosestNeighbor)
            Y_pred[i] = c.most_common(1)[0][0] #mode of values
        return(Y_pred)

#Given k-values in the problem
k_vals = [1, 3, 5, 10, 20, 30, 40, 50, 60]
accuracy = []

a = time.time()
for k in k_vals:
    print(k)
    classifier = kNN()
    classifier.train_func(X_train, Y_train)
    predictions = []
  
    predictions = list(classifier.predict_func(X_test, k))
    
    acc = accuracy_score(Y_test, predictions)
    accuracy.append(acc)
    

    print(f"K-value :\t{k}")
    print("Minutes taken:\t", (time.time()-a)/60)
    print("Accuracy: ", acc)
    print("Accuracy list updates: ", accuracy)
    a = time.time()


plt.plot(k_vals, accuracy, marker = 'o')
plt.title('Plot - K values v/s Accuracy')
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.xticks(k_vals)
plt.savefig('knn_k_vs_acc_image.png')



