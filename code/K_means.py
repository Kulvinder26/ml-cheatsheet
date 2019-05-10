              
# for linear algebra and vectorized operations
import numpy as np                            

# to create an artificial dataset as per requirement
from sklearn.datasets.samples_generator import make_blobs   

# plotting the data for visual understanding 
import matplotlib.pyplot as plt

# Utility functions
from copy import deepcopy                

#For finding the label for the given point based on the distance to nearest centroid
def give_label(X,centroid,n_centroid):     
    label = [None]*X.shape[0]
    for i in range(X.shape[0]):
        k = [None]*centroid.shape[0]
        for j in range(centroid.shape[0]):
            c = dist(X[i],centroid[j],None)
            k[j] = c
        e = list(np.where(k == np.amin(k)))
        label[i] = e[0]
        
    label = [e for e, in label]

    X0 = X[:,0]
    X1 = X[:,1]
    color = []
    
    for l in label:
        if l == 0:
            color.append('green')
        elif l == 1:
            color.append('blue')
        elif l == 2:
            color.append('red')
        elif l == 3:
            color.append('orange')
        else:
            color.append('black')
    k = ['*','P','X','v','D']
    for j in range(n_centroid):
        plt.plot(centroid[j][0],centroid[j][1],k[j],linewidth=20, markersize=20)
    plt.scatter(X0,X1,linewidths=4 ,color = color)
    plt.ylabel('X[1]')
    plt.xlabel('X[0]')
    plt.show()
    return label
    
    
# For taking the mean of all the values at every single centroid to compute new centroid
def move_label(X,centroid,c):               
    for i in range(centroid.shape[0]):
        values = np.array(c)
        searchval = i
        ii = np.where(values == searchval)[0]
        k = np.array(X[ii])
        no_val = len(ii)
        Total_sum0 = np.sum(k[:,0])
        Total_sum1 = np.sum(k[:,1])
        Total_sum = np.array([Total_sum0,Total_sum1])
        avg = Total_sum/no_val
        centroid[i] = avg
    return centroid

def K_means(X,n_centroid):
    
    # Randomly initialising the centroids
    centroid = np.random.uniform(-1,2,(n_centroid,X.shape[1]))    

    # Displaying Initial centroids
    print("Initial Centroid : ")         
    for i in range(n_centroid):
        print("For Class ",i," : ",centroid[i])


    C_old = np.zeros(centroid.shape)
    error = dist(centroid, C_old, None)

    # Runs both functions till centroids converges
    while(error != 0):                   

         c = give_label(X,centroid,n_centroid)
         C_old = deepcopy(centroid)
         centroid = move_label(X,centroid,c)
         error = dist(centroid, C_old, None)
   
     # Displaying Final centroids
    print("Final Optimal Centroid : ")    
    for i in range(n_centroid):
         print("For Class ",i," : ",centroid[i]) 

# For computing L-2 distance between two points
def dist(a, b, ax=1):                     
    return np.linalg.norm(a - b, axis=ax)


# for better visual intution, the given implementation works for cases where number of centroid is in range [1,5], 
# Just change the case you want to see like n_centroid = 2 or 3 or 4 etc.
n_centroid = 3    
X, _ = make_blobs(n_samples=200, centers=n_centroid, n_features=2,random_state=0)  # Creates data points for any classification based algorithm

K_means(X,n_centroid)

