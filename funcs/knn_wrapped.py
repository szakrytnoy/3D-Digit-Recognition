import numpy as np

def knn_classifier(x_test, k, x_train, y_train):
    n_train = x_train.shape[0] #Number of rows in training data     
    
    if len(x_test.shape) == 1:
        n_test = 1 #If there are no columns, than we have only 1 unput row
    else:
        n_test = x_test.shape[0] #Number of rows in test data
    
    label = np.zeros((n_test))
    d = np.zeros((n_test, n_train)) #Predefine array of distances
    for i in range(n_test): #For each test sample
        for j in range(n_train): #For each train sample
            d[i, j] = get_distance(x_test[i], x_train[j], how = 'manhattan') #Get the distance
        
        i_knn = np.argsort(d[i])[:k] #Get indices of sorted array, select k first
        label[i] = np.bincount(y_train[i_knn].astype(int)).argmax()
        #Find the most ocurring element in the labels
    return label.astype(int)
    
def get_distance(x1, x2, how = 'euclidean'): #Where x1 and x2 are vectors
    if how == 'euclidean':
        distance = (sum((x1 - x2)**2))**0.5 #Square root OF sum of squares
    elif how == 'manhattan':
        distance = sum(abs(x1 - x2)) # Sum of absolute differences
    else:
        print('Wrong method. Select euclidean or manhattan')
    return distance