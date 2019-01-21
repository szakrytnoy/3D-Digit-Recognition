# 3D-Digit-Recognition
The practical assignment requires us to recognize and classify the given hand-written digits based on the 3-D location time series. For this purpose, we have used Python as our implementation programming language with NumPy, Pandas and SciPy library functions.

## Information on the given data
We were provided with training dataset where the location information of each stroke of the experiment had been stored into separate comma-separated values (CSV) file. The strokes provided were for numbers 0 to 9 with 100 different observations for each number
(1000 observations in total). The dataset was such that each stroke had different counts based on the time taken by the subject.

## Model Selection
While selecting the model for our manual implementation, we have used high level Python libraries to research different approaches. The model in bold has been chosen.

| Solution  | Results summary |
| ------------- | ------------- |
| Mnist-style feature extraction (28x28 matrix of binary values) + Deep Neural Network in Tensorflow + Keras  | High accuracy (~90%), complex implementation  |
| Custom feature extraction (distance & angle between the first and the last point, variance on x and y axes) + Neural Network | Low accuracy (~60%), complex implementation  |
| Custom feature extraction (distance & angle between the first and the last point, variance on x and y axes) + KNN Classifier in Scikitlearn | Low accuracy (~60%), complex implementation  |
| **Resampling and unit vectors extraction for movements + KNN Classifier** | **High accuracy (~94%), simple implementation**  |

## Data Pre-processing and Feature Extraction (input_funcs.py)
We start with automation of filename generation for the strokes for proper dataset parsing.

```
def get_filenames():
    file_names = [] #Empty list to store file names
    basis = 'stroke'
    width = 4
    for number in range(10):
        for i in range(1, 101):
            i_str = f'{i:{width}}'
            i_str = i_str.replace(' ', '0')
            
            name = basis + '_' + str(number) + '_' + i_str
            file_names.append(name)
    return file_names
```

We then process and parse the data. After importing the csv file into a temporary pandas DataFrame, we name the 3 columns into x, y, and z. These columns represent the coordinates of the digits. We first drop the z coordinates from our dataset.

```
def prepare_input(path_to_folder, file_names):
    x = [] #Empty list to store samples
    
    nparts = 10 #Number of segments for resampling
    for name in file_names:
        path = path_to_folder + name + '.csv' #Generate file-specific path
        
        temp = pd.read_csv(path, engine = 'python') #Read csv into a dataframe
        temp.columns = ['x', 'y', 'z'] #Name the columns: x, y, z
        temp.drop('z', axis = 1, inplace = True) #Drop the third dimension
        temp.drop_duplicates(inplace = True) #Remove duplicates
```

We do this because the z coordinates do not provide any value for the classification of our dataset. We thus decided to work with the x and y coordinates only. We make sure there are no duplicate values in the dataset.

```
        #Resample the observation into nparts data points
        temp = pd.DataFrame(scipy.signal.resample(temp, nparts+1), columns = ['x', 'y'])
        
        #Min-Max normalization into a new DataFrame
        temp_norm = (temp - temp.min())/(temp.max() - temp.min())
```

The dataset is then divided into n = 10 datapoints. We resample the data as the number of counts in each stroke for every digit is different. There are some strokes that have 17 counts while some have more than 220 counts in their observations So, we use the SciPy library function to resample the datapoints. We then use Min-max normalization feature scaling on the new data-frame so that the values of the features of the dataset are scaled to a fixed range. In this program we do not use PCA for dimensionality reduction as knowing the max variance between the 2 coordinates does not provide us with a good reading.

```
        #Get the difference -- track movement of the finger
        temp_diff = temp_norm.diff().dropna() 
        
        #Getting x and y values for UNIT vectors of each movement
        temp_diff['factor'] = (temp_diff['x']**2+temp_diff['y']**2)**0.5
        temp_unit = pd.DataFrame(columns = ['x', 'y'])
        temp_unit['x'] = temp_diff['x']/temp_diff['factor']
        temp_unit['y'] = temp_diff['y']/temp_diff['factor']
```

Now we try to get the values for the unit vectors for each of the 10 movements. We first get the differences between the points for both x and y coordinates. Then we find the length of the vector points, where A and B are the different points in the database. We divide this factor with the difference values for both x any y to get the unit vector values. 

```
        #Flatten nparts unit vectors into a long 1d array
        sample = np.array(temp_unit).flatten().tolist()
        x.append(sample) #Append the 1d array into a list of all samples
```

We then flatten the data we have to a list. Flattening is when we take the rows of data and combine then together in order to get a single 1-D vertical vector. As we are processing all the datasets in cycles, we append the data so that at the end we have 1000 rows of data points, in the main file, after which we convert final list into a 2-D array (named x). We create a vector of labels for the y
values.

Thus, the features that were selected for this program were unit vectors (directions) of movements after resampling. They are represented in x and y coordinates, which are then flattened (the order of the coordinates does not matter since we are going to measure distances between samples). 

## Input call Script (prepare_train.py)

```
import numpy as np
from input_funcs import get_filenames, prepare_input

x, y = prepare_input('./data/', get_filenames())
np.save('x', x)
np.save('y', y)
```

This script of ours calls all the input functions and saves the extracted features in temporary files. Since the training dataset is not modified during testing, this makes the features easier and faster to access.

## Classification (knn_wrapped.py)
The classifier we have used for this program in the K-nearest Neighbor classifier. 

KNN classifier is one of the most commonly used classifiers and one of the least computationally demanding. It works such a way that it does not make any assumptions about the underlying data neither uses the training data for any generalizations. For classification the KNN relies on the distance between the feature vectors, where it classifies unknown data points by finding the most common class among the k closest neighbors and classifying it to the class with the most numbers.

```
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
```

For getting the predicted class, we iterate from 1 to total number of test data points (if there are more than 1) and within the loop find distances between each test sample and all the training samples.

We can use either Euclidean distance or Manhattan distance as the distance metrics. Although empirically we have derived higher accuracy with the Manhattan distance metric, there is also the possibility of using Euclidean distance in our function:

```
def get_distance(x1, x2, how = 'euclidean'): #Where x1 and x2 are vectors
    if how == 'euclidean':
        distance = (sum((x1 - x2)**2))**0.5 #Square root OF sum of squares
    elif how == 'manhattan':
        distance = sum(abs(x1 - x2)) # Sum of absolute differences
    else:
        print('Wrong method. Select euclidean or manhattan')
    return distance
```

We then sort the distances in each row of the array, select the k first elements and find the most frequent label corresponding to the selected datapoints from the training set.

## Testing (test.py)

```
import numpy as np
from input_funcs import prepare_single_input
from knn_wrapped import knn_classifier

test_input = './data/stroke_6_0011.csv'
x_train = np.load('x.npy')
y_train = np.load('y.npy')

x_test = prepare_single_input(test_input)
l = knn_classifier(x_test, 5, x_train, y_train)
```

This script is where all the previous functions are wrapped together. It calls for the importing of the temporary files as well as the classifier function. The test samples input can be anything from 1 to n rows of extracted features. Feature extraction for the test sample is realized in a separate function which is similar to the one used in preprocessing of training dataset. 

## Results
Careful feature extraction and well-thought-out solution design allows for quick (~ 3 sec) and accurate (expected accuracy ~ 94%) classification of 3-D Digits.
