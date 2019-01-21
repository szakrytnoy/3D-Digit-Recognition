import numpy as np
import pandas as pd
import scipy.signal

""" This piece of code just generates filenames """
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

""" Parse the data and prepare input """
def prepare_input(path_to_folder, file_names):
    x = [] #Empty list to store samples
    
    nparts = 10 #Number of segments for resampling
    for name in file_names:
        path = path_to_folder + name + '.csv' #Generate file-specific path
        
        temp = pd.read_csv(path, engine = 'python') #Read csv into a dataframe
        temp.columns = ['x', 'y', 'z'] #Name the columns: x, y, z
        temp.drop('z', axis = 1, inplace = True) #Drop the third dimension
        temp.drop_duplicates(inplace = True) #Remove duplicates
        
        #Resample the observation into nparts data points
        temp = pd.DataFrame(scipy.signal.resample(temp, nparts+1), columns = ['x', 'y'])
        
        #Min-Max normalization into a new DataFrame
        temp_norm = (temp - temp.min())/(temp.max() - temp.min())
        
        #Get the difference -- track movement of the finger
        temp_diff = temp_norm.diff().dropna() 
        
        #Getting x and y values for UNIT vectors of each movement
        temp_diff['factor'] = (temp_diff['x']**2+temp_diff['y']**2)**0.5
        temp_unit = pd.DataFrame(columns = ['x', 'y'])
        temp_unit['x'] = temp_diff['x']/temp_diff['factor']
        temp_unit['y'] = temp_diff['y']/temp_diff['factor']
        
        #Flatten nparts unit vectors into a long 1d array
        sample = np.array(temp_unit).flatten().tolist()
        x.append(sample) #Append the 1d array into a list of all samples
    
    x = np.array(x) #Convert the list of lists into a 2d array
    
    #Observations are ordered -- create vector of labels
    y = np.empty((1000))
    for i in range(10):
        y[i*100 : (i+1)*100] = i
        
    return x, y