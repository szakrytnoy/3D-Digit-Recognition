import numpy as np
from input_funcs import prepare_single_input
from knn_wrapped import knn_classifier

test_input = './data/stroke_6_0011.csv'
x_train = np.load('x.npy')
y_train = np.load('y.npy')

x_test = prepare_single_input(test_input)
l = knn_classifier(x_test, 5, x_train, y_train)