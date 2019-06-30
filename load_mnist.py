from mlxtend.data import loadlocal_mnist
import pandas as pd
import matplotlib.pyplot as plt
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
print ("The current working directory is %s" % dir_path)


X, y = loadlocal_mnist(images_path=dir_path+'/x_train/train-images.idx3-ubyte', 
                       labels_path=dir_path+'/y_train/train-labels.idx1-ubyte')
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

import numpy as np

print('\nDigits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

print('\n\nCreating CSV...')

np.savetxt(fname= dir_path + '/x_train/images.csv', X=X, delimiter=',', fmt='%d')
np.savetxt(fname= dir_path + '/y_train/labels.csv', X=y, delimiter=',', fmt='%d')

print('Created CSV !')

print('\nChecking...\n')
x_data = pd.read_csv(dir_path+'/x_train/images.csv').as_matrix()
y_data = pd.read_csv(dir_path+'/y_train/labels.csv').as_matrix()

print(x_data.shape)

for i in range(23,25):
    img = x_data[i]
    img.shape=(28,28)
    plt.imshow(255-img,cmap='gray')
    plt.show()
    print(y_data[i])
print('Finished!')