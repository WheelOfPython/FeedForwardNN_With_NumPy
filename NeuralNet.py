import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 

np.random.seed(222)
dir_path = os.path.dirname(os.path.realpath(__file__))

class NeuralNetwork():
    def __init__(self, X, Y):
        self.X_data = X # Should be Normilized !!!
        self.Y_data = Y # Manipulate Y data, so that they are in the form of what we want the computer to output
        print('Input shape: ' , self.X_data.shape)
        print('Output shape: ', self.Y_data.shape)
        
        self.layer_sizes = [self.X_data.shape[1], 16, 16, self.Y_data.shape[1]] # Should check for compatability of dimensions
        self.layer_num = len(self.layer_sizes)-1
        self.scalar = 0.01
        
        self.z_list = ['empty' for _ in range(len(self.layer_sizes))] # Layers
        self.a_list = ['empty' for _ in range(len(self.layer_sizes))] # Activated Layers
        self.w_list = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) for i in range(self.layer_num)] # Weights
        self.b_list = [np.random.randn(self.X_data.shape[0], self.layer_sizes[i+1]) for i in range(self.layer_num)] # Bias
        
        self.delta_list = ['empty' for _ in range(self.layer_num)] # Errors
        self.dJdW_list = ['empty' for _ in range(self.layer_num)]  # Gradients
    
    def FeedForword(self, inpt_data):
        self.z_list[0] = inpt_data
        self.a_list[0] = inpt_data # No act!!!
		
        for l in range(self.layer_num):
            self.z_list[l+1] = np.dot(self.a_list[l], self.w_list[l]) + self.b_list[l] # Not w*a
            self.a_list[l+1] = self.act(self.z_list[l+1])
        
        final_layer = self.a_list[-1]
        self.J = np.sum(self.cost(final_layer, self.Y_data))
    
    def Backpropagation(self):
        self.delta_list[-1] = np.multiply(self.cost_prime(self.a_list[-1], self.Y_data), self.act_prime(self.z_list[-1]))
        self.dJdW_list[-1] = np.dot(self.a_list[-2].T, self.delta_list[-1]) #a_list[-2].T!!!
		
        for l in range(2, len(self.layer_sizes)):
            self.delta_list[-l] = np.multiply(np.dot(self.delta_list[-l+1],self. w_list[-l+1].T), self.act_prime(self.z_list[-l]))
            self.dJdW_list[-l] = np.dot(self.a_list[-l-1].T, self.delta_list[-l])
    
    def UpdateWeights(self):
        for i in range(self.layer_num):
            self.w_list[i] -= self.scalar * self.dJdW_list[i]
            self.b_list[i] -= self.scalar * self.delta_list[i]
    
    def Train(self):
        xs=[]
        ys=[]
        print('\nStart Training...\n')
        max_epochs = 10000
        for epoch in range(max_epochs):
            self.FeedForword(self.X_data)
            self.Backpropagation()
            self.UpdateWeights()
            xs.append(epoch)
            ys.append(self.J)
            if epoch % 1000 == 0 and epoch != 0:
                print(epoch, '/', max_epochs)
        print('Finished Training')
        plt.plot(xs[1:],ys[1:])
        plt.show()
        
    def Predict(self, inpt, outpt):
        self.FeedForword(inpt)
        m = np.around(self.a_list[-1]) # Round from float to 0 or 1
        
        s=0.0
        for row in range(len(m)):
            if set(m[row]) == set(outpt[row]):
                s+=1
        print('Correct: ', int(s), '/', len(m))
        print('Accuracy: ', s/len(m) * 100, '%')
        
	#-------Helper Functions-------
    def act(self, x):
        return 1/(1+np.exp(-x))
    def act_prime(self, x):
        return np.exp(-x)/(1+np.exp(-x))**2
    
    def cost(self, guess, goal):
        return 0.5*(guess-goal)**2
    def cost_prime(self, guess, goal):
        return guess-goal


#--------------Import MNIST data----------------
mnist_x_raw = pd.read_csv(dir_path + '/x_train/images.csv').values # Matrix(5999, 784)
mnist_y_raw = pd.read_csv(dir_path + '/y_train/labels.csv').values # Matrix(5999, 1)

mnist_x_train_raw = np.array(mnist_x_raw[:200])
mnist_y_train_raw = mnist_y_raw[:200]

#Check for Errors
img = np.array(mnist_x_train_raw[0]) # New array!!!
img.shape=(28,28)
plt.imshow(255-img,cmap='gray')
plt.show()
print('Image should show: ', mnist_y_train_raw[0])
print(' ')

#--------------Prepare Training Data----------------
mnist_x_train = mnist_x_train_raw/255 # Normalize
mnist_y_train = np.zeros((mnist_y_train_raw.shape[0], 10))

# Manipulate Form of Y data
for i in range(mnist_y_train.shape[0]):
    real_num = mnist_y_train_raw[i]
    if real_num == 0:
        mnist_y_train[i] = [1,0,0,0,0,0,0,0,0,0]
    if real_num == 1:
        mnist_y_train[i] = [0,1,0,0,0,0,0,0,0,0]
    if real_num == 2:
        mnist_y_train[i] = [0,0,1,0,0,0,0,0,0,0]
    if real_num == 3:
        mnist_y_train[i] = [0,0,0,1,0,0,0,0,0,0]
    if real_num == 4:
        mnist_y_train[i] = [0,0,0,0,1,0,0,0,0,0]
    if real_num == 5:
        mnist_y_train[i] = [0,0,0,0,0,1,0,0,0,0]
    if real_num == 6:
        mnist_y_train[i] = [0,0,0,0,0,0,1,0,0,0]
    if real_num == 7:
        mnist_y_train[i] = [0,0,0,0,0,0,0,1,0,0]
    if real_num == 8:
        mnist_y_train[i] = [0,0,0,0,0,0,0,0,1,0]
    if real_num == 9:
        mnist_y_train[i] = [0,0,0,0,0,0,0,0,0,1]

#--------------Make & Train Network----------------
nn = NeuralNetwork(mnist_x_train, mnist_y_train)
nn.Train()


#--------------Make & Predict Sample----------------
mnist_x_sample_raw = np.array(mnist_x_raw[300:500])
mnist_y_sample_raw = mnist_y_raw[300:500]
mnist_x_sample = mnist_x_sample_raw/255
mnist_y_sample = np.zeros((mnist_y_sample_raw.shape[0], 10))
for i in range(mnist_y_sample.shape[0]):
    real_num = mnist_y_sample_raw[i]
    if real_num == 0:
        mnist_y_sample[i] = [1,0,0,0,0,0,0,0,0,0]
    if real_num == 1:
        mnist_y_sample[i] = [0,1,0,0,0,0,0,0,0,0]
    if real_num == 2:
        mnist_y_sample[i] = [0,0,1,0,0,0,0,0,0,0]
    if real_num == 3:
        mnist_y_sample[i] = [0,0,0,1,0,0,0,0,0,0]
    if real_num == 4:
        mnist_y_sample[i] = [0,0,0,0,1,0,0,0,0,0]
    if real_num == 5:
        mnist_y_sample[i] = [0,0,0,0,0,1,0,0,0,0]
    if real_num == 6:
        mnist_y_sample[i] = [0,0,0,0,0,0,1,0,0,0]
    if real_num == 7:
        mnist_y_sample[i] = [0,0,0,0,0,0,0,1,0,0]
    if real_num == 8:
        mnist_y_sample[i] = [0,0,0,0,0,0,0,0,1,0]
    if real_num == 9:
        mnist_y_sample[i] = [0,0,0,0,0,0,0,0,0,1]

nn.Predict(mnist_x_sample, mnist_y_sample)

#Print the weights of the first layer as image to show patterns
for i in range(1):
    img = np.array(nn.w_list[0].T[i]) # New array!!!
    img.shape=(28,28)
    plt.imshow(255-img,cmap='gray')
    plt.show()
