# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 23:19:18 2020

@author: WTDSP_3YearsGoodLuck
"""
# In[1]:
### START CODE HERE ### (~ 1 line of code)
test = "Hello World"
### END CODE HERE ###
print ("test: " + test)
# In[2]:
# GRADED FUNCTION: basic sigmoid
import math
def basic_sigmoid(x):
    """
    Compute sigmoid of x.
    Arguments:
    x -- A scalar
    Return:
    s -- sigmoid(x)
    """ 
    s=1/(1+math.exp(-x))
    return s
basic_sigmoid(3)
# In[3]:
import numpy as np
# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))
# In[4]:
# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)
# In[5]
# GRADED FUNCTION: sigmoid
import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()
def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """
    ### START CODE HERE ### (~ 1 line of code)
    s=1/(1+np.exp(-x))
    ### END CODE HERE ###
    return s
######
x = np.array([1, 2, 3])
sigmoid(x)  
# In[6]:
    #3.2 Sigmoid gradient
# GRADED FUNCTION: sigmoid derivative
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function
    with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to
    calculate the gradient.
    Arguments:
    x -- A scalar or numpy array
    Return:
    ds -- Your computed gradient.
    """
    ### START CODE HERE ### (~ 2 lines of code)
    s =1/(1+np.exp(-x))
    ds =s*(1-s)
    ### END CODE HERE ###
    return ds
#####
x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
# In[7]:
    #3.3 Reshaping arrays
# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    ### START CODE HERE ### (~ 1 line of code)
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2],1)) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
    ### END CODE HERE ###
    return v 
###
# This is a 3 by 3 by 2 array, typically images will be (num px x, num px y,3) where 3represents the RGB values
image = np.array([[[ 0.67826139, 0.29380381],
[ 0.90714982, 0.52835647],
[ 0.4215251 , 0.45017551]],
[[ 0.92814219, 0.96677647],
[ 0.85304703, 0.52351845],
[ 0.19981397, 0.27417313]],
[[ 0.60659855, 0.00533165],
[ 0.10820313, 0.49978937],
[ 0.34144279, 0.94630077]]])
print ("image2vector(image) = " + str(image2vector(image)))
# In[8]:  
#3.4 Normalizing rows 
# GRADED FUNCTION: normalizeRows
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Argument:
    x -- A numpy matrix of shape (n, m)
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    ### START CODE HERE ### (~ 2 lines of code)
    # Compute x norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ...,keepdims = True)
    x_norm = np.linalg.norm(x,axis = 1,keepdims = True)
    x_norm
    # Divide x by its norm.
    x =x/x_norm
    ### END CODE HERE ###
    return x   
#######
x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " +str(normalizeRows(x)))    
# In[9]:    
#3.5 Broadcasting and the softmax function
# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (n, m).
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    ### START CODE HERE ### (~ 3 lines of code)
    import numpy as np
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    # Create a vector x sum that sums each row of x exp. Use np.sum(..., axis = 1,keepdims = True).
    x_sum =np.sum(x_exp, axis = 1,keepdims = True)
    # Compute softmax(x) by dividing x exp by x sum. It should automatically use numpy broadcasting.
    s =x_exp/x_sum
    print(x_exp)
    print(x_exp.shape)
    print(x_sum)
    print(x_sum.shape)
    ### END CODE HERE ###
    return s
#####OUTPUT
x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))   
# In[10]:  
#4 Vectorization
import time
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms"
)
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
            outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) +
"ms")
### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str
(1000*(toc - tic)) + "ms")
### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#####MORE EASY WAY #####
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms"
)
### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) +
"ms")
### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str
(1000*(toc - tic)) + "ms")
### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
# In[11]:
#4.1 Implement the L1 and L2 loss functions
# GRADED FUNCTION: L1
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    ### START CODE HERE ### (~ 1 line of code)
    loss = np.sum(np.abs(y-yhat))
    ### END CODE HERE ###
    return loss
####OUTPUT
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
# In[12]:
# GRADED FUNCTION: L2
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    ### START CODE HERE ### (~ 1 line of code)
    loss =np.dot((y-yhat),(y-yhat))
    ### END CODE HERE ###
    return loss
####OUTPUT
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
