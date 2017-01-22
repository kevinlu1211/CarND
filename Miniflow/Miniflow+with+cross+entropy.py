
# coding: utf-8

# # Creating a neural network from scratch

# In[1]:

import numpy as np 


# In[2]:

class Layer:
    """
    Base class for layers in the network
    
    Arguments:
        `inbound_layers`: A list of layers with edges into this class
    """
    def __init__(self, inbound_layers = []):
        
        # The list of layers with edges into the class
        self.inbound_layers = inbound_layers
        
        # The value of this layer which is calculated during the forward pass
        self.value = None
        
        # The layers that the this layer outputs to
        self.outbound_layers = []
        
        # The gradients for this layer
        # The keys are the input to this layer and their values are the 
        # partials of this layer with respect to that layer 
        self.gradients = {}
        
        # Sets this layer as an outbound layer for all of this layer's inputs
        for layer in inbound_layers: 
            layer.outbound_layers.append(self)
        
    def forward(debug = False):
        # Abstract method that should be implemented for all the derived classes
        raise NotImplementedError
        
    def backward():
        # Abstract method that should be implemented for all the derived classes 
        raise NotImplementedError


# In[3]:

class Input(Layer):
    """
    This layer accepts inputs to the neural network
    """
    def __init__(self):
        # Note here that nothing is set because these values are set during
        # the topological sort
        Layer.__init__(self)
        
    def forward(self, debug = False):
        # Do nothing because nothing is calculated
        pass
    
    def backward(self, debug = False):
        # An Input layer has no inputs so the gradient is zero 
        self.gradients = {self : 0}
        
        # Weights and bias may be inputs, so we need to sum the gradients 
        # from their outbound layers during the backward pass.
        
        # Remember that the goal is to figure out the total change in the cost function
        # with respect to a single parameter, hence the addition
  
        for n in self.outbound_layers:
#             a = self.gradients[self]
#             print(a)
#             b = n.gradients[self]
#             print(b)
            self.gradients[self] += n.gradients[self] 


# In[4]:

class Linear(Layer):
    def __init__(self, X, W, b):
        Layer.__init__(self, [X, W, b])
    
    def forward(self, debug = False):
        X = self.inbound_layers[0].value
        W = self.inbound_layers[1].value
        b = self.inbound_layers[2].value
        self.value = np.dot(X, W) + b
       
        if debug:
            print("Input to layer is:")
            print(X)

            print("Weights of layer is:")
            print(W)

            print("Bias of layer is:")
            print(b)
            
            print("XW + b is:")
            print(self.value)
            
    def backward(self, debug = False):
        
        # Initialize a partial derivative for each of the inbound_layers,
        # remembering here that this dictionary stores the partial derivative of
        # this layer with respect to the inbound layers
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        
        for n in self.outbound_layers:
            # Get the partial derivative for each of the variables in this layer 
            # with respect to the cost
            grad_cost = n.gradients[self]
            
            if debug:
                print("grad_cost is:") 
                print(grad_cost)
                
            
            
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T) 
           
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad_cost)
              
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis = 0, keepdims = False)
        
        if debug:
            print("The derivatives of the cost with respect to the inputs are:")
            print(self.gradients[self.inbound_layers[0]])
            
            print("The derivatives of the cost with respect to the weights are:")
            print(self.gradients[self.inbound_layers[1]])
            
            print("The derivatives of the cost with respect to the biases are:")
            print(self.gradients[self.inbound_layers[2]])
            
    


# In[5]:

class Sigmoid(Layer):
    def __init__(self, layer):
        Layer.__init__(self, [layer])
        
    def _sigmoid(self, x):
        return 1./(1. + np.exp(-x))
    
    def forward(self, debug = False):
        
        self.value = self._sigmoid(self.inbound_layers[0].value)
            
        if debug:
            print("Input to sigmoid layer is:")
            print(self.inbound_layers[0].value)
            
            print("Value after sigmoid activation is:")
            print(self.value)
            
        
    def backward(self, debug = False):
        self.gradients = {n : np.zeros_like(n.value) for n in self.inbound_layers}
        
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_layers[0]] += sigmoid * (1 - sigmoid) * grad_cost
        
        if debug:
            print("The derivatives of the cost with respect to the sigmoid activation is:")
            print(self.gradients[self.inbound_layers[0]])
        


# In[6]:

class MSE(Layer):
    def __init__(self, y, a):
        Layer.__init__(self, [y, a])
        
    def forward(self, debug = False):
        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1) 
    
        # get the number of samples
        self.m = self.inbound_layers[0].value.shape[0] 
        self.diff = y - a
        self.value = np.mean(self.diff**2)
 
        if debug:
            print("True value of y is:")
            print(y)

            print("Predicted value of y is:")
            print(a)
             
            print("y - a is:")
            print(self.diff)

       
    def backward(self, debug = False):
        self.gradients[self.inbound_layers[0]] = (2/self.m) * self.diff
        self.gradients[self.inbound_layers[1]] = (-2/self.m) * self.diff


# In[7]:

class Softmax(Layer):
    def __init__(self, logits):
        Layer.__init__(self, [logits])
    
    def forward(self, debug = False):
        
        exp_logits = np.exp(self.inbound_layers[0].value)
        sum_exp = np.sum(exp_logits)  
        self.value = exp_logits/sum_exp
        
        if debug:
            print("Logits are:")
            print(self.inbound_layers[0].value) 
            
            print("After exponents are:")
            exp_logits = np.exp(self.inbound_layers[0].value)
            print(exp_logits)
            
            print("Probabilities are:")
            sum_exp = np.sum(exp_logits)  
            self.value = exp_logits/sum_exp 
            print(self.value)
             
            
    
    def backward(self, debug = False):
         
        # Define gradient for inbound layers
        self.gradients = {n : np.zeros_like(n.value) for n in self.inbound_layers}
        jacobian = self._calc_jacobian(self.value)
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            if debug:
                print("grad_cost is:")
                print(grad_cost)
                print("The Jacobian is:")
                print(jacobian)
            
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, jacobian)
        
 
        if debug: 
            print("The derivative of the cost with respect to the inputs of the softmax layer is:")
            print(self.gradients[self.inbound_layers[0]])
       
    def _calc_jacobian(self, probs):
        
        # First calculate the off diagonal derivatives
        jacobian = np.dot(-1 * probs.T, probs)
        dims = jacobian.shape[0]
        
        # Now calculate the diagonal derivatives
        for i in range(dims):
            jacobian[i,i] = probs[0,i] * (1 - probs[0,i])
        return(jacobian)


# In[8]:

class CrossEntropy(Layer):
    def __init__(self, y, probs):
        Layer.__init__(self, [y, probs])
    
    def forward(self, debug = False):
        n_samples_in_batch = self.inbound_layers[0].value.shape[0]
        n_classes = self.inbound_layers[0].value.shape[1]
        
        self.y_flat = self.inbound_layers[0].value.reshape(n_samples_in_batch, n_classes)
        self.probs_flat = self.inbound_layers[1].value.reshape(n_samples_in_batch, n_classes)
       
        # Calculate the accuracy
        n_correct = np.sum(np.argmax(self.y_flat, axis = 1) == np.argmax(self.probs_flat, axis = 1))
        self.accuracy = n_correct/n_samples_in_batch
        
        # Calculate the cross entropy
        self.log_probs = np.log(self.probs_flat)
        self.cross_entropy = self.y_flat * self.log_probs
        self.value = -1 * np.sum(self.cross_entropy)
       
        
        if debug:
            print("True values y are:")
            print(self.y_flat)
            print("Probabilities are:")
            print(self.probs_flat)
            print("True value y max index are:")
            print(np.argmax(self.y_flat, axis = 1))
            print(np.argmax(self.y_flat, axis = 1).shape)
            print("Probabilities max index are:")
            print(np.argmax(self.probs_flat, axis = 1))
            print(np.argmax(self.probs_flat, axis = 1).shape) 
            print("Log probabilities are:")
            print(self.log_probs)
            print("Cross entropy is:")
            print(self.cross_entropy)
            print("Cross entropy sum is")
            print(self.value)

            
        
    
    def backward(self, debug = False):
        self.gradients[self.inbound_layers[0]] = -1 * 1/self.probs_flat
        self.gradients[self.inbound_layers[1]] = -1 * self.y_flat/self.probs_flat
        
        if debug:
            print("Gradients of cross entropy with respect to y layer is:")
            print(self.gradients[self.inbound_layers[0]])
            print("Gradients of cross entropy with respect to softmax layer is:")
            print(self.gradients[self.inbound_layers[1]])
        


# In[9]:

def topological_sort(feed_dict):
    input_layers = [n for n in feed_dict.keys()]
    
    G = {}
    
    layers = [n for n in input_layers]
    
    # Think of each element in the layer as a node, in this while loop
    # we are simply finding which layers are connected to which other layers
    while len(layers) > 0:
        # Get the first element of the array
        n = layers.pop(0)
        
        # Check if this layer n is in the dictionary if it isn't add it in
        if n not in G:
            G[n] = {'in' : set(), 'out' : set()}
        # Check if this layer m is in the dictionary if it isn't add it in 
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in' : set(), 'out' : set()}
            # Add the edges between the nodes
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)
        
    L = []
    S = set(input_layers)
    
    while len(S) > 0:
        # Get the last layer 
        n = S.pop()
        
        # Check if it is an input layer, if it is then initialize the value
        if (isinstance(n, Input)):
            n.value = feed_dict[n]
            
        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            
            # if there are no incoming edges to m then add it to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return(L)


# In[10]:

def forward_pass(graph, debug = False):
    for n in graph:
        n.forward(debug)

def backward_pass(graph, debug = False):
    for n in graph[::-1]:
        n.backward(debug) 


# In[11]:

def sgd_update(trainable, learning_rate = 1e-2): 
     
    for t in trainable:
        partial = t.gradients[t]
#         print("Partial derivatives are:")
#         print(partial)
        t.value -= learning_rate * partial 


# # Creating a neural network to classify numbers from the NMIST data set 

# In[12]:

# Import
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split


# In[13]:

# Just using keras to import the data
from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[14]:

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[15]:

# Use one hot encoding for the y label vector
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)
y_train_one_hot = lb.transform(y_train)
y_test_one_hot = lb.transform(y_test)


# In[17]:

# Normalize the grayscale image so that the values range between -0.5 and 0.5, this is so that the sigmoid activation 
# function does not saturate during training 

def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # TODO: Implement Min-Max scaling for grayscale image data
    img_min = np.min(image_data)
    img_max = np.max(image_data)
    a = -0.5
    b = 0.5
    scaled_img = a + ((image_data - img_min) * (b-a))/(img_max - img_min)
    return(scaled_img)

X_train_normalize = normalize_grayscale(X_train)
X_test_normalize = normalize_grayscale(X_test)


# In[19]:

# Now flatten the array into a 1 dimension array
X_train_normalize = np.array(X_train_normalize).reshape(60000, 28 * 28)
X_test_normalize = np.array(X_test_normalize).reshape(10000, 28 * 28 )



# In[21]:

# Now define a validation set
X_train_normalize, X_validation_normalize, y_train_one_hot, y_validation_one_hot = train_test_split(X_train_normalize, y_train_one_hot, test_size = 0.2)

# In[22]:

# Define the network
print("Defining the network")
n_hidden = 10
# n_hidden_2 = 15
n_classes = 10
n_features = X_train_normalize.shape[1]

# Initialize the weights 
print("Setting up the weights")
W1_ = np.random.normal(loc = 0, scale = 0.1, size = (n_features, n_hidden))
b1_ = np.zeros(n_hidden)
# W2_ = np.random.normal(loc = 0, scale = 0.1, size = (n_hidden, n_hidden_2))
# b2_ = np.zeros(n_hidden_2)
# W3_ = np.random.normal(loc = 0, scale = 0.1, size = (n_hidden_2, n_classes))
# b3_ = np.zeros(n_classes)

# W2_ = np.random.normal(loc = 0, scale = 1, size = (n_hidden, n_classes))
# b2_ = np.zeros(n_classes)

# Build the layers for the neural network
X, y, = Input(), Input()
W1, b1 = Input(), Input()
# W2, b2 = Input(), Input()
# W3, b3 = Input(), Input()

l1 = Linear(X, W1, b1)
# s1 = Sigmoid(l1)
# l2 = Linear(s1, W2, b2)
# s2 = Sigmoid(l2)
# l3 = Linear(s2, W3, b3)
probs = Softmax(l1)
cost = CrossEntropy(y, probs)


# Define the input layers to the neural network 
print("Setting the feed_dict")
feed_dict = {
    X: X_train,
    y: y_train_one_hot,
    W1: W1_,
    b1: b1_
#     W2: W2_,
#     b2: b2_
#     W3: W3_,
#     b3: b3_
}

print("Soring the graph nodes")
graph = topological_sort(feed_dict)


# In[29]:

# Now lets run the model
epochs = 500
steps_per_epoch = 100
batch_size = 100
show_per_step = 100
trainables = [W1, b1]
accuracies = []
print("Running the neural network")
for i in range(epochs):
    loss = 0
    print("In epoch {0}".format(i))
    for j in range(steps_per_epoch):
        # Sample a random batch of data 
        X_batch, y_batch = resample(X_train_normalize, y_train_one_hot, n_samples = batch_size)
        
        # Reset the values of X and y 
        X.value = X_batch
        y.value = y_batch
#         print(X.value)
#         print(y.value)
        
        # Now run the forward and backward propagation
#         if (j%200 == 0):
#             print("j is:")
#             print(j)
#             forward_pass(graph, debug = True) 
#             backward_pass(graph, debug = True)
#             print("Loss is {0}".format(graph[-1].value))
        forward_pass(graph, debug = True)
        backward_pass(graph, debug = True)
        
        # Update the weights of or biases and weights
        sgd_update(trainables, learning_rate = 1e-3) 
        loss += graph[-1].value

    print("Epoch: {}, Loss {:.3f}".format(i + 1, loss/steps_per_epoch))
#    print("Average loss per sample is: {0}".format(loss/(steps_per_epoch * batch_size)))
    
    # Use the validation set to see our accuracy
    X.value = X_validation_normalize
    y.value = y_validation_one_hot 
    forward_pass(graph)
    accuracies.append(graph[-1].accuracy)
    if (i%show_per_step == 0):
        print("Loss of validation set in epoch {0} is: {1}".format(i + 1, graph[-1].value)) 
        print(graph[-1].accuracy)
 

# # Test it on the test set
# X.value = X_test
# y.value = y_test
# forward_pass(graph)
# print("Loss of test set is: {0}".format(graph[-1].value))

