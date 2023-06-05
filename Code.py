import numpy
import scipy.special
import matplotlib.pyplot
%matplotlib inline
#neural network class defenition 
class neuralNetwork:
    
    
    #initialise the neural network 
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
      
    
        self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
                  
                  
                  
        self.activation_function= lambda x: scipy.special.expit(x)
        pass
    
        #train the neural network
    def train(self,inputs_list,targets_list):
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T
        
        #forwardprop
        
        #calculates z1
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculates a1
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculates z2
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculates a2
        final_outputs=self.activation_function(final_inputs)
        
        #backwardprop
        
        #calculates c
        output_errors=(final_outputs-targets)**2
        #calculates deriv of c with respect to a2
        output_errors_deriv=2*(final_outputs-targets)
        #calculates deriv of a2 with repect to z2
        final_outputs_deriv=final_outputs*(1-final_outputs)
        #calculates deriv of z2 with respect to a1
        final_input_deriv=self.who.T
        #calculates c with respect to a1
        hidden_errors=numpy.dot(final_input_deriv,(output_errors_deriv*final_outputs_deriv))
       
    
        #calculating cost of weights
        
        #calculating deriv of z2 with respect to who
        final_inputs_deriv2=hidden_outputs.T
        
        #cost for weights between hidden and output layer
        self.who +=self.lr*numpy.dot((output_errors_deriv*final_outputs_deriv),final_inputs_deriv2)
        
        #calculates deriv of c with respect to a1
        hidden_errors_deriv=hidden_errors
        
        #calculates deriv of a1 with respect to z1
        hidden_outputs_deriv=hidden_outputs*(1-hidden_outputs)
        
        #calculates deriv of z1 with respect to wih
        hidden_inputs_deriv=inputs.T
        
        #cost for weights between input and hidden layer
        self.wih +=self.lr*numpy.dot((hidden_errors_deriv*hidden_outputs_deriv),hidden_inputs_deriv)
        
        pass
        
    
    
    #query the neural network 
    def query(self,inputs_list):
        inputs=numpy.array(inputs_list,ndmin=2).T         
                  
                  
                  
        hidden_inputs=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        
                  
        return final_outputs
                  


#number of input hidden and outputnodes 
input_nodes=784
hidden_nodes=100
output_nodes=10 
learning_rate=0.3


#create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


#opens training data file  
data_file=open("mnist_train.csv",'r')
data_list=data_file.readlines()
data_file.close()

for record in data_list:
    #forms training data into sutible array 
    all_values = record.split(',')
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    targets=numpy.zeros(output_nodes)+0.01

    #all_values[0] is the target label for this record
    targets[int(all_values[0])]=0.99
    n.train(inputs,targets)
    pass


#opens test data file 
data_file_test=open("mnist_test.csv",'r')
data_list_test=data_file_test.readlines()
data_file_test.close()


#get the first test record 
all_values_test= data_list_test[0].split(',')

#print the label 
print(all_values_test[0])

#print image
image_test=numpy.asfarray(all_values_test[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_test,cmap='Greys',interpolation='None')

#results
n.query((numpy.asfarray(all_values_test[1:])/255.0*0.99)+0.01)
