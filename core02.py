# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def MatrixCreate(rows,columns):
    """ Create rows and columns matrix """
    return np.zeros((rows,columns));
        
def MatrixRandomize(v):
    """ Randonmize the elements of the matrix"""
    return np.random.rand(v.shape[0],v.shape[1]);
    
def MatrixPerturb(p, prob):
    """ Makes deep copy of parent p and assigns random values to elements of p based on probability prob"""
    child = np.copy(p)
    for x in range(child.shape[0]):
        for y in range(child.shape[1]):
            if(random.random() < prob):
                child[x,y] = np.random.rand(1)
    return child;
    
def CreateNeuralNetwork():
    neuronValues = MatrixCreate(50,10)
    neuronValues[0,:] = np.random.rand(1,10)
    neuronPositions = findNeuronPositions(10)
    synapses = createSynapses(10)
    
    print(neuronValues[0,:])
    # Figure 1
    plt.plot(neuronPositions[0,:],neuronPositions[1,:],'ko',markerfacecolor=[1,1,1], markersize=18)
    
    for point in range(neuronPositions.shape[1]):
        for point2 in range(neuronPositions.shape[1]):
            if(synapses[point,point2] >= 0 ):
                clr = [0,0,0]
            else:
                clr =[0.8,0.8,0.8]
                
            w = int(10*abs(synapses[point,point2]))+1
              #Fig   2 and 3           
            plt.plot([neuronPositions[0][point], neuronPositions[0][point2]],[neuronPositions[1][point],neuronPositions[1][point2]],color=clr,linewidth=w)
            
    plt.show()
    
    plt.clf()
    # set first row to have random values in a range and then call update
    for i in range(1,neuronValues.shape[0]):
        neuronValues = Update(neuronValues,synapses,i)
    plt.imshow(neuronValues, cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
    plt.show()
    
def Update (neuronValues,synapses,i):
    if (i==0):
        return;
    
    for j in range(neuronValues.shape[1]):
        temp = 0
        for k in range(neuronValues.shape[1]):
            temp = temp + (neuronValues[i-1][j]*synapses[j][k])
            temp = np.clip(temp,0,1)
            neuronValues[i][j] = temp;
            
    return neuronValues;
            
    
def createSynapses(size):
    return (np.random.uniform(low=-1.0, high=1.0, size=(size,size)))
    
def findNeuronPositions(num_neurons):
    angle = 0.0
    angleUpdate = 2 * 180 / num_neurons
    neuronPositions = MatrixCreate(2,num_neurons)
    for i in range(num_neurons):
        neuronPositions[0,i] = math.sin(angle)
        neuronPositions[1,i] = math.cos(angle)
        angle = angle + angleUpdate
    return neuronPositions;
    
def Fitness(v):
    return np.mean(v);
    

def HillClimber():
    Genes = MatrixCreate(50,5000)
    fits = MatrixCreate(1,5000)
    parent = MatrixCreate(1, 50) 
    parent = MatrixRandomize(parent) 
    parentFitness = Fitness(parent) 
    for currentGeneration in range(5000):
        Genes[:,currentGeneration] = parent
        print (currentGeneration, parentFitness)
        fits[0,currentGeneration] = parentFitness
        child = MatrixPerturb(parent, 0.05) 
        childFitness = Fitness(child) 
        if childFitness > parentFitness:
                    parent = child 
                    parentFitness = childFitness
               
    return {'fits':fits,'Genes': Genes}
    
def PlotVectorAsLine(fits):
    plt.plot(fits[0,:])
    plt.ylabel('fitness')
    plt.xlabel('Generations')
    plt.show() 
                        
def MultiRuns(num_runs):
    for x in range(num_runs):
        val = HillClimber()
        PlotVectorAsLine(val['fits'])
        
CreateNeuralNetwork()