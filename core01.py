import numpy as np
import random
import matplotlib.pyplot as plt

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
        
#MultiRuns(5)
val = HillClimber()
print(val['Genes'])
plt.imshow(val['Genes'], cmap=plt.cm.gray, aspect='auto', interpolation='nearest')
plt.show()