import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

def plotCM(matrix, labels_names, title, savname):

    matrix = matrix / matrix.sum(axis=1)[:, np.newaxis] 
    thresh = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, "{:0.2f}".format(matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
    plt.imshow(matrix, interpolation='nearest') 
    plt.title(title) 
    plt.colorbar() 
    
    num_class = np.array(range(len(labels_names)))
    

    plt.xticks(num_class, labels_names, rotation=90)  
    plt.yticks(num_class, labels_names) 

    plt.ylabel('Target')
    plt.xlabel('Prediction')

    plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    
    plt.savefig(savname)
    plt.show()
