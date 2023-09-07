import torch
import random
import torch.nn as nn

def Trimmed_mean(input,f):
    '''
    input: batchsize*vector dimension*n
    output: batchsize*vector dimension*n
    '''
    n=input.shape[-1]
    beta=int(random.uniform(f,n//2))

    # for i in range(len())
