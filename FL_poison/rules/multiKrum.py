import torch
import torch.nn as nn

'''
Krum aggregation
- find the point closest to its neignborhood

Reference:
Blanchard, Peva, Rachid Guerraoui, and Julien Stainer. "Machine learning with adversaries: Byzantine tolerant gradient descent." Advances in Neural Information Processing Systems. 2017.
`https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf`

'''


def getKrum(input,f):
    '''
    compute krum or multi-krum of input. O(dn^2)
    
    input : batchsize* vector dimension * n
    
    return 
        krum : batchsize* vector dimension * 1
        mkrum : batchsize* vector dimension * 1
    '''
    # n=20
    # input：torhc.size([1,82826,20])
    n = input.shape[-1] # get the final dimension of the input tensor.

    # The requirement for the Krum and MultiKrum methods
    # f=10
    # worse case 50% malicious points
    # k=8
    if 2*f+2<n:

        k = n - f - 2
        # collection distance, distance from points to points
        # x: torch.size([1,20,82826])
        x = input.permute(0, 2, 1) # 维度换位，x所对应的torch.size([batchsize,n,vector dimension])
        # 计算 张量x的模数
        # cdist: torch.size([1,20,20])
        cdist = torch.cdist(x, x, p=2)
        # find the k+1 nbh of each point
        # nbhDist: [1,20,9]
        # nbh: [1,20,9]
        nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
        # the point closest to its nbh
        # i_star: [1]
        # nbhDist.sum(2): [1,20]
        i_star = torch.argmin(nbhDist.sum(2))
        # krum [1,82826,1]
        krum = input[:, :, [i_star]]
        # Multi-Krum [1,82826,1]
        mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
        return krum, mkrum
    
    else:
        ValueError("Not a valid set of the number of poison clients and honest clients")


class Net(nn.Module):
    def __init__(self, f, mode='mkrum'):
        super(Net, self).__init__()
        assert (mode in ['krum', 'mkrum'])
        self.mode = mode
        self.f=f

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        
        return 
            out : batchsize* vector dimension * 1
        '''
        krum, mkrum = getKrum(input,self.f)

        out = krum if self.mode == 'krum' else mkrum

        return out
