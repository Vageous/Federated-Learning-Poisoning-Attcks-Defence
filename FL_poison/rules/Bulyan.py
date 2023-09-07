import torch
import random
import torch.nn as nn

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
        # mkrum = input[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
        return krum
    
    else:
        ValueError("Not a valid set of the number of poison clients and honest clients")

def select_krums(input,beta,theta):
    # input: [1,82826,theta]
    select_num=int(random.uniform(beta,beta+3))
    values,_=torch.median(input,dim=2)
    new_values=values
    for i in range(theta-1):
        new_values=torch.cat([new_values,values],dim=0)

    dist=torch.abs(torch.sub(torch.squeeze(input),torch.transpose(new_values,dim0=0,dim1=1)))
    new_dist,index_=torch.topk(dist,k=select_num,largest=False)

    new_y=torch.squeeze(input)
    final=torch.index_select(new_y[0],dim=0,index=index_[0])
    for i in range(1,len(new_y)):
        final=torch.cat([final,torch.index_select(new_y[i],dim=0,index=index_[i])])
    final=final.reshape(-1,index_.shape[-1])
    new_final=torch.mean(final,dim=1).reshape(1,input.shape[1],1)
    return new_final


def Bulyan(input,f):
    n=input.shape[-1]
    theta=2*f+3
    beta=theta-2*f
    krums=getKrum(input,f)
    if n>=4*f+3:
        for i in range(theta-1):
            krums=torch.cat([krums,getKrum(input,f)],dim=2)
        return select_krums(krums,beta,theta)
    else:
        ValueError("Not a valid set of the number of poison clients and honest clients")


class  Net(nn.Module):
    def __init__(self,f) -> None:
        super(Net,self).__init__()
        self.f=f


    def forward(self,input):

        bulyan=Bulyan(input,self.f)

        return bulyan