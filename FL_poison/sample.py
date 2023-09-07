import numpy as np
import random
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset
import dataset
import option


def mnist_iid(dataset,num_user):
    num_items=int(len(dataset)/num_user)
    dict_users,all_idx={},[i for i in range(len(dataset))]
    for i in range(num_user):
    #numpy.random.choice(a, size=None, replace=True, p=None)
        dict_users[i]=set(np.random.choice(all_idx,num_items,replace=False))
        all_idx=list(set(all_idx)-dict_users[i])
    return dict_users

def mnist_noiid(dataset,num_user):
    # num_shards:分片数量
    # num_imgs:每个分片中图片的数量
    num_shards,num_imgs=200,300
    # idx_shard:所有分片对应的索引
    idx_shard=[i for i in range(num_shards)]
    # 是一个字典，键是用户的索引，值是一个空的 NumPy 数组，用于存储每个用户拥有的图像索引。字典的长度由 num_user 决定。
    dict_users={i:np.array([],dtype='int64')for i in range(num_user)}
    # 包含所有图像的索引
    idxs=np.arange(num_shards*num_imgs)
    # 数据集中训练集所有标签，numpy数组
    labels=dataset.targets.numpy()
    #  是一个包含图像索引和对应标签的数组，通过垂直堆叠 idxs 和 labels 得到。
    idxs_labels=np.vstack((idxs,labels))
    # 根据标签对图像索引进行排序，通过 argsort() 方法对 idxs_labels[1,:]（标签）进行排序，并将排序后的索引应用到 idxs_labels
    idxs_labels=idxs_labels[:,idxs_labels[1,:].argsort()]
    # 更新为排序后的图像索引
    idxs=idxs_labels[0,:]

    for i in range(num_user):
        # 对于每个用户选两个分片
        rand_set = set(np.random.choice(idx_shard, int(num_shards/num_user), replace=False))
        # 将被选的两个分片索引删除
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # 对于每个被选中的分片索引 rand，
            # 将对应的图像索引范围 rand*num_imgs:(rand+1)*num_imgs 添加到用户 i 的图像索引列表 dict_users[i]
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


class datasetsplit(Dataset):

    def __init__(self,dataset,idx) -> None:
        self.dataset=dataset
        self.idx=list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self,item):
        image,label=self.dataset[self.idx[item]]
        return image,label

class datasetsplit_poison_target(Dataset):

    def __init__(self,dataset,idx) -> None:
        self.dataset=dataset
        self.idx=list(idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self,item):
        image,label=self.dataset[self.idx[item]]
        if label==1:
            label=2
        elif label==2:
            label=3
        return image,label
    
class datasetsplit_poison_untarget(Dataset):
    def __init__(self,dataset,idx) -> None:
        self.dataset=dataset
        self.idx=list(idx)

    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, item):
        image,label=self.dataset[self.idx[item]]
        label=int(random.uniform(1,10))
        return image,label
    
# noiid dataset
# if __name__=='__main__':
#     args=option.parser_args()
#     mnist_train,mnist_test=dataset.get_data(args.dataset)
#     # print(mnist_train)
#     idxs=mnist_noiid(mnist_train,args.num_user)
#     train_set=DataLoader(datasetsplit(mnist_train,idxs[0]),batch_size=args.batchsize,shuffle=False)
#     for batchidx, (image,label) in enumerate(train_set):
#         print(label)
#         break
if __name__=='__main__':
    args=option.parser_args()
    train_set,test_set=dataset.get_data("cancer")
    idxs=mnist_iid(train_set,args.num_user)
    train=DataLoader(datasetsplit(train_set,idxs[0]),batch_size=args.batchsize,shuffle=False)
    for idx,batch in enumerate(train):
        print(batch[0])
        print(batch[1])
        break
# test the poison result
# if __name__=='__main__':
#     args=option.parser_args()
#     mnist_train,mnist_test=dataset.get_data(args.dataset)
#     # print(mnist_train)
#     idxs=mnist_iid(mnist_train,args.num_user)
#     # 第0个客户端数据集
#     train_set=DataLoader(datasetsplit(mnist_train,idxs[0]),batch_size=args.batchsize,shuffle=False)
#     train_set1=DataLoader(datasetsplit_poison_target(mnist_train,idxs[0]),batch_size=args.batchsize,shuffle=False)
#     train_set2=DataLoader(datasetsplit_poison_untarget(mnist_train,idxs[0]),batch_size=args.batchsize,shuffle=False)
    # 94次
#     for batchidx, (image,label) in enumerate(train_set):
#         print(label)
#         break
#     for batchidx,(image,label) in enumerate(train_set1):
#         print(label)
#         break
#     for batchidx, (image,label) in enumerate(train_set2):
#         print(label)
#         break

    

