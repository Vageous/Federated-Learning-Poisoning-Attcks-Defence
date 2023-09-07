import numpy as np

idxs=np.array([1,2,3])
labels=np.array([3,2,1])

# print(np.vstack(x,y))
idxs_labels=np.vstack((idxs,labels))
print(idxs_labels)
    # 根据标签对图像索引进行排序，通过 argsort() 方法对 idxs_labels[1,:]（标签）进行排序，并将排序后的索引应用到 idxs_labels
idxs_labels=idxs_labels[:,idxs_labels[1,:].argsort()]
print(idxs_labels)
    # 更新为排序后的图像索引
idxs=idxs_labels[0,:]
print(idxs)