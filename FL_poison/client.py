import torch
import model
import sample
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from option import parser_args
import transform

class LocalUpdate(object):
    def __init__(self,args,dataset,select_idxs,poison) -> None:
        self.dataset=dataset
        self.num_user=args.num_user
        self.batchsize=args.batchsize
        self.lr=args.lr
        self.local_round=args.local_round
        self.momentum=args.momentum
        self.layer=args.layer
        self.flag=args.flag
        self.scale=args.scale
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")

        if args.model=="letnet":
            self.local_model=model.cnn().to(self.device)
        elif args.model=="alexnet":
            self.local_model=model.cnn1().to(self.device)
        elif args.model=="lstm":
            self.local_model=model.LSTM(args.vocab_size,args.embedding_dim, args.hidden_size, args.num_layers, args.output_size).to(self.device)


        if poison==0:
            self.train_set=DataLoader(sample.datasetsplit(self.dataset,select_idxs),batch_size=self.batchsize,shuffle=True)
        else:
            if args.poison_type=='target':
                self.train_set=DataLoader(sample.datasetsplit_poison_target(self.dataset,select_idxs),batch_size=self.batchsize,shuffle=True)
            else:
                self.train_set=DataLoader(sample.datasetsplit_poison_untarget(self.dataset,select_idxs),batch_size=self.batchsize,shuffle=True)


    def train(self,global_model):
        local_model={}
        for name,param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer=torch.optim.SGD(self.local_model.parameters(),lr=self.lr,momentum=self.momentum)

        self.local_model.train()
        for i in range(self.local_round):
            for batch_idx, batch in enumerate(self.train_set):
                image=batch[0].to(self.device)
                label=batch[1].to(self.device)

                optimizer.zero_grad()
                output=self.local_model(image)

                loss=F.cross_entropy(output,label)
                loss.backward()

                optimizer.step()

        if self.flag==0:
            return self.local_model.state_dict()



# if __name__=='__main__':
#     args=parser_args()
#     kgc=IBBE.KGC(args.bits,args.num_user,args.ID)
#     pp=IBBE.params(kgc.n,kgc.N,kgc.g1,kgc.g2,kgc.g3,kgc.hash1,kgc.hash2,kgc.num)
#     new_local=[]
#     User=IBBE.participant(pp)
#     mnist_train,mnist_test=dataset.get_data(args.dataset)
#     idx=sample.mnist_iid(mnist_train,args.num_user)
#     test=LocalUpdate(args,mnist_train,idx[0],kgc.usk2,pp)
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     global_model=model.cnn().to(device)
#     local=test.train(global_model)
#     for name,params in local.items():
#         if name=="conv2.weight":
#             new_local,index=transform.encode(local[name],args.scale)
#             cipher=User.private_enc(new_local,kgc.usk2[0])
#             print(cipher[0])