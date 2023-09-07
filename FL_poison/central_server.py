import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import init
from copy import deepcopy
import utils

class Server(object):
    def __init__(self,args,model,test_set,select_user_num,poison_user_num) -> None:
        self.clients=[]
        self.n=select_user_num
        self.f=poison_user_num
        self.args=args
        self.Net=None
        self.model=model
        self.emptyStates = None
        self.test_set=test_set
        self.layer=args.layer
        self.init_stateChange()
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")
        self.test_loader=DataLoader(self.test_set,batch_size=args.batchsize,shuffle=True)
        # self.server=IBBE.server(pp)

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def get_deltas(self,local_i):
        self.clients.append(local_i)
    

    def model_aggregate(self,local_model,weight_model):
        for name,params in local_model.items():
            weight_model[name].add_(params)
        return weight_model

    # def cipher_model_aggregate(self,local_model,weight_model):
        
    #     for name,params in local_model.items():
    #         if name==self.layer:
    #             weight_model[name]=self.server.aggregate(weight_model[name],local_model[name])
    #         else:
    #             weight_model[name].add_(params)
                
    # def cipher_model_recover(self,weight_model,h_prime):
    #     for name,params in weight_model.items():
    #         if name==self.layer:
    #             weight_model[name]=self.server.aggre_recover(h_prime,weight_model[name])
        
    def model_average(self,weight_model,num_user):
        for name,params in weight_model.items():
            weight_model[name]=weight_model[name] / num_user
        return weight_model

    def Aggregation(self):
        if self.args.AR=='fedavg':
            return self.FedAvg()
        elif self.args.AR=='krum':
            return self.Krum(self.f)
        elif self.args.AR=='multiKrum':
            return self.MultiKrum(self.f)
        elif self.args.AR=='median':
            return self.FedMedian()
        elif self.args.AR=='bulyan':
            return self.Bulyan(self.f)
        else:
            raise ValueError("Not a valid aggregation rule or aggregation rule not implemented")
    
    def FedAvg(self):
        out = self.FedFuncWholeNet(lambda arr: torch.mean(arr, dim=-1, keepdim=True))
        return out

    def FedMedian(self):
        out = self.FedFuncWholeNet(lambda arr: torch.median(arr, dim=-1, keepdim=True)[0])
        return out

    def Krum(self,poison_user_num):
            from rules.multiKrum import Net
            self.Net=Net(poison_user_num)
            out = self.FedFuncWholeNet(lambda arr: Net(poison_user_num,'krum').cpu()(arr.cpu()))
            return out

    def MultiKrum(self,poison_user_num):
            from rules.multiKrum import Net
            self.Net=Net(poison_user_num)
            out = self.FedFuncWholeNet(lambda arr: Net(poison_user_num,'mkrum').cpu()(arr.cpu()))
            return out

    def Bulyan(self,poison_user_num):
        from rules.Bulyan import Net
        self.Net=Net(poison_user_num)
        out=self.FedFuncWholeNet(lambda arr: Net(poison_user_num).cpu()(arr.cpu()))
        return out

    def FedFuncWholeNet(self, func):
        '''
        The aggregation rule views the update vectors as stacked vectors (1 by d by n).
        '''
        Delta = deepcopy(self.emptyStates) # 与模型相同的，但值为0
        deltas = self.clients # 获取所有客户端的本地模型状态字典
        vecs = [utils.net2vec(delta) for delta in deltas]
        vecs = [vec for vec in vecs if torch.isfinite(vec).all().item()]
        result = func(torch.stack(vecs, 1).unsqueeze(0))  # input as 1 by d by n
        result = result.view(-1)
        # 由一维张量转换为模型状态字典
        utils.vec2net(result, Delta)
        return Delta

    def model_test(self,global_model):
        global_model.eval()
        total_loss=0.0
        correct=0
        dataset_size=0

        for batchidx,batch in enumerate(self.test_loader):
            image=batch[0].to(self.device)
            label=batch[1].to(self.device)
            dataset_size += image.size()[0]

            output=global_model(image)
            total_loss += torch.nn.functional.cross_entropy(output, label,reduction='sum').item()
            #output.data.max(1)[1]：前一个1表示按行寻找最大值，后一个1表示返回一个数组：[最大值，最大值索引]
            # shape：tensor([batchsize])
            pred = output.data.max(1)[1]

            # pred.eq()表示比较两个相同维数张量对应位置元素是否相同，相同返回1，不同返回0
            # pred.eq().sum(),求和
            # pred.eq().sum().item(),取最后的求和值大小
            # shape: tensor([batchszie])
            result=pred.eq(label.data.view_as(pred))

            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc,total_l
