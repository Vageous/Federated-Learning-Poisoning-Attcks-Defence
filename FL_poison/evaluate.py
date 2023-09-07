import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class evaluate(object):
    def __init__(self,args,test_set,train_set) -> None:
        self.device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu")
        self.test_loader=DataLoader(test_set,batch_size=args.batchsize,shuffle=True)
        self.train_loader=DataLoader(train_set,batch_size=args.batchsize,shuffle=True)


    def model_test_testset(self,global_model):
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
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc,total_l
    
    def model_test_trainset(self,global_model):
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
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc,total_l
    

    # label fliping: 1->2; 2->3
    # 有目标投毒攻击，攻击成功的概率
    def ASR_target(self,global_model):
        global_model.eval()
        poison_label=0
        attack_success=0

        for batchidx,batch in enumerate(self.test_loader):
            image=batch[0].to(self.device)
            label=batch[1].to(self.device)
            output=global_model(image)
            label1_index=(label==1).nonzero().reshape(-1)
            label2_index=(label==2).nonzero().reshape(-1)
            poison_label += (len(label1_index)+len(label2_index))
            pred=output.data.max(1)[1]
            result=pred.eq(label.data.view_as(pred))
            for i in range(len(label1_index)):
                if result[label1_index[i]]==False:
                    attack_success += 1
            for i in range(len(label2_index)):
                if result[label2_index[i]]==False:
                    attack_success +=1

        asr= 100.0 * (float(attack_success) / float(poison_label))

        return asr
        

    # untarget poisoning attacks
    def ASR_untarget(self,global_model):
        global_model.eval()
        attack_success=0
        datasize=0

        for batchidx,batch in enumerate(self.test_loader):
            image=batch[0].to(self.device)
            label=batch[1].to(self.device)
            output=global_model(image)
            datasize += image.size()[0]
            pred=output.data.max(1)[1]
            result=pred.eq(label.data.view_as(pred))
            # correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
            attack_success += result.cpu().sum().item()

        asr= 100.0 * (1-(float(attack_success) / float(datasize)))

        return asr



