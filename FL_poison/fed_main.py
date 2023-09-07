import model
import torch
import client
import dataset
import central_server
import sample
import numpy as np
import time
import evaluate
from option import parser_args


def fed_main():
    args=parser_args()
    select_users_num=int(args.rate_1*args.num_user)
    poison_user_num=int(args.rate_2*select_users_num)
    train_set,test_set=dataset.get_data(args.dataset)

    if args.model=="letnet":
        global_model=model.cnn().to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    elif args.model=="alexnet":
        global_model=model.cnn1().to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    elif args.model=="lstm":
        global_model=model.LSTM(args.vocab_size,args.embedding_dim, args.hidden_size, args.num_layers, args.output_size).to(device=torch.device("cuda" if torch.cuda.is_available()==args.device else "cpu"))
    
    if args.iid==0:
        idxs=sample.mnist_iid(train_set,args.num_user)
    else:
        idxs=sample.mnist_noiid(train_set,args.num_user)

    select_idxs=np.random.choice(range(args.num_user),select_users_num,replace=False)
    # poison client list
    honest_user=[]
    poison_user=np.random.choice(range(select_users_num),poison_user_num,replace=False)
    for i in range(select_users_num):
        if i not in poison_user:
            honest_user.append(i)
    server=central_server.Server(args,global_model,test_set,select_users_num,poison_user_num)
    eval=evaluate.evaluate(args,test_set,train_set)

    for iter in range(args.epoch):
        time1=time.time()
 
        if args.poison_flag==0:
            for i in range(len(select_idxs)):
                participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],0)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.get_deltas(local_model)
                    # server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                # server.model_average(weight_model,select_users_num)
                weight_model=server.Aggregation()
                global_model.load_state_dict(weight_model)
            
            acc,total_loss=eval.model_test_testset(global_model)
            print('-'*4+"The experiment result of Epoch %d" % (iter)+'-'*4+'\n')
            print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))

        else:
            for i in range(len(select_idxs)):
                if i in poison_user:
                    participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],1)
                else:
                    participant=client.LocalUpdate(args,train_set,idxs[select_idxs[i]],0)
                if args.flag==0:
                    local_model=participant.train(global_model)
                    server.get_deltas(local_model)
                    # server.model_aggregate(local_model,weight_model)
                else:
                    cipher_local,index=participant.train(global_model)
                    server.cipher_model_aggregate(cipher_local,weight_model)

            if args.flag==0:
                weight_model=server.Aggregation()
                # server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)
            else:
                participant.decrypt(weight_model,0,index)
                server.model_average(weight_model,select_users_num)
                global_model.load_state_dict(weight_model)

            if args.poison_type=='target':
                acc,total_loss=eval.model_test_testset(global_model)
                asr=eval.ASR_target(global_model)

                print('-'*4+"The experiment result of Epoch %d" % (iter)+'-'*4+'\n')
                print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
                print("Target Attack success rate: %f\n" %(asr))
            else:
                acc,total_loss=eval.model_test_testset(global_model)
                asr=eval.ASR_untarget(global_model)
                print('-'*4+"The experiment result of Epoch %d" % (iter)+'-'*4+'\n')
                print("Epoch %d, acc: %f, loss: %f\n" % (iter, acc, total_loss))
                print("Untarget Attack success rate: %f\n" %(asr))



if __name__=='__main__':
    fed_main()
    print("Finish")