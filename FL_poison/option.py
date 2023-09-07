import argparse
import torch

def parser_args():
    parser=argparse.ArgumentParser()

    # some parameters for the federated learning
    parser.add_argument("--epoch",type=int,default=50,help='the training round of the federated learning')
    parser.add_argument("--batchsize",type=int,default=32,help='the bact size of the federated learning')
    parser.add_argument("--lr",type=float,default=0.1,help='the learning rate of the federated learning')
    parser.add_argument("--momentum",type=float,default=0.01,help='the momentum of the SGD algorithm')
    parser.add_argument("--dataset",type=str,default="cancer",help='the dataset used in the federated learning')
    parser.add_argument("--num_user",type=int,default=20,help='the total number of clients in the federated learning')
    parser.add_argument("--local_round",type=int,default=1,help='the local model training rounds')
    parser.add_argument("--rate_1",type=int,default=1,help='the rate of the selected client in num_user')
    parser.add_argument("--device",type=int,default=1,help='whether use the cuda for model training')
    parser.add_argument("--iid",type=int,default=0,help='the type of the federated learning setting')
    parser.add_argument("--model",type=str,default="lstm",help="the type of training model")
    parser.add_argument("--vocab_size",type=int,default=1000,help="The vocab_size of LSTM model")
    parser.add_argument("--embedding_dim",type=int,default=10,help="The embedding_dim of LSTM model")
    parser.add_argument("--hidden_size",type=int,default=64,help="The hidden_size of LSTM model")
    parser.add_argument("--num_layers",type=int,default=2,help="The num_layers of LSTM model")
    parser.add_argument("--output_size",type=int,default=1,help="The output_size of LSTM model")

    # some parameters for the poisoning attacks in federated learning
    parser.add_argument("--rate_2",type=float,default=0.2,help='the rate of the malicious clients in selected clients')
    parser.add_argument('--poison_flag',type=int,default=0,help='whether there exist the poisoning attacks')
    parser.add_argument('--poison_type',type=str,default='target',help='whether the poisoning attack is target or untarget')

    # some parameters for the aggregation rules
    '''
    Aggregation Rules:
    n: the total number of clients; f: the number of malicious clients
        1. FedAvg: no requirement
        2. Median: no requirement
        3. Krum & MultiKrum: 2*f+2<n
        4. Bulyan: n>=4*f+3
        5. Trimmed-mean: no requirement

    '''
    parser.add_argument("--AR",type=str,default='fedavg',help='the aggregation methods for the federated learning')

    # some parameters for the ecnryption algorithm
    parser.add_argument("--bits",type=int,default=20,help='the length of the security parameter')
    parser.add_argument("--ID",default=["user1","user2"],help='the identity of the user')
    parser.add_argument("--scale",default=10000,help='transfer a float number to int')
    parser.add_argument("--flag",default=0,help='whether use the encryption algorithm')
    parser.add_argument("--layer",default="conv2.bias",help='the layer of the model that need to be encrypted')



    args=parser.parse_args()
    return args
    
