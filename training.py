import numpy as np
from model import FraxClassify
import time
from qiskit_ibm_runtime import Session


def data_loader(num_train, num_test, isFashion=False):
    try:
        if isFashion:
            print("Using Fashion MNIST")
            test_label = np.load('data/fmnist_test_Label.npy')[0:num_test]
            train_label = np.load('data/fmnist_train_Label.npy')[0:num_train]
            test_feat = np.load('data/fmnist_test_feat.npy')[0:num_test]
            train_feat = np.load('data/fmnist_train_feat.npy')[0:num_train]
        else:
            print("Using MNIST")
            test_label = np.load('data/mnist_test_Label.npy')[0:num_test]
            train_label = np.load('data/mnist_train_Label.npy')[0:num_train]
            test_feat = np.load('data/mnist_test_feat.npy')[0:num_test]
            train_feat = np.load('data/mnist_train_feat.npy')[0:num_train]
            
        return test_label, train_label, test_feat, train_feat
    except Exception as e:
        print(e)
    
def cut_data(train_label, train_feat, test_label, test_feat, rank, world_size):
    data_len_min = len(train_feat) // world_size
    offset = len(train_feat) % world_size
    if rank < offset:
        start1 = rank*(data_len_min+1)
        end1 = start1+data_len_min+1
    else:
        start1 = offset*(data_len_min+1)+(rank-offset)*data_len_min
        end1 = start1+data_len_min
    data_len_min = len(test_feat) // world_size
    offset = len(test_feat) % world_size
    if rank < offset:
        start2 = rank*(data_len_min+1)
        end2 = start2+data_len_min+1
    else:
        start2 = offset*(data_len_min+1)+(rank-offset)*data_len_min
        end2 = start2+data_len_min
    
    return train_label[start1:end1], train_feat[start1:end1], test_label[start2:end2], test_feat[start2:end2]

def parallel_train(n_qubits, layer_size, world_size, num_train, num_test, update_iter, service, backend, params, isFashion=False):
    test_label, train_label, test_feat, train_feat = data_loader(num_train, num_test, isFashion)
    model = FraxClassify(n_qubits, layer_size, world_size, num_train, num_test, backend, params)
    with Session(service = service, backend = backend):
        for i in range(update_iter):
            st = time.time()
            model.fit_and_eval(train_feat,train_label,test_feat,test_label)
            print(time.time()-st)
            print('_______________NEW________________',i)