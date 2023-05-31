import numpy as np
from model import FraxClassify
import time
from qiskit_ibm_runtime import Session
from sklearn import preprocessing
from sklearn.preprocessing import normalize


def data_loader(atomNo=48, dataPref="data/IBM/train.data_", dataSuf=".100.balanced", testPref="data/IBM/Data_", testSuf=".log", isScaled=False):
    trainFile = dataPref+str(atomNo)+dataSuf
    testFile = testPref+str(atomNo)+testSuf
    
    train = []
    trainLabel = []
    test = []
    testLabel = []
    with open(trainFile) as f:
        for line in f:
            array_1d = np.array([1.0, ] + [ float(x) for x in line.rstrip().split()[1:5] ])
            vecs = normalize(array_1d[:,np.newaxis], axis=0).ravel()
            
            vecs = np.kron(vecs, vecs)
            
            #vecs = np.linalg.norm([ float(x) for x in line.rstrip().split()[1:5] ]) 
            label = 2.0*float(line.rstrip().split()[-1]) - 1
            train.extend([vecs])
            trainLabel.extend([label])
            #print("TR", len(train))
            #print("TR", len(trainLabel))
            #print(vecs, label)
    
    with open(testFile) as f:
        for line in f:
            array_1d = np.array([1.0, ] + [ float(x) for x in line.rstrip().split()[1:5] ])
            vecs = normalize(array_1d[:,np.newaxis], axis=0).ravel()
            
            vecs = np.kron(vecs, vecs)
            
            #vecs = np.linalg.norm([ float(x) for x in line.rstrip().split()[1:5] ]) 
            label = 2.0*float(line.rstrip().split()[-1]) - 1
            test.extend([vecs])
            testLabel.extend([label])
            #print("TE", len(test))
            #print("TE", len(testLabel))
            #print(vecs, label)
    train = np.array(train)
    test = np.array(test)
    testLabel = np.array(testLabel)
    trainLabel = np.array(trainLabel)
    print("AFTER ARRAYING")
    print(train.shape, test.shape, testLabel.shape, trainLabel.shape)
    #print(np.kron(train,train).shape, np.kron(test,test).shape)
    if not isScaled:
        print("End loading data")
        #return np.array(testLabel), np.array(trainLabel), np.array(np.kron(test,test)), np.array(np.kron(train,train))
        return np.array(testLabel), np.array(trainLabel), np.array(test), np.array(train)
        #return testLabel, trainLabel, test, train
    else:
        scaler = preprocessing.StandardScaler().fit(np.array(train))
        #return np.array(testLabel), np.array(trainLabel), scaler.transform(np.array(np.kron(test,test))), scaler.transform(np.array(np.kron(train, train)))
        print("End loading data")
        return np.array(testLabel), np.array(trainLabel), scaler.transform(np.array(test)), scaler.transform(np.array(train))

    
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

def parallel_train(n_qubits, layer_size, world_size, num_train, num_test, update_iter, service, backend, params, isScaled=True, atomNo=48):
    print("Start loading data")
    test_label, train_label, test_feat, train_feat = data_loader(atomNo=atomNo, isScaled=isScaled)
    model = FraxClassify(n_qubits, layer_size, world_size, num_train, num_test, backend, params)
    isEval = False
    with Session(service = service, backend = backend):
        for i in range(update_iter):
            st = time.time()
            if i == update_iter - 1:
                isEval = True
            model.fit_and_eval(train_feat,train_label,test_feat,test_label, isEval=isEval)
            print(time.time()-st)
            print('_______________NEW________________',i)