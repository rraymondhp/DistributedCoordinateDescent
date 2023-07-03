from qiskit_ibm_runtime import QiskitRuntimeService
from training_tomihari import parallel_train
import numpy as np
import time
import pandas as pd

np.random.seed(0)

# QiskitRuntimeService.save_account(channel="ibm_quantum", overwrite=True, token="7c777f4eda9a82089a232b9fd5248756c433f4a52ebf5f8fd48f15eb7bc06a3ed98b5bb05af61464f3e9ca455fb07a2df3008ae0fec33f640d8e90d30ef98456")
# service = QiskitRuntimeService(
#     channel='ibm_quantum',
#     instance='ibm-q-utokyo/internal/hirashi-jst',
# )
service = QiskitRuntimeService()
service = QiskitRuntimeService(instance="ibm-q-utokyo/internal/qc-training22")

### Configuration ###
Q = 2  # number of qubits
L = 2  # number of fraxis layer
W = 2  # number of quantum node
N = 20  # number of training data size
M = 10  # number of testing data size
U = 1  # number of update
backend = service.backend("simulator_mps")
# params = np.zeros((L, Q, 3)) + np.array([0, 0, 1])
params = np.random.rand(L, Q, 3)
params /= np.linalg.norm(params, axis=2, keepdims=True)
update = "inorder"
# update = "random"
trainrate = 1.0
# trainrate = 0.5
preprocessing = "Titanic"
# preprocessing=None
label = "Survived"
# label=None
feat = ["Age", "Embarked"]
# feat=None
CSVpath = "data/titanic/train.csv"
# CSVpath = None

if __name__ == "__main__":
    st = time.time()
    print("Start training")
    print("Q = ", Q, ", L = ", L, ", W = ", W, ", N = ", N, ", M = ", M, ", U = ", U)
    print("update = ", update, ", trainrate = ", trainrate)
    parallel_train(
        Q,
        L,
        W,
        N,
        M,
        U,
        service,
        backend,
        params,
        isFashion=False,
        CSVpath=CSVpath,
        preprocessing=preprocessing,
        update=update,
        trainrate=trainrate,
        isVal=True,
        isEval=True,
        label=label,
        feat=feat,
    )
    print("Implementation time : ", time.time() - st)
