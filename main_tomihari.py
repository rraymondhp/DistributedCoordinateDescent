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
# service = QiskitRuntimeService(instance="ibm-q-utokyo/internal/qc-training22")

### Configuration ###
Q = 6  # number of qubits
L = 3  # number of fraxis layer
W = 3  # number of quantum node
N = 100  # number of training data size
M = 100  # number of testing data size
U = 10  # number of update
backend = service.backend("simulator_mps")
params = np.zeros((L, Q, 3)) + np.array([0, 0, 1])
# params = np.random.rand(L, Q, 3)
params /= np.linalg.norm(params, axis=2, keepdims=True)
# update = "inorder"
# update = "random_all"
# update = "random"
update = "random_per_layer"
# train_rate = 1.0
train_rate = 0.2
# update_rate=1.0
update_rate = 0.5
# preprocessing = "Titanic"
preprocessing = None
# label = "Survived"
label = None
# feat = ["Age", "Embarked"]
feat = None
# CSVpath = "data/titanic/train.csv"
CSVpath = None

if __name__ == "__main__":
    st = time.time()
    print("Start training")
    print("Q = ", Q, ", L = ", L, ", W = ", W, ", N = ", N, ", M = ", M, ", U = ", U)
    print(
        "update = ",
        update,
        ", train_rate = ",
        train_rate,
        ", update_rate = ",
        update_rate,
    )
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
        train_rate=train_rate,
        update_rate=update_rate,
        isVal=True,
        isEval=True,
        label=label,
        feat=feat,
    )
    print("All training finished. Implementation time : ", time.time() - st)
