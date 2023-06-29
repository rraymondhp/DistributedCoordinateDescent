from qiskit_ibm_runtime import QiskitRuntimeService
from training_tomihari import parallel_train
import numpy as np
import time
import pandas as pd

np.random.seed(0)

# QiskitRuntimeService.save_account(channel="ibm_quantum", token="877beba4ad71563f645d0f4c3ca6fba24f0cb1568653e5f5ce930b9e7c63fabc830d834a2dbba6eaa42018fb77da880e6834f888e0383b549a0851afe789958d")
# service = QiskitRuntimeService(
#     channel='ibm_quantum',
#     instance='ibm-q-utokyo/internal/hirashi-jst',
# )
service = QiskitRuntimeService()

Q = 2  # number of qubits
L = 2  # number of fraxis layer
W = 10  # number of quantum node
N = 30  # number of training data size
M = 30  # number of testing data size
U = 1  # number of update
backend = service.backend("simulator_mps")
# params = np.zeros((L, Q, 3)) + np.array([0, 0, 1])
params = np.random.rand(L, Q, 3)
params /= np.linalg.norm(params, axis=2, keepdims=True)
update = "inorder"
trainrate = 1.0
# trainrate = 0.5
update = "random"
if __name__ == "__main__":
    st = time.time()
    train_path = "data/titanic/train.csv"
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
        CSV=train_path,
        preprocessing="Titanic",
        update=update,
        trainrate=trainrate,
    )
    print("Implementation time : ", time.time() - st)
