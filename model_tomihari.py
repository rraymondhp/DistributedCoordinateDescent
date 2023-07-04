from circuit.ansatz import FraxisAnsatz, replace_FraxisAnsatz, FraxisFeatureMap
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
import random

from qiskit.primitives import Sampler

# from qiskit_ibm_runtime import Sampler  ##THIS DOES NOT WORK ANYMORE!!!!
from qiskit.compiler import transpile
import sys
import itertools


def rand_nodup(a, b, k):
    r = set()
    while len(r) < k:
        r.add(random.randint(a, b - 1))
    return list(r)


class FraxClassify:
    def __init__(
        self,
        n_qubits,
        layer_size,
        world_size,
        train_size,
        test_size,
        backend,
        params,
        update="inorder",
        train_rate=1.0,
        update_rate=1.0,
    ):
        self.n_qubits = n_qubits
        self.layer_size = layer_size
        self.world_size = world_size
        self.params = params
        self.train_size = int(train_size * train_rate)
        self.test_size = test_size
        self.backend = backend
        self.update = update
        self.update_rate = update_rate

    def fit_and_eval(self, X, y, X2, y2, isVal=True, isEval=False):
        sampler = Sampler()
        if self.update == "inorder":
            order = list(
                itertools.product(np.arange(self.layer_size), np.arange(self.n_qubits))
            )
        elif self.update == "random_all":
            order = np.random.permutation(
                list(
                    itertools.product(
                        np.arange(self.layer_size), np.arange(self.n_qubits)
                    )
                )
            )
        elif self.update == "random":
            order = np.random.permutation(
                list(
                    itertools.product(
                        np.random.randint(0, self.layer_size, self.layer_size),
                        np.random.randint(0, self.n_qubits, self.n_qubits),
                    )
                )
            )
        elif self.update == "random_per_layer":
            order = list(
                itertools.chain.from_iterable(
                    [
                        list(
                            zip(
                                np.random.randint(0, self.layer_size, self.n_qubits),
                                np.arange(self.n_qubits),
                            )
                        )
                        for _ in range(self.layer_size)
                    ]
                )
            )
        order = order[: int(self.update_rate * len(order))]
        print("order_layer: ", order)
        acc = []
        for i in range(self.layer_size * self.n_qubits):
            a, b = order[i]
            R = np.zeros((3, 3))
            train_index = rand_nodup(0, y.shape[0], self.train_size)
            for c in range(self.train_size // self.world_size):
                qcs = []
                for d in range(6):
                    qcs.append(
                        QuantumCircuit(self.n_qubits * self.world_size, self.world_size)
                    )
                feature_map = []
                for d in range(self.world_size):
                    feature_map.append(
                        FraxisFeatureMap(
                            self.n_qubits, X[train_index[d + c * self.world_size]]
                        )
                    )
                for d in range(a):
                    original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                    for e in range(6):
                        for f in range(self.world_size):
                            qcs[e].compose(
                                feature_map[f],
                                qubits=range(
                                    self.n_qubits * f, self.n_qubits * (f + 1), 1
                                ),
                                inplace=True,
                            )
                            qcs[e].compose(
                                original_ansatz,
                                qubits=range(
                                    self.n_qubits * f, self.n_qubits * (f + 1), 1
                                ),
                                inplace=True,
                            )
                for d in range(6):
                    for e in range(self.world_size):
                        qcs[d].compose(
                            feature_map[e],
                            qubits=range(self.n_qubits * e, self.n_qubits * (e + 1), 1),
                            inplace=True,
                        )
                ansatzs = replace_FraxisAnsatz(self.n_qubits, b, self.params[a])
                for d in range(6):
                    for e in range(self.world_size):
                        qcs[d].compose(
                            ansatzs[d],
                            qubits=range(self.n_qubits * e, self.n_qubits * (e + 1), 1),
                            inplace=True,
                        )
                for d in range(a + 1, self.layer_size, 1):
                    original_ansatz = FraxisAnsatz(self.n_qubits, self.params[d])
                    for e in range(6):
                        for f in range(self.world_size):
                            qcs[e].compose(
                                feature_map[f],
                                qubits=range(
                                    self.n_qubits * f, self.n_qubits * (f + 1), 1
                                ),
                                inplace=True,
                            )
                            qcs[e].compose(
                                original_ansatz,
                                qubits=range(
                                    self.n_qubits * f, self.n_qubits * (f + 1), 1
                                ),
                                inplace=True,
                            )
                for d in range(6):
                    qcs[d].measure(
                        range(0, self.world_size * self.n_qubits, self.n_qubits),
                        range(self.world_size),
                    )
                # print("B4 TRANSPILE")
                transpiled_circuits = transpile(
                    qcs, backend=self.backend, optimization_level=1
                )
                # print("AF TRANSPILE")
                # print("saving circuit...", len(transpiled_circuits))
                # for idx, _c in enumerate(transpiled_circuits):
                #     _c.draw(filename="img/circuit_"+str(idx)+"_0630"+".png", output="mpl")
                #     # _c.draw(filename="img/circuit_"+str(idx)+".tex", output="latex_source")
                # sys.exit(0)
                # print("running Sampler...")
                result = sampler.run(circuits=transpiled_circuits).result().quasi_dists
                # print("done")
                r6s = np.zeros((6, self.world_size))
                for d in range(6):
                    for e in result[d]:
                        for f in range(self.world_size):
                            if (e >> f) % 2 == 1:
                                r6s[d, f] -= result[d][e]
                            else:
                                r6s[d, f] += result[d][e]
                R[0, 0] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * 2
                    * r6s[0]
                )
                R[0, 1] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * (2 * r6s[3] - r6s[0] - r6s[1])
                )
                R[0, 2] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * (2 * r6s[4] - r6s[0] - r6s[2])
                )
                R[1, 1] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * 2
                    * r6s[1]
                )
                R[1, 2] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * (2 * r6s[5] - r6s[1] - r6s[2])
                )
                R[2, 2] += np.sum(
                    y[train_index[c * self.world_size : (c + 1) * self.world_size : 1]]
                    * 2
                    * r6s[2]
                )
            R[1, 0] = R[0, 1]
            R[2, 0] = R[0, 2]
            R[2, 1] = R[1, 2]
            # Decrease the difference in behavior
            # R = np.where(R*R<1e-8, 0, R)
            # R /= self.train_size
            print("R\n", R)
            R /= self.train_size
            print("size\n", self.train_size)
            print("R\n", R)
            eigenvalues, eigenvectors = np.linalg.eigh(R)
            self.params[a, b] = eigenvectors[:, np.argmax(eigenvalues.real)]
            print("Max value of eigenvalues: ", np.max(eigenvalues))
            if isVal:
                acc_and_score = self.eval(X, y)
                print(
                    "ACC_train: ", acc_and_score[0], "\nSCORE_train: ", acc_and_score[1]
                )
                acc.append(acc_and_score[0])
            ##COMMENTED OUT BY RRHP FOR SPEEDING UP TRAINING ON 2023/05/12
            # acc_and_score = self.eval(X2, y2)
            #
            # print('ACC_test: ',acc_and_score[0],'\nSCORE_test: ',acc_and_score[1])
            ##END OF COMMENTED OUT
            print("params\n", self.params)
            # sys.exit(0)
        # print(self.params)
        if isEval:
            acc_and_score = self.eval(X2, y2)
            print("ACC_test: ", acc_and_score[0], "\nSCORE_test: ", acc_and_score[1])
            print("Number of test data: ", y2.shape[0])
            print(
                "true_positives: ",
                acc_and_score[2],
                "\ntrue_negatives: ",
                acc_and_score[3],
                "\nfalse_positives: ",
                acc_and_score[4],
                "\nfalse_negatives: ",
                acc_and_score[5],
            )
            print("y2: ", y2)
            try:
                print(
                    "Precision: ",
                    acc_and_score[2] / (acc_and_score[2] + acc_and_score[4]),
                )
            except e:
                print("Error: ", e)
                print("Precision: nan")
            try:
                print(
                    "Recall: ", acc_and_score[2] / (acc_and_score[2] + acc_and_score[5])
                )
            except e:
                print("Error: ", e)
                print("Recall: nan")
            try:
                print(
                    "F1: ",
                    2
                    * acc_and_score[2]
                    / (2 * acc_and_score[2] + acc_and_score[4] + acc_and_score[5]),
                )
            except e:
                print("Error: ", e)
                print("F1: nan")
        return acc

    def eval(self, X, y):
        sampler = Sampler()
        acc_and_score = np.zeros(6)
        for a in range(y.shape[0] // self.world_size):
            qc = QuantumCircuit(self.n_qubits * self.world_size, self.world_size)
            feature_map = []
            for b in range(self.world_size):
                feature_map.append(
                    FraxisFeatureMap(self.n_qubits, X[b + a * self.world_size])
                )
            for b in range(self.layer_size):
                original_ansatz = FraxisAnsatz(self.n_qubits, self.params[b])
                for c in range(self.world_size):
                    qc.compose(
                        feature_map[c],
                        qubits=range(self.n_qubits * c, self.n_qubits * (c + 1), 1),
                        inplace=True,
                    )
                    qc.compose(
                        original_ansatz,
                        qubits=range(self.n_qubits * c, self.n_qubits * (c + 1), 1),
                        inplace=True,
                    )
            qc.measure(
                range(0, self.n_qubits * self.world_size, self.n_qubits),
                range(self.world_size),
            )
            result = (
                sampler.run(
                    circuits=[qc],
                )
                .result()
                .quasi_dists
            )
            Zexp = np.zeros(self.world_size)
            for b in result[0]:
                for c in range(self.world_size):
                    if (b >> c) % 2 == 1:
                        Zexp[c] -= result[0][b]
                    else:
                        Zexp[c] += result[0][b]
            true = np.where(
                y[a * self.world_size : (a + 1) * self.world_size : 1] * Zexp > 0,
                1,
                0,
            )
            false = np.where(
                y[a * self.world_size : (a + 1) * self.world_size : 1] * Zexp < 0,
                1,
                0,
            )
            true_positives = np.sum(
                np.where(
                    true * y[a * self.world_size : (a + 1) * self.world_size : 1] > 0,
                    1,
                    0,
                )
            )
            true_negatives = np.sum(
                np.where(
                    true * y[a * self.world_size : (a + 1) * self.world_size : 1] < 0,
                    1,
                    0,
                )
            )
            false_positives = np.sum(
                np.where(
                    false * y[a * self.world_size : (a + 1) * self.world_size : 1] < 0,
                    1,
                    0,
                )
            )
            false_negatives = np.sum(
                np.where(
                    false * y[a * self.world_size : (a + 1) * self.world_size : 1] > 0,
                    1,
                    0,
                )
            )
            acc_and_score[0] += np.sum(true)
            acc_and_score[1] += (
                np.sum(y[a * self.world_size : (a + 1) * self.world_size : 1] * Zexp)
                * 2
            )
            acc_and_score[2] += true_positives
            acc_and_score[3] += true_negatives
            acc_and_score[4] += false_positives
            acc_and_score[5] += false_negatives
        acc_and_score[0] /= y.shape[0]
        acc_and_score[1] /= y.shape[0]
        return acc_and_score
