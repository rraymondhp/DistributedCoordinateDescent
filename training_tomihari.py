import numpy as np
from model_tomihari import FraxClassify
import time
import pandas as pd
from qiskit_ibm_runtime import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def data_loader(
    num_train,
    num_test,
    isFashion=False,
    CSVpath=None,
    feat=None,
    label=None,
    preprocessing=None,
):
    try:
        if isFashion:
            print("Using Fashion MNIST")
            test_label = np.load("data/fmnist_test_Label.npy")[0:num_test]
            train_label = np.load("data/fmnist_train_Label.npy")[0:num_train]
            test_feat = np.load("data/fmnist_test_feat.npy")[0:num_test]
            train_feat = np.load("data/fmnist_train_feat.npy")[0:num_train]
        elif CSVpath is not None:
            print("Using CSV: ", CSVpath)
            try:
                data = pd.read_csv(CSVpath)
            except Exception as e:
                print("Error Cannot read CSV file: ", CSVpath)
                raise e
            if preprocessing == "Titanic":
                print("Using Titanic preprocessing")
                train_label, train_feat, test_label, test_feat = preprocessing_titanic(
                    data, num_train, num_test
                )
            else:
                train, test = train_test_split(data, test_size=0.2, random_state=0)
                print("No preprocessing")
                train_label = train[label].to_numpy()[0:num_train]
                train_feat = train.drop(label, axis=1).to_numpy()[0:num_train]
                test_label = test[label].to_numpy()[0:num_test]
                test_feat = test.drop(label, axis=1).to_numpy()[0:num_test]
        else:
            print("Using MNIST")
            test_label = np.load("data/mnist_test_Label.npy")[0:num_test]
            train_label = np.load("data/mnist_train_Label.npy")[0:num_train]
            test_feat = np.load("data/mnist_test_feat.npy")[0:num_test]
            train_feat = np.load("data/mnist_train_feat.npy")[0:num_train]
        return test_label, train_label, test_feat, train_feat
    except Exception as e:
        print("Error in data_loader:", e)
        raise e


def cut_data(train_label, train_feat, test_label, test_feat, rank, world_size):
    data_len_min = len(train_feat) // world_size
    offset = len(train_feat) % world_size
    if rank < offset:
        start1 = rank * (data_len_min + 1)
        end1 = start1 + data_len_min + 1
    else:
        start1 = offset * (data_len_min + 1) + (rank - offset) * data_len_min
        end1 = start1 + data_len_min
    data_len_min = len(test_feat) // world_size
    offset = len(test_feat) % world_size
    if rank < offset:
        start2 = rank * (data_len_min + 1)
        end2 = start2 + data_len_min + 1
    else:
        start2 = offset * (data_len_min + 1) + (rank - offset) * data_len_min
        end2 = start2 + data_len_min

    return (
        train_label[start1:end1],
        train_feat[start1:end1],
        test_label[start2:end2],
        test_feat[start2:end2],
    )


def preprocessing_titanic(data, num_train, num_test):
    # Fill Null values
    mean = data["Age"].mean()
    std = data["Age"].std()
    is_null = data["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = data["Age"].astype(int)
    data["Embarked"] = data["Embarked"].fillna("S")
    data = data.fillna(data["Fare"].mean())

    # Label Encoding
    le = LabelEncoder()
    data["Pclass"] = le.fit_transform(data["Pclass"])
    le = LabelEncoder()
    data["Sex"] = le.fit_transform(data["Sex"])
    le = LabelEncoder()
    data["Embarked"] = le.fit_transform(data["Embarked"])
    # print("info:", data.info())

    # Drop the unnecessary columns
    data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Relabeling
    data["Survived"] = data["Survived"].replace(0, -1)

    # Train Test Split
    train, test = train_test_split(data, test_size=0.2)
    train_label = train["Survived"].to_numpy()[0:num_train]
    train_feat = train.drop(["Survived"], axis=1).to_numpy()[0:num_train]
    test_label = test["Survived"].to_numpy()[0:num_test]
    test_feat = test.drop(["Survived"], axis=1).to_numpy()[0:num_test]

    # Standardization
    sc = StandardScaler()
    train_feat = sc.fit_transform(train_feat)
    test_feat = sc.transform(test_feat)

    # Normalization
    train_feat = np.kron(train_feat, train_feat)
    train_feat = np.kron(test_feat, test_feat)
    train_feat /= np.linalg.norm(train_feat, ord=2, axis=1, keepdims=True)
    test_feat /= np.linalg.norm(test_feat, ord=2, axis=1, keepdims=True)

    return train_label, train_feat, test_label, test_feat


def parallel_train(
    n_qubits,
    layer_size,
    world_size,
    num_train,
    num_test,
    update_iter,
    service,
    backend,
    params,
    isFashion=False,
    CSVpath=None,
    preprocessing=None,
    update="inorder",
    trainrate=1.0,
    isVal=True,
    isEval=True,
    label=None,
    feat=None,
):
    """
    Parallel training of the model
    Args:
        n_qubits: number of qubits
        layer_size: number of layers
        world_size: number of workers
        num_train: number of training data
        num_test: number of test data
        update_iter: number of iterations
        service: service name
        backend: backend name
        params: parameters for the model
        isFashion: whether to use fashion mnist
        CSVpath: path to the csv file
        preprocessing: preprocessing function
        update: update method
        trainrate: training rate
        isVal: whether to use validation set
        isEval: whether to use test set
        label: label data
        feat: feature data
    """
    test_label, train_label, test_feat, train_feat = data_loader(
        num_train,
        num_test,
        isFashion=isFashion,
        CSVpath=CSVpath,
        feat=feat,
        label=label,
        preprocessing=preprocessing,
    )
    model = FraxClassify(
        n_qubits,
        layer_size,
        world_size,
        num_train,
        num_test,
        backend,
        params,
        update=update,
        trainrate=trainrate,
    )
    with Session(service=service, backend=backend):
        for i in range(update_iter):
            st = time.time()
            model.fit_and_eval(
                train_feat,
                train_label,
                test_feat,
                test_label,
                isVal=isVal,
                isEval=isEval,
            )
            print("Implementation time: ", time.time() - st)
            print("_______________NEW________________ Iteration: ", i)
