# Distributed QML by Fraxis ansatz/encoding (modified)

The original version is [here](https://github.com/Izuho/senior-thesis). 
This code can be run on *IBM Quantum Lab* with web interfaces, but can also be run on local computers having access to Qiskit runtime. 

Below is the example of executing Qiskit runtime

> service = QiskitRuntimeService(<br>
&emsp;&emsp;channel='ibm_quantum',<br>
&emsp;&emsp;instance='ibm-q-utokyo/internal/cs-slecture8',<br>
)

You can change the experimental settings by the following code in *main.ipynb*.

> Q = 2 # number of qubits per quantum circuit<br>
L = 2 # number of Fraxis layers, i.e., encoding and ansatz combined<br>
W = 100 # number of quantum nodes. Thus, Q*W=200 is the total number of qubits<br>
N = 63 # number of training instances<br>
M = 63 # number of testing instances<br>
U = 5 # number of sweep updates <br>
atomNo = 45 # the number of amino acid. To read data for training and testing
backend = service.backend('ibm_washington') # the name of devices

## Modification to run experiments on Side Chain Rotamer Classification

The main python notebook files are (they are basically the same)
> main_seattle_molecules.ipynb  #This is to run on simulator_mps <br>
> main_seattle_molecules-Copy1.ipynb #This is to run on ibm_seattle 

You may want to look at the `data_loader()` defined at `training_molecules.py` to get an understanding of providing instances for training and testing. Basically, you have to place the data files under the directory `data/IBM`.


