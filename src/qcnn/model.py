from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qcnn.circuits import conv_layer, pool_layer
import math

def qcnn(size, estimator=None):
    est = estimator or Estimator()
    feature_map = ZFeatureMap(size)

    ansatz = QuantumCircuit(size, name="Ansatz")

    for layer in range(int(math.log2(size))):
        n = size // (2 ** layer) # num qubits in this layer
        if n < 2:
            break

        first = size - n # first qubit in this layer

        ansatz.compose(conv_layer(n, f"c{layer+1}"), list(range(first, size)), inplace=True)
        ansatz.compose(pool_layer(list(range(0, n // 2)), list(range(n // 2, n)), f"p{layer+1}"), list(range(first, size)), inplace=True)

    # Combining the feature map and ansatz
    circuit = QuantumCircuit(size)
    circuit.compose(feature_map, range(size), inplace=True)
    circuit.compose(ansatz, range(size), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (size - 1), 1)])

    # we decompose the circuit for the QNN to avoid additional data copying
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=est
    )

    return circuit, qnn