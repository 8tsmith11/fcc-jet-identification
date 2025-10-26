from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def ansatz(num_qubits, reps=5):
    return RealAmplitudes(num_qubits, reps=reps)

def auto_encoder_circuit(num_latent, num_trash):
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)
    circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()
    auxiliary_qubit = num_latent + 2 * num_trash
    # swap test
    circuit.h(auxiliary_qubit)
    for i in range(num_trash):
        circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)

    circuit.h(auxiliary_qubit)
    circuit.measure(auxiliary_qubit, cr[0])
    return circuit