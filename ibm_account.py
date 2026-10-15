from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="Copy your token from the IBM Quantum dashboard", 
    instance="Copy your instance CRN from the IBM Quantum dashboard",
    set_as_default=True,
    overwrite=True,
)
