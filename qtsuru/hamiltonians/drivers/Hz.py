import torch
from ...operators import Operators
from ..base import Hamiltonian
from ...buffermanager import BufferManager


class Hz(Hamiltonian):
    """
    Quantum Hamiltonian representing a sum of Pauli-Z operators on each qubit.

    Implements the Hamiltonian and its time evolution for a transverse field
    in the Z direction, commonly used in quantum spin models.

    Methods:
        hamiltonian(psi, out=None): Applies the sum of Pauli-Z operators to state `psi`.
        evolution(psi, time, out=None): Evolves the state `psi` under the Hamiltonian
                                       for a given time.
    """


    def __init__(self, L: int, qubits = None, device="cpu"):
        """
        Initializes the Hz Hamiltonian for a system of L qubits.

        Args:
            L (int): Number of qubits.
            device (str, optional): Device for tensor operations (default 'cpu').
        """
        super().__init__(L, device)

        self.ops = Operators(L, device)

        if qubits is None:
            self.qubits = range(L)
        else:
            self.qubits = qubits
        


    def hamiltonian(self, psi, out=None):
        """
        Applies the Hamiltonian (sum of Pauli-Z) to the quantum state vector `psi`.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            out (torch.Tensor, optional): Tensor to store the result. If None, a new tensor is created.

        Returns:
            torch.Tensor: Result of the Hamiltonian applied to `psi`.
        """
        if out is None:
            out = torch.zeros_like(psi)
        else:
            out.zero_()
        
        tmppsi = self.manager.get()

        for qubit in self.qubits:
            self.ops.Z(psi, qubit, out=tmppsi)
            out.add_(tmppsi)

        self.manager.release(tmppsi)
        return out


    def evolution(self, psi, time, out=None):
        """
        Evolves the quantum state `psi` under the Hz Hamiltonian for a time `time`.

        Args:
            psi (torch.Tensor): Initial quantum state vector.
            time (float): Evolution time parameter.
            out (torch.Tensor, optional): Tensor to store the evolved state.
                                          If None, a new tensor is created.

        Returns:
            torch.Tensor: Quantum state after evolution.
        """

        tmppsi1 = self.manager.get()
        tmppsi2 = self.manager.get()

        tmppsi1.copy_(psi)

        for qubit in self.qubits:
            self.ops.Rz(tmppsi1, 2 * time, qubit, out=tmppsi2)
            tmppsi2, tmppsi1 = tmppsi1, tmppsi2

        if out is None:
            out = tmppsi1.clone()
        else:
            out.copy_(tmppsi1)        

        self.manager.release(tmppsi1)
        self.manager.release(tmppsi2)
        return out
