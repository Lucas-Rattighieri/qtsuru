import torch
from abc import ABC, abstractmethod
from ..buffermanager import BufferManager

class Hamiltonian(ABC):
    """
    Abstract base class for quantum Hamiltonians.

    Inherits from Operators and ABC to define a common interface for Hamiltonian
    operators and their time evolution on quantum state vectors.

    Attributes:
        tmppsi1 (torch.Tensor): Temporary tensor for intermediate quantum state calculations.
        tmppsi2 (torch.Tensor): Temporary tensor for intermediate quantum state calculations.
    """


    def __init__(self, L: int, device = "cpu"):
        self.L = L
        self.dim = 2 ** L
        self.device = device
        self.dtype = torch.complex128
       
        self.manager = BufferManager.get_manager(self.dim, self.device, self.dtype)

    
    @abstractmethod
    def hamiltonian(self, psi, out=None):
        """
        Abstract method to apply the Hamiltonian operator on a quantum state vector.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            torch.Tensor: Result of applying the Hamiltonian to `psi`.
        """
        pass
    
    @abstractmethod
    def evolution(self, psi, time, out=None):
        """
        Abstract method to evolve a quantum state vector under the Hamiltonian for a given time.

        Args:
            psi (torch.Tensor): Input quantum state vector.
            time (float): Evolution time parameter.
            out (torch.Tensor, optional): Output tensor to store the evolved state. If None, a new tensor is created.

        Returns:
            torch.Tensor: Quantum state vector after evolution.
        """

        pass
