import torch
from .bitops import BitOps
from .buffermanager import BufferManager

class States():

    def __init__(self, L: int, device = "cpu"):
        self.device = device
        self.L = L
        self.dim = 2 ** L
        self.bitops = BitOps(L, device=device)
        self.dtype = torch.complex128

        self.indices = BufferManager.get_index(self.dim, self.device)

        self.int_manager = BufferManager.get_manager(self.dim, device, torch.int32 if L < 32 else torch.int64)
        

    def zero_vector(self, out = None):
        """
        Returns the zero vector (all entries equal to 0) in the Hilbert space of dimension 2^L.

        Parameters:
            out (torch.Tensor, optional): Output tensor to store the result.
                If None, a new tensor is created.

        Returns:
            torch.Tensor: Zero vector with shape (2^L,) and complex dtype.
        """

        if out is None:
            out = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.zero_()

        return out


    def uniform_superposition(self, out=None):
        """
        Generates the normalized uniform superposition over all computational basis states.
    
        The resulting state corresponds to:
            (1 / sqrt(2^L)) * sum_{x=0}^{2^L - 1} |x⟩
    
        Parameters:
        - out (torch.Tensor, optional): Tensor to store the output state. If None, a new tensor is allocated.
    
        Returns:
        - torch.Tensor: Normalized state vector of shape (2**L,) representing the uniform superposition.
        """
        if out is None:
            out = torch.ones(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.fill_(1)
    
        out.mul_(2 ** (-self.L / 2))
        return out


    def z_state(self, state:int, out = None) -> torch.Tensor:
        """
        Returns the computational basis state |state>.

        Parameters:
            state (int): Index of the desired basis state (0 <= state < 2^L).
            out (torch.Tensor, optional): Output tensor to store the state.
                If None, a new tensor is created.

        Returns:
            torch.Tensor: State vector in the Hilbert space.
        """

        if out is None:
            out = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.zero_()

        out[state] = 1

        return out


    def x_state(self, state:int, out = None) -> torch.Tensor:
        """
        Returns the state |state> in the X basis, i.e., H^{⊗L} |state>.

        Parameters:
            state (int): Index of the basis state in the computational basis.
            out (torch.Tensor, optional): Output tensor.
                If None, a new tensor is created.

        Returns:
            torch.Tensor: State vector after applying H^{⊗L} to |state>.
        """

        if out is None:
            out = torch.ones(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.fill_(1)

        tmp1 = self.int_manager.get()
        tmp2 = self.int_manager.get()


        qubits_ones = [qubit for qubit in range(self.L) if state & (1 << qubit)]

        self.bitops.xor_bits(self.indices, qubits_ones, out=tmp1)

        torch.add(1, tmp1, alpha= -2, out=tmp2)
        out.mul_(tmp2)
        out.mul_(2**(-self.L / 2))

        self.int_manager.release(tmp1)
        self.int_manager.release(tmp2)
        return out


    def y_state(self, state: int, out = None) -> torch.Tensor:
        """
        Returns the state |state> in the Y basis, i.e., (SH)^{⊗L} |state>.

        Parameters:
            state (int): Index of the basis state in the computational basis.
            out (torch.Tensor, optional): Output tensor.
                If None, a new tensor is created.

        Returns:
            torch.Tensor: State vector after applying (SH)^{⊗L} to |state>.
        """

        if out is None:
            out = torch.empty(self.dim, dtype=self.dtype, device=self.device)
        
        tmp1 = self.int_manager.get()
        tmp2 = self.int_manager.get()

        self.bitops.count_bits(self.indices, out=tmp1)

        torch.bitwise_and(tmp1, 1, out=tmp2)
        torch.mul(tmp2, (1j - 1), out=out)
        out.add_(1)

        tmp1.bitwise_right_shift_(1)
        tmp1.bitwise_and_(1)
        torch.add(1, tmp1, alpha= -2, out=tmp2)
        out.mul_(tmp2)

        torch.bitwise_and(self.indices, state, out=tmp2)
        qubits_ones = [qubit for qubit in range(self.L) if state & (1 << qubit)]

        self.bitops.xor_bits(self.indices, qubits_ones, out=tmp1)

        torch.add(1, tmp1, alpha= -2, out=tmp2)
        out.mul_(tmp2)
        out.mul_(2**(-self.L / 2))

        self.int_manager.release(tmp1)
        self.int_manager.release(tmp2)
        return out


    def hamming_weight_state(self, hamming_weight: int, out = None) -> torch.Tensor:
        """
        Generates a normalized quantum state corresponding to the uniform superposition
        of all computational basis states with a fixed Hamming weight.

        Args:
            hamming_weight (int): Number of 1s in the bitstring (Hamming weight).
            out (torch.Tensor, optional): Output tensor to store the result. If None, a new tensor is created.

        Returns:
            torch.Tensor: A normalized tensor of shape (2**L,) representing the quantum state
            where only the basis states with the given Hamming weight are nonzero.
        """
        if out is None:
            out = torch.ones(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.fill_(1)

        if hamming_weight == 0:
            self.z_state(0, out)
            return out
        
        tmp = self.int_manager.get()

        self.bitops.count_bits(self.indices, out=tmp)

        torch.eq(tmp, hamming_weight, out=out)

        norm = torch.linalg.norm(out)
        out.div_(norm)

        self.int_manager.release(tmp)
        return out
      
