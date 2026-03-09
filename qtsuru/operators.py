import torch
from .bitops import BitOps
from .buffermanager import BufferManager

class Operators():

    def __init__(self, L: int, device = "cpu"):
        self.device = device
        self.L = L
        self.dim = 2 ** L
        self.dtype = torch.complex128

        self.bitops = BitOps(L, device=device)
            
        self.indices = BufferManager.get_index(self.dim, self.device)

        self.int_manager = self.bitops.manager
        self.manager = BufferManager.get_manager(self.dim, device, self.dtype)

   
    
    def X(self, psi: torch.Tensor, qubits, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the Pauli-X (quantum NOT) gate to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubits (int or list[int]): Index or list of indices of qubits to apply the X gate.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the X gate(s).
        """

        if out is None:
            nout = torch.empty_like(psi)
        elif out is psi:
            nout = self.manager.get()
        else:
            nout = out
        
        tmp = self.int_manager.get()
        
        self.bitops.flip_bits(self.indices, qubits, out=tmp)
        
        torch.index_select(psi, 0, tmp, out=nout)

        self.int_manager.release(tmp)
        
        if out is psi:
            psi.copy_(nout)
            self.manager.release(nout)
        return out


    def Z(self, psi: torch.Tensor, qubits, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the Pauli-Z gate to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubits (int or list[int]): Index or list of indices of qubits to apply the Z gate.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the Z gate(s).
        """  
        if out is None:
            out = torch.empty_like(psi)

        tmp = self.int_manager.get()

        self.bitops.xor_bits(self.indices, qubits, out=tmp)

        torch.add(1, tmp, alpha= -2, out=tmp)

        torch.mul(tmp, psi, out=out)

        self.int_manager.release(tmp)
        return out


    def Y(self, psi: torch.Tensor, qubits, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the Pauli-Y gate to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubits (int or list[int]): Index or list of indices of qubits to apply the Y gate.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the Y gate(s).
        """
        if out is None:
            nout = torch.empty_like(psi)
        elif out is psi:
            nout = self.manager.get()
        else:
            nout = out
        
        tmp = self.int_manager.get()

        self.bitops.flip_bits(self.indices, qubits, out=tmp)
        torch.index_select(psi, 0, tmp, out=nout)

        self.bitops.xor_bits(self.indices, qubits, out=tmp)
        torch.add(1, tmp, alpha= -2, out=tmp)

        nout.mul_(tmp)
        if isinstance(qubits, int):
            torch.mul(nout, -1j, out=out)
        else:
            torch.mul(nout, (-1j) ** len(qubits), out=out)
        if out is psi:
            self.manager.release(nout)

        self.int_manager.release(tmp)
        return out


    def H(self, psi: torch.Tensor, qubit: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the Hadamard gate to the specified qubit of the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubit (int): Index of the qubit to which the Hadamard gate is applied.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Raises:
            ValueError: If `qubit` is not an integer.

        Returns:
            torch.Tensor: State vector after applying the Hadamard gate.
        """

        if not isinstance(qubit, int):
            raise ValueError("qubit must be an integer")

        if out is None:
            out = torch.empty_like(psi)
        
        self.X(psi, qubit, out=out)

        tmp = self.int_manager.get()

        self.bitops.get_bit(self.indices, qubit, out=tmp)
        torch.add(1, tmp, alpha= -2, out=tmp)

        out.mul_(tmp)
        out.add_(psi)
        out.mul_(tmp)
        out.mul_(0.7071067811865475)

        self.int_manager.release(tmp)
        return out


    def S(self, psi: torch.Tensor, qubit: int, dagger = False, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the phase gate S (or its Hermitian conjugate S† if dagger=True) to the specified qubit
        of the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubit (int): Index of the qubit to which the S or S† gate is applied.
            dagger (bool, optional): If True, applies the Hermitian conjugate S† gate. Defaults to False.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Raises:
            ValueError: If `qubit` is not an integer.

        Returns:
            torch.Tensor: State vector after applying the S or S† gate.
        """
        if not isinstance(qubit, int):
            raise ValueError("qubit must be an integer")

        phase = -1 if dagger else 1

        if out is None:
            out = torch.empty_like(psi)
        
        tmp = self.int_manager.get()

        self.bitops.get_bit(self.indices, qubit, out=tmp)
        torch.mul(tmp, phase * 1j - 1, out=out)
        out.add_(1)
        out.mul_(psi)

        self.int_manager.release(tmp)
        return out


    def Rx(self, psi: torch.Tensor, angle, qubit: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the single-qubit rotation Rx gate by the specified angle to the given qubit
        of the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            angle (float or torch.Tensor): Rotation angle in radians.
            qubit (int): Index of the qubit to apply the Rx rotation.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Raises:
            ValueError: If `qubit` is not an integer.

        Returns:
            torch.Tensor: State vector after applying the Rx rotation.
        """
        if not isinstance(qubit, int):
            raise ValueError("qubits must be an integer")

        if out is None:
            out = torch.empty_like(psi)
        

        if not isinstance(angle, torch.Tensor):
            theta = torch.tensor(angle, dtype=self.dtype, device=self.device)
        else:
            theta = angle.to(dtype=self.dtype, device=self.device)

        ctheta = torch.cos(theta / 2)
        istheta = 1j * torch.sin(theta / 2)

        self.X(psi, qubit, out=out)
        out.mul_(-istheta)
        out.add_(psi, alpha=ctheta)

        return out


    def Ry(self, psi: torch.Tensor, angle, qubit: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the single-qubit rotation Ry gate by the specified angle to the given qubit
        of the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            angle (float or torch.Tensor): Rotation angle in radians.
            qubit (int): Index of the qubit to apply the Rx rotation.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Raises:
            ValueError: If `qubit` is not an integer.

        Returns:
            torch.Tensor: State vector after applying the Rx rotation.
        """
        if not isinstance(qubit, int):
            raise ValueError("qubits must be an integer")

        if out is None:
            out = torch.empty_like(psi)
        

        if not isinstance(angle, torch.Tensor):
            theta = torch.tensor(angle, dtype=self.dtype, device=self.device)
        else:
            theta = angle.to(dtype=self.dtype, device=self.device)

        ctheta = torch.cos(theta / 2)
        istheta = 1j * torch.sin(theta / 2)

        self.Y(psi, qubit, out=out)
        out.mul_(-istheta)
        out.add_(psi, alpha=ctheta)

        return out


    def Rz(self, psi: torch.Tensor, angle, qubit: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the single-qubit rotation Rz gate by the specified angle to the given qubit
        of the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            angle (float or torch.Tensor): Rotation angle in radians.
            qubit (int): Index of the qubit to apply the Rx rotation.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Raises:
            ValueError: If `qubit` is not an integer.

        Returns:
            torch.Tensor: State vector after applying the Rx rotation.
        """
        if not isinstance(qubit, int):
            raise ValueError("qubits must be an integer")

        if out is None:
            out = torch.empty_like(psi)
        
 
        if not isinstance(angle, torch.Tensor):
            theta = torch.tensor(angle, dtype=self.dtype, device=self.device)
        else:
            theta = angle.to(dtype=self.dtype, device=self.device)

        ctheta = torch.cos(theta / 2)
        istheta = 1j * torch.sin(theta / 2)

        self.Z(psi, qubit, out=out)
        out.mul_(-istheta)
        out.add_(psi, alpha=ctheta)

        return out


    def CNOT(self, psi: torch.Tensor, control: int, target: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the controlled-NOT (CNOT) gate with specified control and target qubits
        to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            control (int): Index of the control qubit.
            target (int): Index of the target qubit.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the CNOT gate.
        """
        if out is None:
            out = torch.empty_like(psi)
        
        tmp = self.int_manager.get()

        self.bitops.get_bits(self.indices, control, out=tmp)
        tmp.bitwise_left_shift_(target)
        tmp.bitwise_xor_(self.indices)

        torch.index_select(psi, 0, tmp, out=out)

        self.int_manager.release(tmp)
        return out


    def CZ(self, psi: torch.Tensor, control: int, target: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the controlled-Z (CZ) gate with specified control and target qubits
        to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            control (int): Index of the control qubit.
            target (int): Index of the target qubit.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the CZ gate.
        """
        if out is None:
            out = torch.empty_like(psi)
        
        tmp = self.int_manager.get()

        if target < control:
            control, target = target, control


        tmp.copy_(self.indices)

        tmp.bitwise_right_shift_(target - control)
        tmp.bitwise_and_(self.indices)
        tmp.bitwise_right_shift_(control)
        tmp.bitwise_and_(1)

        torch.add(1, tmp, alpha= -2, out=tmp)

        torch.mul(psi, tmp, out=out)

        self.int_manager.release(tmp)
        return out


    def SWAP(self, psi: torch.Tensor, qubit_i: int, qubit_j: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies the SWAP gate, exchanging the states of two qubits in the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            qubit_i (int): Index of the first qubit to swap.
            qubit_j (int): Index of the second qubit to swap.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the SWAP gate.
        """
        if out is None:
            out = torch.empty_like(psi)
        
        tmp = self.int_manager.get()

        self.bitops.permute_bits(self.indices, qubit_i, qubit_j, out=tmp)
        torch.index_select(psi, 0, tmp, out=out)

        self.int_manager.release(tmp)
        return out


    def tofolli(self, psi: torch.Tensor, controls, target: int, out: torch.Tensor = None) -> torch.Tensor:
        """
        Applies a Toffoli (CCNOT) gate with multiple control qubits and one target qubit
        to the quantum state vector `psi`.

        Parameters:
            psi (torch.Tensor): Quantum state vector of shape (2**L,).
            controls (list[int]): List of indices of control qubits.
            target (int): Index of the target qubit.
            out (torch.Tensor, optional): Output tensor to store the result.
                                          If not provided, a new tensor with the same shape as `psi` is created.

        Returns:
            torch.Tensor: State vector after applying the Toffoli gate.
        """
        if out is None:
            out = torch.empty_like(psi)
        
        tmp = self.int_manager.get()

        self.bitops.and_bits(self.indices, controls, out=tmp)
        tmp.bitwise_left_shift_(target)
        tmp.bitwise_xor_(self.indices)

        torch.index_select(psi, 0, tmp, out=out)
      
        self.int_manager.release(tmp)
        return out
