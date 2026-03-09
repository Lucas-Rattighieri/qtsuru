import torch
from .buffermanager import BufferManager

class DiagonalOps:

    def __init__(self, L: int, device='cpu'):

        self.device = device
        self.L = L
        self.dim = 2 ** L
        self.dtype = torch.complex128

        self.manager = BufferManager.get_manager(self.dim, self.device, self.dtype)

        self.diag_z = torch.tensor([1.0, -1.0], dtype=self.dtype, device=self.device)
        self.diag_number = torch.tensor([0.0, 1.0], dtype=self.dtype, device=self.device)



    def operator(self, op: torch.Tensor, pos: list[int], coef: complex = 1, out = None):
        """
        Build a diagonal operator as the Kronecker product of `op` applied to
        the specified qubit positions, scaled by a coefficient.

        Parameters
        ----------
        op : torch.Tensor
            1-qubit diagonal operator to be applied at the given positions.
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The resulting diagonal operator as a 1D tensor of length `2**L`.
        """

        if out is None:
            out = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            out.zero_()

        if coef == 0:
            return out

        
        for i, k in enumerate(pos):
            if k < 0 or k >= self.L:
                raise ValueError(f"Positions must be between 0 and {self.L}")
            else:
                pos[i] = self.L - k - 1
                
        pos = sorted(pos)



        resultado = self.manager.get()
        ones = self.manager.get()
        ones.fill_(1)
        # print('diagops', id(resultado), id(ones), id(out))

        resultado[0] = coef
        nout = out

        j = 0
        size = 1

        for i in pos:
            if i > j:
                ident = ones[:2 ** (i-j)]

                torch.kron(resultado[:size], ident, out=nout[:size * 2 ** (i-j)])
                nout, resultado = resultado, nout
                size *= 2 ** (i-j)

            torch.kron(resultado[:size], op, out=nout[:size*2])
            nout, resultado = resultado, nout

            size *= 2
            j = i + 1


        if j < self.L:
            ident = ones[:2 ** (self.L - j)]
            torch.kron(resultado[:size], ident, out=nout[:size * (2 ** (self.L - j))])
            nout, resultado = resultado, nout

        if out is not resultado:
            out.copy_(resultado)
        else:
            nout, resultado = resultado, nout

        self.manager.release(resultado)
        self.manager.release(ones)
        return out


    def z_chain(self, pos: list[int], coef: complex = 1, out = None):
        """
        Generate a diagonal operator consisting of a chain of Pauli-Z operators
        applied at the specified qubit positions.

        Parameters
        ----------
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The diagonal representation of the Z-chain operator as a 1D tensor.
        """
        return self.operator(self.diag_z, pos, coef, out)


    def number_chain(self, pos: list[int], coef: complex = 1, out = None):
        """
        Generate a diagonal operator consisting of a chain of number operators
        (|1><1| projectors) applied at the specified qubit positions.

        Parameters
        ----------
        pos : list[int]
            List of qubit positions (0-based, leftmost qubit is position 0).
        coef : complex, optional
            Scalar factor multiplying the resulting operator (default: 1).
        out : torch.Tensor, optional
            Preallocated output tensor to store the result. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor
            The diagonal representation of the number-chain operator as a 1D tensor.
        """
        return self.operator(self.diag_number, pos, coef, out)
