import torch
from ..base import Hamiltonian
from ...diagonalops import DiagonalOps
from ...buffermanager import BufferManager


class Hmaxcut(Hamiltonian):
    """
    Class representing the Hamiltonian for the MaxCut problem.

    Note
    ----
    If `fix_vertex=True`, the vertex with index 0 is **always assigned to set 0**
    in all partitions.

    Parameters
    ----------
    num_vertices : int
        Total number of vertices in the graph (qubits in the circuit).
    adjacency_matrix : torch.Tensor
        Adjacency matrix representing the graph connections and weights.
    weigth : float, optional
        Multiplicative factor applied to the Hamiltonian (default: 1).
    fix_vertex : bool, optional
        If `True`, fixes vertex 0 in set 0 and reduces the number of qubits by one.
    consider_identity : bool, optional
        If `True`, adds an identity term to the Hamiltonian.
    device : str, optional
        Device where tensors will be allocated (default: "cpu").
    """

    def __init__(self, 
                 num_vertices: int,
                 adjacency_matrix, 
                 weigth: float = 1, 
                 fix_vertex: bool = False,
                 consider_identity=False,
                 device="cpu"):
                     
        self.num_vertices = num_vertices
        self.fix_vertex = fix_vertex
        self.adjacency_matrix = adjacency_matrix
        self.weigth = weigth
        self.consider_identity = consider_identity

        L = num_vertices - self.fix_vertex

        super().__init__(L, device)

        self.diag = DiagonalOps(L, self.device)
        self.diag_hamiltonian = torch.empty(2**L, device=self.device, dtype=self.dtype)
        self.create_diagonal_hamiltonian()

    def _index(self, vertex):
        """
        Adjusts the vertex index when `fix_vertex` is enabled.

        If `fix_vertex=True`, all vertex indices are shifted by one 
        because vertex 0 is treated separately.

        Parameters
        ----------
        vertex : int
            Original vertex index.

        Returns
        -------
        int
            Adjusted index or -1 if the index is invalid.
        """
        if self.fix_vertex:
            if vertex == 0 and vertex >= self.num_vertices:
                return -1
        return vertex - self.fix_vertex

    def create_diagonal_hamiltonian(self):
        """
        Constructs the Hamiltonian in diagonal form.

        Iterates over all connected vertex pairs and accumulates the corresponding
        interaction terms into `self.diag_hamiltonian`.

        Returns
        -------
        torch.Tensor
            Tensor representing the diagonal Hamiltonian.
        """
        self.diag_hamiltonian.zero_()
        out_chain = self.manager.get()
        identity_weigth = 0

        for vertex_i in range(self.num_vertices):
            index_i = self._index(vertex_i)
            for vertex_j in range(vertex_i + 1, self.num_vertices):
                index_j = self._index(vertex_j)

                if self.adjacency_matrix[vertex_i, vertex_j] != 0:
                    identity_weigth += self.adjacency_matrix[vertex_i, vertex_j]
    
                    if self.fix_vertex and vertex_i == 0:
                        self.diag.z_chain([index_j], self.adjacency_matrix[vertex_i, vertex_j], out=out_chain)
                    else:
                        self.diag.z_chain([index_i, index_j], self.adjacency_matrix[vertex_i, vertex_j], out=out_chain)
    
                    self.diag_hamiltonian.add_(out_chain)
        
        if self.consider_identity:
            out_chain.fill_(-identity_weigth)
            self.diag_hamiltonian.add_(out_chain)

        if self.weigth != 1:
            self.diag_hamiltonian.mul_(self.weigth)                

        self.manager.release(out_chain)
        return self.diag_hamiltonian

    def hamiltonian(self, psi, out=None):
        """
        Applies the Hamiltonian to the state `psi`.

        Parameters
        ----------
        psi : torch.Tensor
            State vector to which the Hamiltonian is applied.
        out : torch.Tensor, optional
            Output tensor. If `None`, a new tensor will be created.

        Returns
        -------
        torch.Tensor
            Result of applying the Hamiltonian to `psi`.
        """
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(psi, self.diag_hamiltonian, out=out)
        return out

    def evolution(self, psi, time, out=None):
        """
        Applies time evolution to the state `psi` under the Hamiltonian.

        The operation performed is:
        .. math:: \\psi(t) = e^{-iHt} \\psi(0)

        Parameters
        ----------
        psi : torch.Tensor
            Initial state vector.
        time : float
            Evolution time.
        out : torch.Tensor, optional
            Output tensor. If `None`, a new one will be created.

        Returns
        -------
        torch.Tensor
            State evolved after time `time`.
        """
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(self.diag_hamiltonian, -1j * time, out=out)
        out.exp_()
        out.mul_(psi)
        return out

    def partition(self, state):
        """
        Returns the vertex partition based on a computational basis state.

        Divides the vertices into two sets (`vertices_0` and `vertices_1`)
        based on the bits of the input state.
        If `fix_vertex=True`, vertex 0 is **always assigned to set 0**.

        Parameters
        ----------
        state : int
            Computational basis state represented as an integer.

        Returns
        -------
        tuple[list[int], list[int]]
            Two vertex sets: `vertices_0` and `vertices_1`.
        """
        vertices_0 = []
        vertices_1 = []

        if self.fix_vertex:
            vertices_0.append(0)  # vertex 0 fixed in set 0

        for i in range(self.L):
            if (state >> i) & 1:
                vertices_1.append(i + self.fix_vertex)
            else:
                vertices_0.append(i + self.fix_vertex)

        return vertices_0, vertices_1
