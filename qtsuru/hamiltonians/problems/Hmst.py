import torch
from ..base import Hamiltonian
from ...diagonalops import DiagonalOps
from ...buffermanager import BufferManager


class Hmst(Hamiltonian):
    """
    Hamiltonian model for the Minimum Spanning Tree (MST) problem.

    This class builds the diagonal Hamiltonian corresponding to the MST 
    formulation using binary variables for edges (e_{u,v}) and auxiliary 
    variables (x_{u,v}). The Hamiltonian encodes both penalty terms and 
    cost terms, making it suitable for variational quantum algorithms.
    """

    def __init__(self,
                 num_vertices: int,
                 adjacency_matrix,
                 penalty_weigth:float,
                 cost_weigth: float,
                 device="cpu"):
        """
        Initialize the Hamiltonian for the MST problem.

        Args:
            num_vertices (int): Number of vertices in the graph.
            adjacency_matrix (torch.Tensor or np.ndarray): 
                Adjacency matrix of the input graph.
            penalty_weight (float): Coefficient for constraint penalties.
            cost_weight (float): Coefficient for edge costs.
            device (str, optional): Device where tensors are allocated 
                ("cpu" or "cuda"). Default is "cpu".
        """

        self.num_vertices = num_vertices
        self.adjacency_matrix = adjacency_matrix
        self.cost_weigth = cost_weigth
        self.penalty_weigth = penalty_weigth


        self.indices = {}
        self.variables = []
        self._creat_indices()

        L = len(self.indices)

        super().__init__(L, device)

        self.diag = DiagonalOps(L, self.device)
        self.diag_hamiltonian = torch.empty(2**L, device=self.device, dtype=self.dtype)
        self.create_diagonal_hamiltonian()


    def _creat_indices(self):
        """
        Create indices for variables.

        Internal method that assigns unique indices for:
        - x_{u,v}: auxiliary variables used in constraints.
        - e_{u,v}: binary variables representing edges.

        Populates `self.indices` and `self.variables`.
        """

        k = 0

        # x_{u,v}
        for i in range(1, self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                self.indices[("x", i, j)] = k
                self.variables.append(("x", i, j))
                k += 1

        # e_{u,v}
        for i in range(self.num_vertices):
            for j in range(1, self.num_vertices):
                if i != j and self.adjacency_matrix[i,j] != 0:
                    self.indices[("e", i, j)] = k
                    self.variables.append(("e", i, j))
                    k += 1


    def _index(self, type_var, vertex_u, vertex_v):
        """
        Retrieve the index of a variable.

        Args:
            type_var (str): Type of variable ('e' or 'x').
            vertex_u (int): First vertex.
            vertex_v (int): Second vertex.

        Returns:
            int: Index of the variable, or -1 if not found.

        Raises:
            ValueError: If type_var is not 'e' or 'x'.
        """

        if not (type_var == 'e' or type_var == 'x'):
            raise ValueError("type_var must be 'e' or 'x'.")

        return self.indices.get((type_var, vertex_u, vertex_v), -1)


    def create_diagonal_hamiltonian(self):
        """
        Build the diagonal Hamiltonian.

        Constructs the full diagonal Hamiltonian as a tensor, 
        combining penalty terms (constraints F_{I,1}, F_{I,2}, F_{I,3}) 
        and the cost term O_I.

        Returns:
            torch.Tensor: Diagonal Hamiltonian of size (2^L,).
        """

        self.diag_hamiltonian.zero_()

        buffer1 = self.manager.get()
        buffer2 = self.manager.get()
        buffer2.zero_()

        # F_{I, 1} (x)
        for vertex_u in range(1, self.num_vertices):
            for vertex_v in range(vertex_u+1, self.num_vertices):
                for vertex_w in range(vertex_v+1, self.num_vertices):
                    index_Xuw = self._index("x", vertex_u, vertex_w)
                    index_Xuv = self._index("x", vertex_u, vertex_v)
                    index_Xvw = self._index("x", vertex_v, vertex_w)

                    self.diag.number_chain([index_Xuw], out=buffer1)
                    buffer2.add_(buffer1)
                    self.diag.number_chain([index_Xuv, index_Xvw], out=buffer1)
                    buffer2.add_(buffer1)
                    self.diag.number_chain([index_Xuv, index_Xuw], out=buffer1)
                    buffer2.sub_(buffer1)
                    self.diag.number_chain([index_Xuw, index_Xvw], out=buffer1)
                    buffer2.sub_(buffer1)

        self.diag_hamiltonian.add_(buffer2, alpha=self.penalty_weigth)
        buffer2.zero_()

        # F_{I, 2} (e, x)
        for vertex_u in range(1, self.num_vertices):
            for vertex_v in range(vertex_u+1, self.num_vertices):
                if self.adjacency_matrix[vertex_u, vertex_v] != 0:
                    index_Euv = self._index("e", vertex_u, vertex_v)
                    index_Evu = self._index("e", vertex_v, vertex_u)
                    index_Xuv = self._index("x", vertex_u, vertex_v)

                    self.diag.number_chain([index_Euv], out=buffer1)
                    buffer2.add_(buffer1)
                    self.diag.number_chain([index_Euv, index_Xuv], out=buffer1)
                    buffer2.sub_(buffer1)

                    self.diag.number_chain([index_Evu, index_Xuv], out=buffer1)
                    buffer2.add_(buffer1)

        self.diag_hamiltonian.add_(buffer2, alpha=self.penalty_weigth)
        buffer2.zero_()

        # F_{I, 3} (e)
        for vertex_v in range(1, self.num_vertices):
            buffer2.fill_(1)
            for vertex_u in range(0, self.num_vertices):
                if self.adjacency_matrix[vertex_u, vertex_v] != 0:

                    index_Euv = self._index("e", vertex_u, vertex_v)

                    self.diag.number_chain([index_Euv], out=buffer1)
                    buffer2.sub_(buffer1)
            buffer2.pow_(2)

            self.diag_hamiltonian.add_(buffer2, alpha=self.penalty_weigth)


        buffer2.zero_()

        # O_I (e)
        for vertex_u in range(1, self.num_vertices):
            for vertex_v in range(vertex_u+1, self.num_vertices): #vertex_u+
                if self.adjacency_matrix[vertex_u, vertex_v] != 0:

                    index_Euv = self._index("e", vertex_u, vertex_v)
                    index_Evu = self._index("e", vertex_v, vertex_u)


                    self.diag.number_chain([index_Euv], self.adjacency_matrix[vertex_u, vertex_v], out=buffer1)
                    buffer2.add_(buffer1)

                    self.diag.number_chain([index_Evu], self.adjacency_matrix[vertex_u, vertex_v], out=buffer1)
                    buffer2.add_(buffer1)

        vertex_v0 = 0
        for vertex_u in range(1, self.num_vertices):
            if self.adjacency_matrix[vertex_v0, vertex_u] != 0:

                index_Ev0u = self._index("e", vertex_v0, vertex_u)

                self.diag.number_chain([index_Ev0u], self.adjacency_matrix[vertex_v0, vertex_u], out=buffer1)
                buffer2.add_(buffer1)

        self.diag_hamiltonian.add_(buffer2, alpha=self.cost_weigth)

        self.manager.release(buffer1)
        self.manager.release(buffer2)
        return self.diag_hamiltonian


    def hamiltonian(self, psi, out = None):
        """
        Apply the Hamiltonian to a quantum state.

        Args:
            psi (torch.Tensor): Input state vector.
            out (torch.Tensor, optional): Output buffer. 
                If None, a new tensor is created.

        Returns:
            torch.Tensor: Result of H|psi>.
        """
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(psi, self.diag_hamiltonian, out = out)

        return out


    def evolution(self, psi, time, out=None):
        """
        Apply the time evolution operator exp(-i H t) to a state.

        Args:
            psi (torch.Tensor): Input state vector.
            time (float): Evolution time.
            out (torch.Tensor, optional): Output buffer. 
                If None, a new tensor is created.

        Returns:
            torch.Tensor: Evolved state.
        """
        if out is None:
            out = torch.empty_like(psi)

        torch.mul(self.diag_hamiltonian, -1j * time, out=out)
        out.exp_()
        out.mul_(psi)

        return out


    def variable_values(self, state):
        """
        Extract variable values from a computational basis state.

        Args:
            state (int): Integer encoding of the basis state.

        Returns:
            dict: Mapping from variable identifiers (tuple) 
            to binary values {0,1}.
        """
        values = {}

        for var in self.variables:
            bit = state & 1
            state >>= 1
            values[var] = bit
        return values


    def build_adjacency_matrix(self, state):
        """
        Build the adjacency matrix corresponding to a basis state.

        Args:
            state (int): Integer encoding of the basis state.

        Returns:
            torch.Tensor: Symmetric adjacency matrix where 
            edges with value 1 are active in the state.
        """
        values = self.variable_values(state)
        adj = torch.zeros((self.num_vertices, self.num_vertices))

        for (t, u, v), value in values.items():
            if t == 'e' and value == 1:
                adj[u, v] = 1
                adj[v, u] = 1

        return adj
