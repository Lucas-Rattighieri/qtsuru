import torch
from ..hamiltonians import Hamiltonian
from ..buffermanager import BufferManager

def falqon(
    problem_hamiltonian: Hamiltonian,
    driver_hamiltonian: Hamiltonian,
    initial_state: torch.Tensor,
    time_step: float,
    num_layers: int,
    initial_beta: float = 0.0,
    final_state: torch.Tensor = None,
    return_data: bool = False,
    track_fidelities: list = None,
    print_interval: int = 0,
    ):
    """
    Execute the Feedback-based Algorithm for Quantum OptimizatioN (FALQON).

    Parameters
    ----------
    problem_hamiltonian : Hamiltonian
        Hamiltonian encoding the optimization problem.
    driver_hamiltonian : Hamiltonian
        Driver Hamiltonian used for state exploration.
    initial_state : torch.Tensor
        Initial quantum state vector.
    time_step : float
        Time step Δt used in the evolution.
    num_layers : int
        Number of evolution layers to apply.
    initial_beta : float, optional
        Initial value of the control parameter beta (default: 0.0).
    final_state : torch.Tensor, optional
        Preallocated tensor to store the evolving state. If None, a copy
        of the initial state is created (default: None).
    return_data : bool, optional
        If True, also return energies, betas, and optionally fidelities (default: False).
    track_fidelities : list of int, optional
        Indices of basis states to track fidelities during the evolution.
        If None, fidelities are not computed (default: None).
    print_interval : int, optional
        If > 0, prints the energy value every `print_interval` layers (default: 0).

    Returns
    -------
    torch.Tensor
        Final state after evolution if `return_data` is False.
    tuple
        If `return_data` is True, returns:
            (final_state, energies, betas)
        or
            (final_state, energies, betas, fidelities) if `track_fidelities` is provided.
    """


    # Type checking of Hamiltonians
    if not isinstance(problem_hamiltonian, Hamiltonian):
        raise TypeError("problem_hamiltonian must be of type Hamiltonian.")
    if not isinstance(driver_hamiltonian, Hamiltonian):
        raise TypeError("driver_hamiltonian must be of type Hamiltonian.")


    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)

    # Initialization of auxiliary tensors
    manager = BufferManager(final_state.numel(), final_state.device, final_state.dtype)
    work_buffer1 = manager.get()
    work_buffer2 = manager.get()


    beta = initial_beta

    # Coleta de dados
    if return_data:
        energies = []
        betas = []
        fidelities = [] if track_fidelities is not None else None

    # Main loop
    for layer in range(1, num_layers + 1):
        # Up |psi>
        problem_hamiltonian.evolution(final_state, time_step, out=work_buffer1)
        # Ud |psi>
        driver_hamiltonian.evolution(work_buffer1, beta * time_step, out=final_state)

        # Hp |psi>
        problem_hamiltonian.hamiltonian(final_state, out=work_buffer1)
        # Hd Hp |psi>
        driver_hamiltonian.hamiltonian(work_buffer1, out=work_buffer2)

        # <psi| Hd Hp |psi>
        PsiHdHpPsi = torch.vdot(final_state, work_buffer2)

        # Beta calculation: -<psi| i[Hd, Hp] |psi>
        beta = -1j * (PsiHdHpPsi - torch.conj(PsiHdHpPsi))

        if return_data:
            # Expected energy: <psi| Hp |psi>
            energy = torch.vdot(final_state, work_buffer1)

            # optional fidelities: |<basis_states|psi>|^2
            if track_fidelities is not None:
                fidelities.append([float(torch.abs(final_state[basis_states]).real ** 2) for basis_states in track_fidelities])

            energies.append(float(energy.real))
            betas.append(float(beta.real))

            if print_interval and layer % print_interval == 0:
                print(f"Layer {layer}, E = {energy.real}")
    
    manager.release(work_buffer1)
    manager.release(work_buffer2)

    # Return
    if return_data:
        if track_fidelities is not None:
            return final_state, energies, betas, fidelities
        return final_state, energies, betas

    return final_state
