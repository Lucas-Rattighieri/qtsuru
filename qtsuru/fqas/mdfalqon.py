import torch
from ..hamiltonians import Hamiltonian
from ..buffermanager import BufferManager


def mdfalqon(
    problem_hamiltonian: Hamiltonian,
    driver_hamiltonians: Hamiltonian,
    initial_state: torch.Tensor,
    time_step: float,
    num_layers: int,
    initial_betas: list = None,
    final_state: torch.Tensor = None,
    return_data: bool = False,
    track_fidelities: list = None,
    print_interval: int = 0,
    ):
    """
    Execute the Feedback-based Algorithm for Quantum OptimizatioN (FALQON) for multi-drivers.

    Parameters
    ----------
    problem_hamiltonian : Hamiltonian
        Hamiltonian encoding the optimization problem.
    driver_hamiltonians : list of Hamiltonian
        Driver Hamiltonian used for state exploration.
    initial_state : torch.Tensor
        Initial quantum state vector.
    time_step : float
        Time step Δt used in the evolution.
    num_layers : int
        Number of evolution layers to apply.
    initial_beta : list of float, optional
        Initial values of the control parameters beta (one per driver).
        If None, initialized as zeros.
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
    if not isinstance(driver_hamiltonians, (list, tuple)) or not all(
        isinstance(d, Hamiltonian) for d in driver_hamiltonians
    ):
        raise TypeError("driver_hamiltonians must be a list of Hamiltonian.")

    n_drivers = len(driver_hamiltonians)


    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)


    if initial_betas is None:
        beta = [0.0 for _ in range(n_drivers)]
    else:
        beta = list(initial_betas)

    # Coleta de dados
    if return_data:
        energies = []
        betas = []
        fidelities = [] if track_fidelities is not None else None


    # Initialization of auxiliary tensors
    manager = BufferManager(final_state.numel(), final_state.device, final_state.dtype)
    work_buffer1 = manager.get()
    work_buffer2 = manager.get()


    # Main loop
    for layer in range(1, num_layers + 1):
        # Up |psi>
        problem_hamiltonian.evolution(final_state, time_step, out=work_buffer1)

        tmp_state1 = work_buffer1
        tmp_state2 = work_buffer2

        for d in range(n_drivers):
            # Ud^(d) |psi>
            driver_hamiltonians[d].evolution(tmp_state1, beta[d] * time_step, out=tmp_state2)
            tmp_state1, tmp_state2 = tmp_state2, tmp_state1
        final_state.copy_(tmp_state1)


        # Hp |psi>
        problem_hamiltonian.hamiltonian(final_state, out=work_buffer1)

        for d in range(n_drivers):

            # Hd Hp |psi>
            driver_hamiltonians[d].hamiltonian(work_buffer1, out=work_buffer2)

            # <psi| Hd Hp |psi>
            PsiHdHpPsi = torch.vdot(final_state, work_buffer2)

            # Beta calculation: -<psi| i[Hd, Hp] |psi>
            beta[d] = -1j * (PsiHdHpPsi - torch.conj(PsiHdHpPsi))

        if return_data:
            # Expected energy: <psi| Hp |psi>
            energy = torch.vdot(final_state, work_buffer1)

            # optional fidelities: |<basis_states|psi>|^2
            if track_fidelities is not None:
                fidelities.append([float(torch.abs(final_state[basis_states]).real ** 2) for basis_states in track_fidelities])

            energies.append(float(energy.real))
            betas.append([float(b.real) for b in beta])

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
