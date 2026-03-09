import torch
from ..hamiltonians import Hamiltonian
from ..buffermanager import BufferManager

def trfalqon(
    problem_hamiltonian: Hamiltonian,
    driver_hamiltonian: Hamiltonian,
    initial_state: torch.Tensor,
    time_step: float,
    num_layers: int,
    time_rescaling_derivative,
    args_derivative = None,
    initial_beta: float = 0.0,
    final_state: torch.Tensor = None,
    return_data: bool = False,
    track_fidelities: list = None,
    print_interval: int = 0,
    ):
    """
    Execute the Time-Rescaled Feedback-based Algorithm for Quantum OptimizatioN (TR-FALQON).

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
    final_time : float
        Total evolution time used in the rescaling function.
    time_rescaling_derivative : callable
        Function that computes the derivative of the time rescaling function.
        It must accept a float (time) and additional parameters in `args_derivative`.
    args_derivative : list, optional
        Extra arguments passed to `time_rescaling_derivative` (default: []).
    initial_beta : float, optional
        Initial value of the control parameter beta (default: 0.0).
    final_state : torch.Tensor, optional
        Preallocated tensor to store the evolving state. If None, a copy
        of the initial state is created (default: None).
    work_buffer1 : torch.Tensor, optional
        Auxiliary tensor for intermediate calculations. If None, allocated internally.
    work_buffer2 : torch.Tensor, optional
        Auxiliary tensor for intermediate calculations. If None, allocated internally.
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

    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)

    if args_derivative is None:
        args_derivative = []

    # Initialization of auxiliary tensors
    manager = BufferManager(final_state.numel(), final_state.device, final_state.dtype)
    work_buffer1 = manager.get()
    work_buffer2 = manager.get()


    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)

    beta = initial_beta

    # Data collection
    if return_data:
        energies = []
        betas = []
        fidelities = [] if track_fidelities is not None else None

    df_tau = time_rescaling_derivative(time_step, *args_derivative)

    # Main loop
    for layer in range(1, num_layers + 1):

        # Up |psi>
        problem_hamiltonian.evolution(final_state, df_tau * time_step, out=work_buffer1)
        # Ud |psi>
        driver_hamiltonian.evolution(work_buffer1, df_tau * beta * time_step, out=final_state)

        # Hp |psi>
        problem_hamiltonian.hamiltonian(final_state, out=work_buffer1)
        # Hd Hp |psi>
        driver_hamiltonian.hamiltonian(work_buffer1, out=work_buffer2)

        # <psi| Hd Hp |psi>
        PsiHdHpPsi = torch.vdot(final_state, work_buffer2)

        # f'(layer * Delta t) 
        df_tau = time_rescaling_derivative(time_step * (layer+1), *args_derivative)

        # Beta calculation: -<psi| i[Hd, Hp] |psi> / (f'_layer)
        beta = -1j * (PsiHdHpPsi - torch.conj(PsiHdHpPsi)) / df_tau

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

    # Return
    if return_data:
        if track_fidelities is not None:
            return final_state, energies, betas, fidelities
        return final_state, energies, betas
    return final_state



