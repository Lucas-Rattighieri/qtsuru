import torch
from ..hamiltonians import Hamiltonian
from ..buffermanager import BufferManager


def sofalqon(
    problem_hamiltonian: Hamiltonian,
    driver_hamiltonian: Hamiltonian,
    initial_state: torch.Tensor,
    time_step: float,
    num_layers: int,
    initial_beta: float = 0.0,
    hybrid_approach: bool = True,
    final_state: torch.Tensor = None,
    return_data: bool = False,
    track_fidelities: list = None,
    print_interval: int = 0,
    ):
    """
    Execute the Second Order Feedback-based Algorithm for Quantum OptimizatioN (SO-FALQON).

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
    hybrid_approach: bool, optional  
        Defines the strategy for computing the control parameter β.  
        - If True, uses a hybrid approach, alternating between the β 
            from the standard FALQON (first-order) rule and the second-order β, 
            selecting the most suitable value at each iteration.  
        - If False, exclusively applies the second-order β in all layers.
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


    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)

    # Initialization of auxiliary tensors
    manager = BufferManager(final_state.numel(), final_state.device, final_state.dtype)
    work_buffer1 = manager.get()
    work_buffer2 = manager.get()



    if final_state is None:
        final_state = initial_state.clone()
    else:
        final_state.copy_(initial_state)

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

        if return_data:
            # Expected energy: <psi| Hp |psi>
            energy = torch.vdot(final_state, work_buffer1)
            energies.append(float(energy.real))

        # Hd Hp |psi>
        driver_hamiltonian.hamiltonian(work_buffer1, out=work_buffer2)

        # <psi| Hd Hp |psi>
        PsiHdHpPsi = torch.vdot(final_state, work_buffer2)

        # <psi| Hp Hd Hp |psi>
        PsiHpHdHpPsi = torch.vdot(work_buffer1, work_buffer2)

        # Hd |psi>
        driver_hamiltonian.hamiltonian(final_state, out=work_buffer1)

        # <psi| Hd Hd Hp |psi>
        PsiHdHdHpPsi = torch.vdot(work_buffer1, work_buffer2)

        # Hp Hd |psi>
        problem_hamiltonian.hamiltonian(work_buffer1, out=work_buffer2)

        # <psi| Hd Hp Hd |psi>
        PsiHdHpHdPsi = torch.vdot(work_buffer1, work_buffer2)

        # Hp Hp Hd |psi>
        problem_hamiltonian.hamiltonian(work_buffer2, out=work_buffer1)

        # <psi| Hp Hp Hd |psi>
        PsiHpHpHdPsi = torch.vdot(final_state, work_buffer1)


        # <psi| i[Hd, Hp] |psi>
        A = 1j * (PsiHdHpPsi - torch.conj(PsiHdHpPsi))

        beta1 = - A

        # 0.5 <psi| [[Hd, Hp], Hd] |psi>
        B = PsiHdHpHdPsi - 0.5 * (PsiHdHdHpPsi + torch.conj(PsiHdHdHpPsi))

        # <psi| [[Hd, Hp], Hp] |psi>
        C = - 2 * PsiHpHdHpPsi + PsiHpHpHdPsi + torch.conj(PsiHpHpHdPsi)

        beta2 = - (A + time_step * C) / (2 * time_step * torch.abs(B))

        if hybrid_approach:
            beta = beta2 if torch.abs(beta1) > torch.abs(beta2) else beta1
        else:
            # beta2 = - (A + time_step * C) / (2 * time_step * (B**2+1)**(0.5))
            beta = beta2 

        if return_data:
            # optional fidelities: |<basis_states|psi>|^2
            if track_fidelities is not None:
                fidelities.append([float(torch.abs(final_state[basis_states]).real ** 2) for basis_states in track_fidelities])

            betas.append(float(beta.real))

            if print_interval and layer % print_interval == 0:
                print(f"Layer {layer}, E = {energy.real}")

    # Return
    if return_data:
        if track_fidelities is not None:
            return final_state, energies, betas, fidelities
        return final_state, energies, betas

    return final_state


















