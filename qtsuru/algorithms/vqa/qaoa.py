import torch

from ...hamiltonians import Hamiltonian
from ...buffermanager import BufferManager


class QAOA:

    def __init__(self,
        L: int,
        Hp: Hamiltonian,
        Hd: Hamiltonian,
        psi0: torch.Tensor,
        copy_psi0=False,
        metrics: dict=None,
    ):

        if not isinstance(Hp, Hamiltonian):
            raise TypeError("Hp must be of type Hamiltonian.")

        if not isinstance(Hd, Hamiltonian):
            raise TypeError("Hd must be of type Hamiltonian.")

        if not isinstance(psi0, torch.Tensor):
            raise TypeError("psi0 must be of type Tensor.")

        self.L = L
        self.Hp = Hp
        self.Hd = Hd

        if metrics is None:
            self.metrics = {}
        else:
            if not isinstance(metrics, dict):
                raise TypeError("metrics must be a dict of functions.")

            if not all(callable(f) for f in metrics.values()):
                raise TypeError("All metrics must be callable.")

            self.metrics = metrics.copy()

        self.manager = BufferManager.get_manager(2 ** L, psi0.device, psi0.dtype)

        if copy_psi0:
            self.psi0 = psi0.clone()
        else:
            self.psi0 = psi0


    def add_metric(self, name: str, function):
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string.")

        if not callable(function):
            raise TypeError("Metric must be callable.")

        self.metrics[name] = function


    def remove_metric(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Metric name must be a string.")

        if name not in self.metrics:
            print(f"Metric '{name}' does not exist.")
        else:
            del self.metrics[name]


    def clear_metrics(self):
        self.metrics.clear()


    def update_psi0(self, new_psi0):
        self.psi0.copy_(new_psi0)


    @torch.no_grad()
    def run(self,
        gammas,
        betas,
        out=None,
        print_interval: int=0,
        return_data=False,
        data_to_cpu=False,
    ):

        if out is None:
            out = self.psi0.clone()
        else:
            out.copy_(self.psi0)

        device = out.device
        dtype = out.real.dtype

        gammas = torch.as_tensor(gammas, device=device, dtype=dtype)
        betas = torch.as_tensor(betas, device=device, dtype=dtype)

        if gammas.ndim != 1:
            raise ValueError("gammas must be one-dimensional.")

        if betas.ndim != 1:
            raise ValueError("betas must be one-dimensional.")

        if gammas.numel() != betas.numel():
            raise ValueError("gammas and betas must have the same length.")

        num_layers = gammas.numel()

        buffer = self.manager.get()


        # Data allocation
        if return_data:

            data = {
                "layer": torch.arange(num_layers + 1, device=device),
                "energy": torch.empty(num_layers + 1, device=device, dtype=dtype),
                "gamma": torch.empty(num_layers + 1, device=device, dtype=dtype),
                "beta": torch.empty(num_layers + 1, device=device, dtype=dtype),
            }

            # Initial energy
            self.Hp.hamiltonian(out, out=buffer)

            data["energy"][0].copy_(torch.vdot(out, buffer).real)
            data["gamma"][0].zero_()
            data["beta"][0].zero_()

            # Additional metrics
            for name, function in self.metrics.items():
                value = function(out)

                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=device)

                data[name] = torch.empty((num_layers + 1,) + value.shape, device=value.device, dtype=value.dtype)
                data[name][0] = value


        # Evolution
        for layer in range(num_layers):

            gamma = gammas[layer]
            beta = betas[layer]

            # U_p
            self.Hp.evolution(out, gamma, out=buffer)

            # U_d
            self.Hd.evolution(buffer, beta, out=out)


            # Data
            if return_data:
                self.Hp.hamiltonian(out, out=buffer)

                data["energy"][layer].copy_(torch.vdot(out, buffer).real)
                data["gamma"][layer].copy_(gamma)
                data["beta"][layer].copy_(beta)

                for name, function in self.metrics.items():
                    data[name][layer] = function(out)


            # Print
            if print_interval and layer % print_interval == 0:
                if return_data:
                    print(f"Layer {layer}, Energy = {data['energy'][layer].item()}")
                else:
                    print(f"Layer {layer}")


        self.manager.release(buffer)


        # CPU transfer
        if return_data:
            if data_to_cpu:
                data = {
                    name: value.cpu().numpy() if torch.is_tensor(value) else value
                    for name, value in data.items()
                }

            return out, data

        return out


    @torch.no_grad()
    def energy(self, parameters):

        num_layers = len(parameters) // 2

        gammas = parameters[:num_layers]
        betas = parameters[num_layers:]

        out = self.psi0.clone()
        buffer = self.manager.get()

        for layer in range(num_layers):

            # U_p
            self.Hp.evolution(out, gammas[layer], out=buffer)

            # U_d
            self.Hd.evolution(buffer, betas[layer], out=out)

        # Hp |psi>
        self.Hp.hamiltonian(out, out=buffer)

        energy = torch.vdot(out, buffer).real.item()

        self.manager.release(buffer)

        return energy
