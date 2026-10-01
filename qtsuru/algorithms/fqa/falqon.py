import torch
from ...hamiltonians import Hamiltonian
from ...buffermanager import BufferManager


class FALQON:

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
        delta_t: float,
        num_layers: int,
        beta_0: float=0,
        out=None,
        print_interval: int=0,
        return_data=False,
        data_to_cpu=False,
    ):

        if out is None:
            out = self.psi0.clone()
        else:
            out.copy_(self.psi0)

        buffer1 = self.manager.get()
        buffer2 = self.manager.get()

        device = out.device
        dtype = out.real.dtype


        PsiHdHpPsi = torch.empty((), device=device, dtype=out.dtype)

        beta = torch.as_tensor(beta_0, device=device, dtype=dtype)
        effective_dt = torch.empty((), device=device, dtype=dtype)


        # Data allocation
        if return_data:

            data = {
                "layer": torch.arange(num_layers + 1, device=device),
                "energy": torch.empty(num_layers + 1, device=device, dtype=dtype),
                "beta": torch.empty(num_layers + 1, device=device, dtype=dtype),
            }


            # Initial energy
            self.Hp.hamiltonian(out, out=buffer1)

            data["energy"][0].copy_(torch.vdot(out, buffer1).real)

            data["beta"][0].fill_(0)


            # Additional metrics
            for name, function in self.metrics.items():

                value = function(out)

                if not torch.is_tensor(value):
                    value = torch.as_tensor(value, device=device)

                data[name] = torch.empty((num_layers + 1,) + value.shape, device=value.device, dtype=value.dtype)

                data[name][0] = value


        # Evolution
        for layer in range(1, num_layers + 1):

            # U_p
            self.Hp.evolution(out, delta_t, out=buffer1)

            # U_d
            torch.mul(beta, delta_t, out=effective_dt)
            self.Hd.evolution(buffer1, effective_dt, out=out)


            # Hp |psi>
            self.Hp.hamiltonian(out, out=buffer1)

            # Hd Hp |psi>
            self.Hd.hamiltonian(buffer1, out=buffer2)

            torch.vdot(out, buffer2, out=PsiHdHpPsi)


            # Data
            if return_data:

                data["energy"][layer].copy_(torch.vdot(out, buffer1).real)

                data["beta"][layer].copy_(beta)

                for name, function in self.metrics.items():
                    data[name][layer] = function(out)


            # Print
            if print_interval and layer % print_interval == 0:

                if return_data:  # evite usar essa opção se estiver rodando na GPU
                    print(f"Layer {layer}, Energy = {data['energy'][layer].item()}")
                else:
                    print(f"Layer {layer}")


            # Feedback
            beta.copy_(PsiHdHpPsi.imag).mul_(2)


        self.manager.release(buffer1)
        self.manager.release(buffer2)


        # CPU transfer
        if return_data:

            if data_to_cpu:
                data = {
                    name: value.cpu().numpy() if torch.is_tensor(value) else value
                    for name, value in data.items()
                }

            return out, data

        return out
