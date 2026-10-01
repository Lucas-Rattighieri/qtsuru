import torch

from ...hamiltonians import Hamiltonian
from ...buffermanager import BufferManager


class SOFALQON:

    def __init__(self,
        L: int,
        Hp: Hamiltonian,
        Hd: Hamiltonian,
        psi0: torch.Tensor,
        Btol: float = 1e-8,
        hybrid: bool=True,
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


        if Btol <= 0:
            raise ValueError("Btol should be positive.")
        self.Btol = Btol

        self.hybrid = hybrid

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
        buffer3 = self.manager.get()

        device = out.device
        dtype = out.real.dtype


        beta = torch.as_tensor(beta_0, device=device, dtype=dtype)
        beta_fo = torch.empty((), device=device, dtype=dtype)
        beta_so = torch.empty((), device=device, dtype=dtype)

        B = torch.empty((), device=device, dtype=dtype)
        C = torch.empty((), device=device, dtype=dtype)

        driver_dt = torch.empty((), device=device, dtype=dtype)

        scalar1 = torch.empty((), device=device, dtype=out.dtype)
        scalar2 = torch.empty((), device=device, dtype=out.dtype)


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

            data["beta"][0].copy_(beta)


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
            driver_dt.copy_(beta).mul_(delta_t)
            self.Hd.evolution(buffer1, driver_dt, out=out)

            # Hp |psi>
            self.Hp.hamiltonian(out, out=buffer1)

            # Hd |psi>
            self.Hd.hamiltonian(out, out=buffer2)

            # Hd Hp |psi>
            self.Hd.hamiltonian(buffer1, out=buffer3)


            # First-order feedback
            torch.vdot(out, buffer3, out=scalar1)

            beta_fo.copy_(scalar1.imag).mul_(2)

            # <Hd psi | Hd Hp psi>
            torch.vdot(buffer2, buffer3, out=scalar1)

            # <Hp psi | Hd Hp psi>
            torch.vdot(buffer1, buffer3, out=scalar2)

            # Hp Hd |psi>
            self.Hp.hamiltonian(buffer2, out=buffer3)

            # B = <Hd psi | Hp | Hd psi> - Re <Hd psi | Hd Hp psi>
            B.copy_(torch.vdot(buffer2, buffer3).real).sub_(scalar1.real).abs_()
            B.clamp_min_(self.Btol)

            # C = 2 Re[<Hp psi | Hp Hd psi> - <Hp psi | Hd Hp psi>]
            C.copy_(torch.vdot(buffer1, buffer3).real).sub_(scalar2.real).mul_(2)



            # beta_so = (beta_FO - dt C) / (2 dt B)
            beta_so.copy_(C).mul_(-delta_t)
            beta_so.add_(beta_fo)
            beta_so.div_(B).div_(2 * delta_t)


            # Data
            if return_data:

                data["energy"][layer].copy_(torch.vdot(out, buffer1).real)

                data["beta"][layer].copy_(beta)

                for name, function in self.metrics.items():
                    data[name][layer] = function(out)


            # Print
            if print_interval and layer % print_interval == 0:

                if return_data:
                    print(f"Layer {layer}, Energy = {data['energy'][layer].item()}")
                else:
                    print(f"Layer {layer}")

            
            # Hybrid feedback
            if self.hybrid:
                torch.where(torch.isfinite(beta_so) & (beta_so.abs() < beta_fo.abs()), beta_so, beta_fo, out=beta)
            else:
                beta.copy_(beta_so)


        self.manager.release(buffer1)
        self.manager.release(buffer2)
        self.manager.release(buffer3)


        # CPU transfer
        if return_data:

            if data_to_cpu:
                data = {
                    name: value.cpu().numpy() if torch.is_tensor(value) else value
                    for name, value in data.items()
                }

            return out, data

        return out
