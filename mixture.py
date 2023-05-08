import typing
import math
import dolfin
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x, k=100):
    r"""
    Heaviside function

    .. math::
       \frac{\mathrm{d}}{\mathrm{d}x} \max\{x,0\}

    or

    .. math::
        \frac{1}{1 + e^{-k (x - 1)}}
    """

    # return dolfin.conditional(dolfin.ge(x, 0.0), 1.0, 0.0)
    return 1 / (1 + dolfin.exp(-k * (x - 1)))


class ActiveTension:
    def __init__(self,force_amp=1.0, t_start=50, tau1=20, tau2=110):
        self.force_amp = force_amp
        self.t_start = t_start
        self.tau1 = tau1
        self.tau2 = tau2

    def __call__(self, t):
        force_amp, t_start = self.force_amp, self.t_start
        tau1, tau2 = self.tau1, self.tau2

        beta = -math.pow(tau1/tau2, -1/(1 - tau2/tau1)) + math.pow(tau1/tau2,\
                -1/(-1 + tau1/tau2))
        force = ((force_amp)*(np.exp((t_start - t)/tau1) -\
                np.exp((t_start - t)/tau2))/beta) 
        
        #works for t array or scalar:
        force = force*(t>=t_start) + 0.0*(t<t_start)
        return force
    
    def plot(self,show=True):
        t = np.linspace(0,1000,1001)
        f = self(t)
        plt.plot(t,f)
        plt.xlabel('Time (ms)')
        plt.ylabel('Force (normalized)')
        plt.title('Reference active tension')
        if show:
            plt.show()




class Constituent:
    pass


class Fiber(Constituent):
    def __init__(
        self,
        f0: dolfin.Function,
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = Fiber.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        
    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "a_f": dolfin.Constant(18472.0),
            "b_f": dolfin.Constant(16.026),
        }


    def W4(self, I4):
        
        a = self.parameters["a_f"]
        b = self.parameters["b_f"]

        return a / (2.0 * b) * heaviside(I4) * (dolfin.exp(b * (I4 - 1) ** 2) - 1.0)

    def W(self,F):
        I4 = dolfin.inner(F * self.f0, F * self.f0)
        return self.W4(I4) 


class ActiveFiber(Fiber):
    def __init__(self,
        f0: dolfin.Function,
        active: callable,
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = ActiveFiber.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        self.active = active
        self.tau = dolfin.Constant(0.0) #maybe not needed

    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "a_f": dolfin.Constant(18472.0),
            "b_f": dolfin.Constant(16.026),
            "T_ref": dolfin.Constant(1.0),
        }
    
    def update_active(self,time):
        #maybe not needed, could make tau a local variable in Wactive and update there?
        T = self.active(time)
        self.tau.assign(T)


    def Wactive(self,I4):
        T_ref = self.parameters['T_ref']
        return T_ref * self.tau * (I4 - 1)
    

    def W(self,F):
        I4 = dolfin.inner(F * self.f0, F * self.f0)
        return self.W4(I4) + self.Wactive(I4)


class NeoHookeanBackground(Constituent):
    def __init__(
        self,
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = NeoHookeanBackground.default_parameters()
        self.parameters.update(parameters)
        
    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "c1": dolfin.Constant(72.0),
            "kappa": dolfin.Constant(1e6),
        }


    def W1(self, I1):
        c1 = self.parameters["c1"]
        return c1*(I1-3) 

    def W_compress(self,J):
        return 0.25 * self.parameters["kappa"] * (J**2 - 1 - 2 * dolfin.ln(J))

    def W(self, F):
        J = dolfin.det(F)
        C = F.T * F
        I1 = pow(J, -2 / 3) * dolfin.tr(C)
        return self.W1(I1) + self.W_compress(J)
    


class FungBackground(Constituent):
    def __init__(
        self,
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = FungBackground.default_parameters()
        self.parameters.update(parameters)
        
    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "a": dolfin.Constant(59.0),
            "b": dolfin.Constant(8.023),
            "kappa": dolfin.Constant(1e6),
        }


    def W1(self, I1):
        a = self.parameters["a"]
        b = self.parameters["b"]
        return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0)

    def W_compress(self,J):
        return 0.25 * self.parameters["kappa"] * (J**2 - 1 - 2 * dolfin.ln(J))

    def W(self, F):
        J = dolfin.det(F)
        C = F.T * F
        I1 = pow(J, -2 / 3) * dolfin.tr(C)
        return self.W1(I1) + self.W_compress(J)
    

class Mixture:
    def __init__(
        self,
        constituents: list[Constituent],
        mass_fractions: list[float],
        rho_0: float = 1.05   #10^3 kg/m^3
    ) -> None:
        self.constituents = constituents
        self.mass_fractions = mass_fractions
        self.rho_0 = rho_0
    
    def strain_energy(self, F):
        """
        Strain-energy density function.
        """

        W = dolfin.Constant(0.0)
        
        for frac, const in zip(self.mass_fractions, self.constituents):
            W += frac * const.W(F)

        return self.rho_0*W
        
        