import typing

import dolfin


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


class HolzapfelOgden:
    def __init__(
        self,
        f0: dolfin.Function,
        n0: dolfin.Function,
        tau: dolfin.Constant = dolfin.Constant(0.0),
        parameters: typing.Optional[typing.Dict[str, dolfin.Constant]] = None,
    ) -> None:

        parameters = parameters or {}
        self.parameters = HolzapfelOgden.default_parameters()
        self.parameters.update(parameters)

        self.f0 = f0
        self.n0 = n0
        self.tau = tau
        print('Tau: ', type(self.tau), self.tau)

    @staticmethod
    def default_parameters() -> typing.Dict[str, dolfin.Constant]:
        return {
            "a": dolfin.Constant(59.0),
            "b": dolfin.Constant(8.023),
            "a_f": dolfin.Constant(18472.0),
            "b_f": dolfin.Constant(16.026),
            "a_n": dolfin.Constant(2481.0),
            "b_n": dolfin.Constant(11.120),
            "a_fn": dolfin.Constant(216.0),
            "b_fn": dolfin.Constant(11.436),
            "kappa": dolfin.Constant(1e6),
            "eta": dolfin.Constant(1e2),
        }

    def W_1(self, I1):
        a = self.parameters["a"]
        b = self.parameters["b"]

        return a / (2.0 * b) * (dolfin.exp(b * (I1 - 3)) - 1.0)

    def W_4(self, I4, direction):
        assert direction in ["f", "n"]
        a = self.parameters[f"a_{direction}"]
        b = self.parameters[f"b_{direction}"]

        return a / (2.0 * b) * heaviside(I4) * (dolfin.exp(b * (I4 - 1) ** 2) - 1.0)

    def W_8(self, I8):
        """
        Cross fiber-sheet contribution.
        """
        a = self.parameters["a_fn"]
        b = self.parameters["b_fn"]

        return a / (2.0 * b) * (dolfin.exp(b * I8**2) - 1.0)

    def W_compress(self, J):
        """
        Compressibility contribution
        """
        return 0.25 * self.parameters["kappa"] * (J**2 - 1 - 2 * dolfin.ln(J))

    def W_visco(self, E_dot):
        """Viscoelastic contributions"""
        return 0.5 * self.parameters["eta"] * dolfin.tr(E_dot * E_dot)

    def Wactive(self, I4f):
        return dolfin.Constant(0.5) * self.tau * (I4f - 1)

    def strain_energy(self, F):
        """
        Strain-energy density function.
        """

        # Invariants
        J = dolfin.det(F)
        C = F.T * F
        I1 = pow(J, -2 / 3) * dolfin.tr(C)
        I4f = dolfin.inner(F * self.f0, F * self.f0)
        I4n = dolfin.inner(F * self.n0, F * self.n0)
        I8fn = dolfin.inner(F * self.f0, F * self.n0)

        # Compressibility
        Wcompress = self.W_compress(J)

        # Active stress
        Wactive = self.Wactive(I4f)

        W1 = self.W_1(I1)
        W4f = self.W_4(I4f, "f")
        W4n = self.W_4(I4n, "n")
        W8fn = self.W_8(I8fn)

    
        W = W1 + W4f + W4n + W8fn + Wcompress + Wactive
        return W
