"""
Solving a differential equation with a neural network
========================================================

This script solves the equation of a damped harmonic oscillator with a physics-informed neural network (PINN).
Instead of fitting measurements, the network :math:`u(t)` is trained so that it satisfies the differential
equation and the initial conditions. The solution is compared with the exact solution and with an ordinary
network fitted to a few measurements.
"""
import math

import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Optimisers import ADAM
from DLL.DeepLearning.Losses import MSE


torch.manual_seed(0)

# %%
# The equation
# ------------
#
# The position :math:`u(t)` of a damped harmonic oscillator satisfies
#
# .. math::
#
#     u''(t) + 2\delta u'(t) + \omega_0^2 u(t) = 0, \quad u(0) = 1, \quad u'(0) = 0,
#
# where :math:`\delta` is the damping and :math:`\omega_0` the natural frequency. For :math:`\delta < \omega_0`,
# the exact solution is :math:`u(t) = e^{-\delta t}\left(\cos(\omega t) + \frac{\delta}{\omega}\sin(\omega t)\right)`
# with :math:`\omega = \sqrt{\omega_0^2 - \delta^2}`. It is used here only to check the results.

delta, omega0 = 1.5, 10.0
omega = math.sqrt(omega0 ** 2 - delta ** 2)


def exact_solution(t):
    return torch.exp(-delta * t) * (torch.cos(omega * t) + delta / omega * torch.sin(omega * t))


t_plot = torch.linspace(0, 1, 500, dtype=torch.float64)

# %%
# The physics-informed loss
# -------------------------
#
# The network is evaluated at 100 collocation points :math:`t_i` in [0, 1], and the loss penalises the residual
# of the equation and the initial conditions:
#
# .. math::
#
#     L = \frac{1}{N}\sum_{i=1}^N \left(\frac{u''(t_i) + 2\delta u'(t_i) + \omega_0^2 u(t_i)}{\omega_0^2}\right)^2 + (u(0) - 1)^2 + \left(\frac{u'(0)}{\omega_0}\right)^2.
#
# The derivatives of the network are approximated with central finite differences,
# :math:`u'(t) \approx (u(t + h) - u(t - h)) / 2h` and :math:`u''(t) \approx (u(t + h) - 2u(t) + u(t - h)) / h^2`.
# The network is evaluated at all the needed points in one batch, so the residuals are linear combinations of
# the network outputs. The gradient of the loss with respect to every output is therefore easy to write down,
# and :meth:`~DLL.DeepLearning.Model.Model.backward` propagates it to the parameters. The model uses double
# precision to keep the finite differences accurate.

h = 1e-3
n_collocation = 100
t_collocation = torch.linspace(0, 1, n_collocation, dtype=torch.float64)
# all points where the network is evaluated: t - h, t, t + h for the residual and -h, 0, h for the initial conditions
points = torch.cat([t_collocation - h, t_collocation, t_collocation + h, torch.tensor([-h, 0, h], dtype=torch.float64)])

# the residual divided by omega0^2 is a * u(t + h) + b * u(t) + c * u(t - h)
a = (1 / h ** 2 + delta / h) / omega0 ** 2
b = (-2 / h ** 2 + omega0 ** 2) / omega0 ** 2
c = (1 / h ** 2 - delta / h) / omega0 ** 2


def build_network():
    network = Model(1, data_type=torch.float64)
    network.add(Dense(32, activation=Tanh()))
    network.add(Dense(32, activation=Tanh()))
    network.add(Dense(32, activation=Tanh()))
    network.add(Dense(0))
    network.compile(optimiser=ADAM(learning_rate=0.002), loss=MSE())
    return network


pinn = build_network()
n_steps = 8000
losses = []
pinn.train()
for step in range(n_steps):
    pinn.optimiser.zero_grad()
    u = pinn.predict(points.unsqueeze(1))
    u_minus, u_center, u_plus = u[:n_collocation], u[n_collocation:2 * n_collocation], u[2 * n_collocation:3 * n_collocation]
    u0_minus, u0, u0_plus = u[-3], u[-2], u[-1]

    residual = a * u_plus + b * u_center + c * u_minus
    initial_value = u0 - 1
    initial_slope = (u0_plus - u0_minus) / (2 * h) / omega0
    loss = (residual ** 2).mean() + initial_value ** 2 + initial_slope ** 2

    gradient = torch.zeros_like(u)
    gradient[2 * n_collocation:3 * n_collocation] += 2 * residual * a / n_collocation
    gradient[n_collocation:2 * n_collocation] += 2 * residual * b / n_collocation
    gradient[:n_collocation] += 2 * residual * c / n_collocation
    gradient[-2] += 2 * initial_value
    gradient[-1] += 2 * initial_slope / (2 * h * omega0)
    gradient[-3] -= 2 * initial_slope / (2 * h * omega0)
    pinn.backward(gradient)
    pinn.optimiser.update_parameters()

    losses.append(loss.item())
    if (step + 1) % 2000 == 0:
        print(f"Step {step + 1}: physics loss {loss.item():.2e}")
pinn.eval()

# %%
# A network fitted to measurements
# --------------------------------
#
# For comparison, an ordinary network with the same architecture is fitted to 12 noisy measurements from the first
# 40 % of the time interval. It knows nothing about the physics.

t_data = torch.linspace(0, 0.4, 12, dtype=torch.float64)
u_data = exact_solution(t_data) + 0.03 * torch.randn(12, dtype=torch.float64)
data_network = build_network()
data_network.fit(t_data.unsqueeze(1), u_data, epochs=3000)

# %%
# Results
# -------
#
# The physics-informed network follows the exact solution over the whole interval, although it has never seen a
# single value of the solution. The network fitted to the measurements matches them, but has no reason to
# continue the oscillation outside the measured interval. The short spikes in the loss are typical of the Adam
# optimiser with a constant learning rate, and the loss recovers from them quickly.

u_pinn = pinn.predict(t_plot.unsqueeze(1))
u_data_network = data_network.predict(t_plot.unsqueeze(1))
u_exact = exact_solution(t_plot)
print(f"Maximum error of the physics-informed network: {(u_pinn - u_exact).abs().max().item():.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(t_plot, u_exact, color="black", linewidth=3, alpha=0.3, label="exact solution")
ax[0].plot(t_plot, u_pinn, color="C0", label="physics-informed network")
ax[0].plot(t_plot, u_data_network, color="C1", linestyle="--", label="network fitted to the measurements")
ax[0].scatter(t_data, u_data, color="C1", s=20, zorder=3, label="measurements")
ax[0].set_xlabel("t")
ax[0].set_ylabel("u(t)")
ax[0].set_ylim(-1.2, 1.4)
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].semilogy(losses)
ax[1].set_xlabel("Training step")
ax[1].set_ylabel("Physics-informed loss")
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()
