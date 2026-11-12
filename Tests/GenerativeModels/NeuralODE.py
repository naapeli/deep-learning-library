"""
Neural ordinary differential equations
=========================================

This script learns the dynamics of a damped pendulum from observed trajectories with a neural ordinary
differential equation (neural ODE). The state :math:`x = (\\theta, \\omega)` consists of the angle and the angular
velocity, and the network :math:`f` models its time derivative,

.. math::

    \\frac{dx}{dt} = f(x).

Predictions are made by integrating :math:`f` from an initial state with a numerical solver, and the network is
trained by backpropagating through the steps of the solver. Unlike a network that maps a state directly to the
next one, the learned vector field can be integrated for any length of time and from any initial state.
"""
import torch
import matplotlib.pyplot as plt

from DLL.DeepLearning.Model import Model
from DLL.DeepLearning.Layers import Dense
from DLL.DeepLearning.Layers.Activations import Tanh
from DLL.DeepLearning.Optimisers import ADAM


torch.manual_seed(0)

# %%
# Data
# ----
#
# The true dynamics are those of a damped pendulum, :math:`\\theta' = \\omega` and
# :math:`\\omega' = -\\sin\\theta - 0.3\\omega`. They are only used to generate the data. 40 trajectories from
# random initial states are simulated accurately with the Runge-Kutta method and observed every 0.1 seconds for 3
# seconds.


def true_dynamics(x):
    theta, omega = x[:, 0], x[:, 1]
    return torch.stack([omega, -torch.sin(theta) - 0.3 * omega], dim=1)


def runge_kutta(x, dt, n_steps, substeps=10):
    states = [x]
    h = dt / substeps
    for _ in range(n_steps):
        for _ in range(substeps):
            k1 = true_dynamics(x)
            k2 = true_dynamics(x + h / 2 * k1)
            k3 = true_dynamics(x + h / 2 * k2)
            k4 = true_dynamics(x + h * k3)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(x)
    return torch.stack(states, dim=1)


dt, n_observations, n_trajectories = 0.1, 31, 40
initial_states = torch.stack([4 * torch.rand(n_trajectories) - 2, 3 * torch.rand(n_trajectories) - 1.5], dim=1)
trajectories = runge_kutta(initial_states, dt, n_observations - 1)  # shape (n_trajectories, n_observations, 2)

# %%
# The model and the solver
# ------------------------
#
# The vector field :math:`f` is a small dense network. The solver is the explicit Euler method with two steps
# between consecutive observations, :math:`x_{k + 1} = x_k + hf(x_k)` with :math:`h = 0.05`.

f = Model(2)
f.add(Dense(64, activation=Tanh()))
f.add(Dense(64, activation=Tanh()))
f.add(Dense(2))
f.compile(optimiser=ADAM(learning_rate=0.003))

substeps = 2
h = dt / substeps


def integrate(x, n_steps):
    states = [x]
    for _ in range(n_steps):
        x = x + h * f.predict(x)
        states.append(x)
    return states


# %%
# Training
# --------
#
# Every training step takes a batch of segments of 10 observations from random trajectories, integrates the model
# from the first state of each segment and compares the predicted states with the observed ones with the mean
# squared error.
#
# The gradient is propagated backwards through the solver. If :math:`a_{k + 1}` is the gradient of the loss with
# respect to the state :math:`x_{k + 1}`, the step :math:`x_{k + 1} = x_k + hf(x_k)` gives
# :math:`a_k = a_{k + 1} + hJ_f(x_k)^Ta_{k + 1}` plus the direct gradient of the loss at :math:`x_k`, and the
# parameters of :math:`f` receive the gradient :math:`h(\\partial f(x_k) / \\partial\\theta)^Ta_{k + 1}`. Both are
# computed by :meth:`~DLL.DeepLearning.Model.Model.backward`. A model remembers only its latest forward pass, so
# :math:`f(x_k)` is evaluated again before every backward pass.

segment_length, batch_size, n_steps = 10, 64, 1500
losses = []
f.train()
for step in range(n_steps):
    trajectory = torch.randint(0, n_trajectories, (batch_size,))
    start = torch.randint(0, n_observations - segment_length, (batch_size,))
    targets = torch.stack([trajectories[trajectory, start + i] for i in range(segment_length + 1)], dim=1)

    states = integrate(targets[:, 0], segment_length * substeps)
    observed = torch.stack(states[substeps::substeps], dim=1)
    losses.append(((observed - targets[:, 1:]) ** 2).mean().item())

    f.optimiser.zero_grad()
    n_terms = batch_size * segment_length * 2
    adjoint = torch.zeros(batch_size, 2)
    for k in range(segment_length * substeps, 0, -1):
        if k % substeps == 0:
            adjoint = adjoint + 2 * (states[k] - targets[:, k // substeps]) / n_terms
        f.predict(states[k - 1])
        adjoint = adjoint + f.backward(h * adjoint)
    f.optimiser.update_parameters()
    if (step + 1) % 250 == 0:
        print(f"Step {step + 1}: mean squared error {sum(losses[-250:]) / 250:.5f}")
f.eval()

plt.figure(figsize=(8, 4))
plt.semilogy(losses)
plt.xlabel("Training step")
plt.ylabel("Mean squared error of the segments")
plt.grid(alpha=0.3)
plt.tight_layout()

# %%
# Predicting new trajectories
# ---------------------------
#
# The model was trained on segments of only one second, but it can be integrated for much longer. Here it
# predicts the next 15 seconds from three initial states that were not in the training data.

test_initial_states = torch.tensor([[1.8, 0.0], [-1.0, 1.4], [0.3, -1.2]])
n_test_steps = 150
true_test = runge_kutta(test_initial_states, dt, n_test_steps)
predicted_test = torch.stack(integrate(test_initial_states, n_test_steps * substeps)[::substeps], dim=1)
time = dt * torch.arange(n_test_steps + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, ax in enumerate(axes):
    ax.plot(time, true_test[i, :, 0], color="black", linewidth=3, alpha=0.3, label="true angle")
    ax.plot(time, predicted_test[i, :, 0], color="C0", label="predicted angle")
    ax.plot(time, true_test[i, :, 1], color="black", linewidth=3, alpha=0.15, label="true angular velocity")
    ax.plot(time, predicted_test[i, :, 1], color="C1", linestyle="--", label="predicted angular velocity")
    ax.set_xlabel("Time")
    ax.set_title(f"Initial state ({test_initial_states[i, 0]:.1f}, {test_initial_states[i, 1]:.1f})")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout()

# %%
# The learned vector field
# ------------------------
#
# The vector field of the model next to the true one. Inside the region covered by the training trajectories, the
# model has learned the spiral towards the resting state of the pendulum.

grid = torch.linspace(-2.5, 2.5, 21)
theta, omega = torch.meshgrid(grid, grid, indexing="xy")
points = torch.stack([theta.flatten(), omega.flatten()], dim=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (name, field) in zip(axes, (("True vector field", true_dynamics(points)), ("Learned vector field", f.predict(points)))):
    ax.quiver(theta, omega, field[:, 0].reshape(theta.shape), field[:, 1].reshape(theta.shape), color="C0", alpha=0.6)
    for trajectory in trajectories[:10]:
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="gray", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel(r"Angle $\theta$")
    ax.set_ylabel(r"Angular velocity $\omega$")
axes[0].plot([], [], color="gray", label="training trajectories")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()
