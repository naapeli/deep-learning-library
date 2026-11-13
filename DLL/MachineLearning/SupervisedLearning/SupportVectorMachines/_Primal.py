from copy import copy

import torch


_SMOOTHING_SCHEDULE = (1.0, 0.1, 0.01, 0.001)


def _objective(beta, K_1, targets, loss, C):
    return 0.5 * beta @ K_1 @ beta + C * loss.loss(K_1 @ beta, targets)


def _fit_primal(K, targets, loss, C, epochs, tol=1e-6):
    """
    Minimises the primal objective of a kernel support vector machine with Newton's method, as in
    `this paper <https://doi.org/10.1162/neco.2007.19.5.1155>`_. The objective is

    .. math::

        J(\\beta) = \\frac{1}{2}\\beta^TK_1\\beta + C\\sum_{i=1}^n l(y_i, f_i),\\quad f = K_1\\beta,

    where :math:`K_1 = K + 1`. Adding 1 to the kernel includes the bias :math:`b = \\sum_i\\beta_i` in the model. The
    gradient of the objective is :math:`K_1(\\beta + Cg)` and the hessian is :math:`K_1(I + C\\text{diag}(h)K_1)`, where
    :math:`g` and :math:`h` are the gradient and the diagonal of the hessian of the loss with respect to :math:`f`. The
    Newton step is therefore the solution of :math:`(I + C\\text{diag}(h)K_1)s = \\beta + Cg`, and the step length is
    halved until the objective decreases.

    The squared losses have a nonzero hessian and are minimised directly. The hinge and the epsilon-insensitive losses
    are piecewise linear with a zero hessian, so their corners are smoothed with a quadratic function, whose width
    (the smoothing) is decreased towards zero. Every smoothed problem is started from the solution of the previous one.

    Args:
        K (torch.Tensor of shape (n_samples, n_samples)): The kernel matrix of the training data.
        targets (torch.Tensor of shape (n_samples,)): The labels or the target values passed to the loss.
        loss: A loss with the attributes squared and smoothing, such as Hinge or EpsilonInsensitive.
        C (float): The regularisation parameter.
        epochs (int): The maximum number of Newton iterations.
        tol (float): The training stops once every element of the Newton step is smaller than tol.

    Returns:
        beta (torch.Tensor of shape (n_samples,)): The coefficients of the kernel expansion.
        bias (torch.Tensor): The bias of the decision function.
        history (torch.Tensor): The primal objective with the original loss after every iteration.
    """
    n = len(targets)
    K_1 = K + 1
    identity = torch.eye(n, dtype=K.dtype)
    beta = torch.zeros(n, dtype=K.dtype)
    history = []
    stages = [loss] if loss.squared else [copy(loss) for _ in _SMOOTHING_SCHEDULE]
    if not loss.squared:
        for stage, smoothing in zip(stages, _SMOOTHING_SCHEDULE):
            stage.smoothing = smoothing

    for stage in stages:
        objective = _objective(beta, K_1, targets, stage, C)
        while len(history) < epochs:
            scores = K_1 @ beta
            gradient = beta + C * stage.gradient(scores, targets)
            hessian = stage.hessian(scores, targets)
            step = torch.linalg.solve(identity + C * hessian.unsqueeze(1) * K_1, gradient)
            # backtracking line search, since a full Newton step can overshoot on a piecewise quadratic objective
            step_length = 1.0
            while True:
                new_beta = beta - step_length * step
                new_objective = _objective(new_beta, K_1, targets, stage, C)
                if new_objective <= objective or step_length < 1e-4:
                    break
                step_length /= 2
            beta, objective = new_beta, new_objective
            history.append(_objective(beta, K_1, targets, loss, C).item())
            if torch.max(torch.abs(step_length * step)) < tol:
                break
    return beta, beta.sum(), torch.tensor(history)
