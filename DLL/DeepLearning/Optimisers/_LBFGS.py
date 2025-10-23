import torch
from collections import deque

from ._BaseOptimiser import BaseOptimiser


class LBFGS(BaseOptimiser):
    """
    Limited-memory Broyden-Fletcher-Goldfarb-Shanno optimiser. A second order method and approximates the hessian matrix using changes in position and gradient. Hence, requires more memory than first order methods.

    Args:
        loss (Callable[[], float]): The target function. For a deep learning model, one could use eg. lambda: model.loss.loss(model.predict(x_train), y_train).
        history_size (int, optional): The number of old changes in position and gradient stored. Must be a non-negative integer. Defaults to 10.
        maxiterls (int, optional): The maximum number of iterations in the line search. Must be a non-negative integer. Defaults to 20.
    """
    def __init__(self, loss, history_size=10, maxiterls=20):
        if not isinstance(history_size, int) or history_size < 0:
            raise ValueError("history_size must be a non-negative integer.")
        if not isinstance(maxiterls, int) or maxiterls < 0:
            raise ValueError("maxiterls must be a non-negative integer.")
        
        self.loss = loss
        self.history_size = history_size
        self.maxiterls = maxiterls
    
    def initialise_parameters(self, model_parameters):
        """
        Initialises the optimiser with the parameters that need to be optimised.

        Args:
            model_parameters (list[torch.Tensor]): The parameters that will be optimised. Must be a list or a tuple of torch tensors.
        """

        if not isinstance(model_parameters, list | tuple):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.s_history = [deque([], maxlen=self.history_size) for _ in model_parameters]
        self.y_history = [deque([], maxlen=self.history_size) for _ in model_parameters]
        self.prevs = [param.clone() for param in model_parameters]
    
    def update_parameters(self):
        """
        Takes a step towards the optimum for each parameter.
        """

        for i, param in enumerate(self.model_parameters):
            if param.grad is None:
                continue

            current = param.clone()
            current.grad = param.grad.clone()
            s = current - self.prevs[i]
            y = current.grad - self.prevs[i].grad if self.prevs[i].grad is not None else torch.zeros_like(current.grad)
            
            if s.flatten() @ y.flatten() > 1e-8:
                self.s_history[i].append(s)  # automatically pops the oldest values as maxlen is spesified for the deque
                self.y_history[i].append(y)  # automatically pops the oldest values as maxlen is spesified for the deque

            direction = self._recursion_two_loops(i)
            learning_rate = self._line_search(param, direction)
            param -= learning_rate * direction.view_as(param)
            
            self.prevs[i] = current
    
    def _recursion_two_loops(self, i):
        q = self.model_parameters[i].grad.clone().flatten()
        alphas = torch.zeros(self.history_size)
        rhos = torch.tensor([1.0 / (self.y_history[i][j].flatten() @ self.s_history[i][j].flatten()) for j in range(self.history_size) if len(self.s_history[i]) > j])

        for j, (s, y) in enumerate(zip(reversed(self.s_history[i]), reversed(self.y_history[i]))):
            j = len(self.s_history[i]) - j - 1
            s, y = s.flatten(), y.flatten()
            alphas[j] = rhos[j] * (s @ q)
            q -= alphas[j] * y
        
        if len(self.s_history[i]) > 0:
            last_s, last_y = self.s_history[i][-1].flatten(), self.y_history[i][-1].flatten()
            Hk_0 = (last_s @ last_y) / last_y @ last_y
        else:
            Hk_0 = 1
        r = Hk_0 * q

        for j, (s, y) in enumerate(zip(self.s_history[i], self.y_history[i])):
            s, y = s.flatten(), y.flatten()
            beta = rhos[j] * (y @ r)
            r += (alphas[j] - beta) * s
        return r
    
    def _loss_param(self, param, new_value):
        param.data = new_value
        loss = self.loss()
        return loss
    
    def _line_search(self, param, direction):
        invphi = 2 / (1 + 5 ** 0.5)

        a, b = 0.0, 10.0
        tol = 1e-5

        orig_param = param.clone()
        direction = direction.view_as(orig_param)

        l = a + (1 - invphi) * (b - a)
        mu = a + invphi * (b - a)

        loss_l = self._loss_param(param, orig_param - l * direction)
        loss_mu = self._loss_param(param, orig_param - mu * direction)

        iter = 0
        while b - a > tol and iter < self.maxiterls:
            if loss_l > loss_mu:
                a = l
                l = mu
                mu = a + invphi * (b - a)
                loss_l = loss_mu
                loss_mu = self._loss_param(param, orig_param - mu * direction)
            else:
                b = mu
                mu = l
                l = a + (1 - invphi) * (b - a)
                loss_mu = loss_l
                loss_l = self._loss_param(param, orig_param - l * direction)
            iter += 1

        step = (a + b) / 2
        param.data = orig_param.data
        return step
