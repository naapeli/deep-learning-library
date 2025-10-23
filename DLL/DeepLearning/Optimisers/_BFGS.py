import torch

from ._BaseOptimiser import BaseOptimiser


class BFGS(BaseOptimiser):
    """
    Broyden-Fletcher-Goldfarb-Shanno optimiser. A second order method and approximates the hessian matrix using changes in position and gradient. Stores the full inverse Hessian.
    
    Args:
        loss (Callable[[], float]): The target function. For a deep learning model, one could use eg. lambda: model.loss.loss(model.predict(x_train), y_train).
        maxiterls (int, optional): Maximum iterations in line search. Defaults to 20.
    """
    def __init__(self, loss, maxiterls=20):
        if not isinstance(maxiterls, int) or maxiterls < 0:
            raise ValueError("maxiterls must be a non-negative integer.")
        
        self.loss = loss
        self.maxiterls = maxiterls

    def initialise_parameters(self, model_parameters):
        """
        Initialises parameters and the full inverse Hessian approximation.
        """
        if not isinstance(model_parameters, (list, tuple)):
            raise TypeError("model_parameters must be a list or a tuple of torch tensors.")

        self.model_parameters = model_parameters
        self.prevs = [param.clone() for param in model_parameters]
        self.prev_grads = [torch.zeros_like(param) for param in model_parameters]
        self.Hs = [torch.eye(param.numel()) for param in model_parameters]

    def update_parameters(self):
        """
        Update each parameter using BFGS.
        """
        for i, param in enumerate(self.model_parameters):
            if param.grad is None:
                continue

            current = param.clone()
            current_grad = param.grad.clone().flatten()
            s = (current - self.prevs[i]).flatten()
            y = current_grad - self.prev_grads[i].flatten()

            if s @ y > 1e-8:
                rho = 1.0 / (y @ s)
                I = torch.eye(len(s), device=s.device, dtype=s.dtype)
                H = self.Hs[i]
                H = (I - rho * s[:, None] @ y[None, :]) @ H @ (I - rho * y[:, None] @ s[None, :]) + rho * s[:, None] @ s[None, :]
                self.Hs[i] = H

            direction = self.Hs[i] @ current_grad
            step = self._line_search(param, direction)
            param.data -= step * direction.view_as(param)

            self.prevs[i] = current
            self.prev_grads[i] = current_grad.clone()

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
