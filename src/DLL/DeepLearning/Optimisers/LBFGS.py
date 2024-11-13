import torch
from collections import deque

from .BaseOptimiser import BaseOptimiser


class LBFGS(BaseOptimiser):
    def __init__(self, loss, history_size=10):
        self.loss = loss
        self.history_size = history_size
    
    def initialise_parameters(self, model_parameters):
        self.model_parameters = model_parameters
        self.s_history = [deque([], maxlen=self.history_size) for _ in model_parameters]
        self.y_history = [deque([], maxlen=self.history_size) for _ in model_parameters]
        self.prevs = [param.clone() for param in model_parameters]
        self.prev_Bs = [torch.ones_like(param.flatten()) for param in model_parameters]
    
    def update_parameters(self):
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

            Bs = self._recursion_two_loops(i)

            learning_rate = self._line_search(param, Bs)
            param -= learning_rate * Bs.view_as(param)
            
            self.prevs[i] = current
            self.prev_Bs[i] = Bs
    
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
    
    def _line_search(self, param, direction):
        step = 1.0
        initial_func_value = self.loss()
        grad_dot_direction = torch.sum(param.grad.flatten() * direction)
        orig_param = param.clone()
        c = 1e-4
        
        while True:
            new_param_value = orig_param - step * direction.view_as(orig_param)
            param.data = new_param_value
            new_func_value = self.loss()
            
            # Does not include the other wolfe condition as it requires us to calculate the gradient of the parameter
            # at the new point, which is only possible to calculate using finite differences with current architecture.
            if new_func_value <= initial_func_value + c * step * grad_dot_direction:
                break
            else:
                step *= 0.5
        
        param.data = orig_param.data
        return step
