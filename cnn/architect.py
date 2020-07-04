import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        # 只优化α
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        更新architecture里参数w的过程，对应论文里公式6第一项的 w − ξ * d_w L_{train}(w, α)
    #不直接用外面的optimizer来进行w的更新，而是自己新建一个unrolled_model展开，主要是因为我们这里的更新不能对Network的w进行更新
        """
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            # momentum * v，用的就是architecture里进行w更新的momentum
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        # 前一项是对 theta 求梯度，后一项是 theta 的正则项
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        # 对参数w进行更新，等价于optimizer.step()
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            # 用论文的提出的方法，做 bilevel optimization，即 Second-order Approximation
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            # 不用论文提出的bilevel optimization，只是简单的对α求导，即论文中的 First-order Approximation
            self._backward_step(input_valid, target_valid)
        # 根据反向传播得到的梯度进行参数α的更新
        # 这些参数的梯度是由loss.backward()或者v.grad得到的，optimizer存了这些参数的指针
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """
        计算论文里的公式6，d_α L_{val}(w',α) ，其中 w' = w − ξ * d_w L_{train}(w, α)
        :return:
        """
        # 计算 w'
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        # 计算d_α L_{val}(w',α)，对alpha求梯度
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # 计算d_w' L_{val}(w',α)，对w'求梯度
        vector = [v.grad.data for v in unrolled_model.parameters()]
        # 计算论文里的公式8 (d_α L_{train}(w^+,α) - d_α L_{train}(w^-,α)) / (2 * epsilon)
        # 其中w^+ = w + epsilon * (d_w' L_{val}(w',α)) , w^- = w - epsilon * (d_w' L_{val}(w',α))
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # 对应论文里的公式7 d_α L_{val}(w',α) - epsilon * (d_{\alpha, w} L_{train}(w,a) * d_w' L_{val}(w',α))
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # 对 α 的梯度进行更新
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        计算论文里的公式8 (d_α L_{train}(w^+,α) - d_α L_{train}(w^-,α)) / (2 * epsilon)
        其中w^+ = w + epsilon * (d_w' L_{val}(w',α)) , w^- = w - epsilon * (d_w' L_{val}(w',α))
        """
        # epsilon
        R = r / _concat(vector).norm()

        # w^+
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        # d_α L_{train}(w^+,α)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # w^-
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        # d_α L_{train}(w^-,α)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # 将模型的参数从 w^- 恢复成 w
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
