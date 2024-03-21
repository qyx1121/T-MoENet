
### copy from LIMoE


#from distutils.command.config import config
import os
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np

from transformers.activations import ACT2FN
from .adapter import Adapter
from collections import OrderedDict
from copy import deepcopy

#-------------------#
# MoE

class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = ACT2FN["gelu"]
        self.log_soft = nn.LogSoftmax(1)
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.log_soft(out)
        return out



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # torch.nonzero:
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        
        #stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if len(stitched.shape) == 3:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            else:
                stitched = stitched.mul(self._nonzero_gates)

        if len(stitched.shape) == 3:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        else:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        
        #combined[combined == 0] = np.finfo(float).eps
        # back to log space
        #return combined.log()
        return combined


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, 
                 noisy_gating = True, 
                 ds_factor = 8.0,
                 num_experts = 4,
                 moe_input_size = 768,
                 top_k = 2,
                 dropout = 0.1,
                 gating = 'linear',
                 routing = None
                 ):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = moe_input_size
        self.k = top_k
        # instantiate experts
        self.gating = gating
        self.experts = nn.ModuleList([Adapter(ds_factor, moe_input_size, dropout=dropout) for i in range(self.num_experts)])
        self.routing = routing
        self.infer_expert = None


        if self.routing != 'random':
            if gating == 'linear':
                self.w_gate = nn.Parameter(torch.zeros(self.input_size, num_experts), requires_grad=True)
            elif gating == 'cosine':
                self.w_gate = CosineTopKGate(self.input_size, self.num_experts)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

            self.softplus = nn.Softplus()
            self.softmax = nn.Softmax(-1)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.k # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    
    def random_k_gating(self, features, train):
        if train:
            idx = torch.randint(0, self.num_experts, 1)
            results = self.experts[idx](features)

        else:
            results = []
            for i in range(self.num_experts):
                tmp = self.num_experts[i](features)
                results.append(tmp)
            
            results = torch.stack(results, dim=0).mean(dim=0)

        return results



    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        #clean_logits = self.w_gate(x)
        if self.gating == 'linear':
            clean_logits = x @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim= -1)  
        
        top_k_logits = top_logits[:, :self.k] if len(top_logits.shape) == 2 else top_logits[:, :, :self.k]    
        top_k_indices = top_indices[:, :self.k] if len(top_indices.shape) == 2 else top_indices[:, :, :self.k]
        
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
       
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, frame_features, train=True, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """

        if self.routing is 'random':
            loss = None
            load = None
            if train:
                gates = torch.zeros(x.shape[0], self.num_experts)
                random_idx = torch.randint(0, self.num_experts, (x.shape[0],))
                gates[torch.arange(x.shape[0]), random_idx] = 1
                gates = gates.to(x.device)
                dispatcher = SparseDispatcher(self.num_experts, gates)
            
                expert_inputs = dispatcher.dispatch(frame_features) 
                gates = dispatcher.expert_to_gates() 
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
                y = dispatcher.combine(expert_outputs)
            else:
                if self.infer_expert is None:
                    weights = [self.experts[i].state_dict() for i in range(self.num_experts)]
                    merge_weights = OrderedDict()
                    for idx, it in enumerate(weights):
                        for k,v in it.items():
                            merge_weights[k] = v / self.num_experts if idx==0 else merge_weights[k] + v / self.num_experts
                
                    self.infer_expert = deepcopy(self.experts[0])
                    self.infer_expert.load_state_dict(merge_weights)
                
                y = self.infer_expert(frame_features)
            
            return y, loss, load 
        
        else:
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            gates, load = self.noisy_top_k_gating(x, train)
            # calculate importance loss
            importance = gates.sum(dim=0)
            
            # calculate loss
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            dispatcher = SparseDispatcher(self.num_experts, gates)
            
            expert_inputs = dispatcher.dispatch(frame_features) 
            gates = dispatcher.expert_to_gates() 
            expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
            y = dispatcher.combine(expert_outputs)
            return y, loss, load

class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, proj_dim=256, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits

