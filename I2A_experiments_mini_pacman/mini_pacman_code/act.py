import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class ACTFromCell(nn.Module):
    def __init__(self, rnn_cell, max_ponder=100, epsilon=0.01,
                 batch_first=False):
        super(ACTFromCell, self).__init__()
        self.rnn_cell = rnn_cell
        self.batch_first = batch_first
        self.max_ponder = max_ponder
        self.epsilon = epsilon

        self._is_lstm = isinstance(self.rnn_cell, nn.LSTMCell)
        self.ponder_linear = nn.Linear(self.rnn_cell.hidden_size, 1)

    def forward(self, input_, hx=None, compute_ponder_cost=True):
        if self.batch_first:
            # Move batch to second
            input_ = input_.transpose(0, 1)
        if hx is None:
            hx = Variable(input_.data.new(
                input_.size(1), self.rnn_cell.hidden_size
            ).zero_())
            if self._is_lstm:
                cx = hx

        # Pre-allocate variables
        time_size, batch_size, input_dim_size = input_.size()
        selector = input_.data.new(batch_size).byte()
        hx_list, cx_list = [], []
        ponder_cost = 0
        ponder_times = []

        # For each t
        for input_row in input_:

            accum_h = Variable(input_.data.new(batch_size).zero_())
            accum_hx = Variable(input_.data.new(
                input_.size(1), self.rnn_cell.hidden_size
            ).zero_())
            if self._is_lstm:
                accum_cx = Variable(input_.data.new(
                    input_.size(1), self.rnn_cell.hidden_size
                ).zero_())
            selector = selector.fill_(1)

            if self._is_lstm:
                accum_cx = accum_cx.zero_()
            step_count = Variable(input_.data.new(batch_size).zero_())
            input_row_with_flag = torch.cat([
                input_row,
                Variable(input_row.data.new(batch_size, 1).zero_())
            ], dim=1)
            if compute_ponder_cost:
                step_ponder_cost = Variable(input_.data.new(batch_size).zero_())

            for act_step in range(self.max_ponder):
                idx = bool_to_idx(selector)
                if compute_ponder_cost:
                    # Weird but matches formulation
                    step_ponder_cost[idx] = -accum_h[idx]

                if self._is_lstm:
                    hx[idx], cx[idx] = self.rnn_cell(
                        input_row_with_flag[idx], (hx[idx], cx[idx]))
                else:
                    hx[idx] = self.rnn_cell(input_row_with_flag[idx], hx[idx])
                accum_hx[idx] += hx[idx]
                h = F.sigmoid(self.ponder_linear(hx[idx]).squeeze(1))
                accum_h[idx] += h
                p = h - (accum_h[idx] - 1).clamp(min=0)
                accum_hx[idx] += p.unsqueeze(1) * hx[idx]
                if self._is_lstm:
                    accum_cx[idx] += p.unsqueeze(1) * cx[idx]
                step_count[idx] += 1
                selector = (accum_h < 1 - self.epsilon).data
                if not selector.any():
                    break
                input_row_with_flag[:, input_dim_size] = 1

            ponder_times.append(step_count.data.cpu().numpy())
            if compute_ponder_cost:
                ponder_cost += step_ponder_cost

            hx = accum_hx / step_count.clone().unsqueeze(1)
            hx_list.append(hx)

            if self._is_lstm:
                cx = accum_cx / step_count.clone().unsqueeze(1)
                cx_list.append(cx)

        if self._is_lstm:
            all_hx = [
                torch.stack(hx_list),
                torch.stack(cx_list),
            ]
            hx, cx = hx.unsqueeze(0), cx.unsqueeze(1)
        else:
            all_hx = torch.stack(hx_list)
            hx = hx.unsqueeze(0)

        if self.batch_first:
            # Move batch to first
            if self._is_lstm:
                all_hx = all_hx[0].transpose(0, 1)
            else:
                all_hx = all_hx.transpose(0, 1)

        return all_hx, hx, {
            "ponder_cost": ponder_cost,
            "ponder_times": ponder_times,
        }

    def reset_parameters(self):
        self.rnn_cell.reset_parameters()
        self.ponder_linear.reset_parameters()
        self.ponder_linear.bias.data.fill_(1)


def bool_to_idx(idx):
    return idx.nonzero().squeeze(1)


def resolve_model(config):
    if config.use_act:
        model_class = ParityACTModel
    else:
        model_class = ParityRNNModel

    model = model_class(config)
    if config.cuda:
        model = model.cuda()
    return model
