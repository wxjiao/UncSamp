# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy')
class CrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _, sample_status = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'sample_status': sample_status,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # By xxx: take sentence lprob when eval
        origin_target = model.get_targets(sample, net_output)
        sample_status = None
        if not model.training:
            _lprobs = lprobs  # BxLxV
            _target = origin_target.unsqueeze(-1)  # BxLx1
            _pad_mask = _target.eq(self.padding_idx)  # BxLx1
            # lprob of target tokens
            target_lprob = _lprobs.gather(dim=-1, index=_target)
            mtarget_lprob = target_lprob.masked_fill(_pad_mask, 2.0)
            # generate list
            # Reduce the batchsize if zip error occurs
            sample_lprob_list = mtarget_lprob.squeeze(-1).tolist()
            sample_id_list = sample['id'].squeeze().tolist()
            sample_status = list(zip(sample_id_list, sample_lprob_list))
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = origin_target.view(-1)     # By xxx, different from the 'label_smoothed_cross_entropy'
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss, sample_status

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sample_status = [log.get('sample_status', 0) for log in logging_outputs]    # Add by xxx
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'sample_status': sample_status,     # Add by xxx
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
