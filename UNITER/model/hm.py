"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for HM model
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import UniterPreTrainedModel, UniterModel
from .attention import MultiheadAttention


class UniterForHm(UniterPreTrainedModel):
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.uniter = UniterModel(config, img_dim)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.dense(pooled_output)
        logits = self.dropout(logits)
        # logits = self.classifier(logits).squeeze(1)
        logits = self.classifier(logits)

        if compute_loss:
            # hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            hm_loss = F.cross_entropy(logits, targets, reduction='none')
            return hm_loss
        else:
            return logits


class UniterForHmPaired(UniterPreTrainedModel):
    """ Finetune UNITER for HM (paired format)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.hm_output = nn.Linear(config.hidden_size*2, 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        # concat CLS of the pair
        n_pair = pooled_output.size(0) // 2
        reshaped_output = pooled_output.contiguous().view(n_pair, -1)
        logits = self.hm_output(reshaped_output).squeeze(1)

        if compute_loss:
            hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            return hm_loss
        else:
            return logits


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


class UniterForHmPairedAttn(UniterPreTrainedModel):
    """ Finetune UNITER for HM
        (paired format with additional attention layer)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.attn1 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.attn2 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(2*config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))
        self.attn_pool = AttentionPool(config.hidden_size,
                                       config.attention_probs_dropout_prob)
        # self.hm_output = nn.Linear(2*config.hidden_size, 1)
        self.hm_output = nn.Linear(2 * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        # separate left image and right image
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs//2, tl*2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs//2, tl*2
                                                       ).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out,
                                 key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out,
                                 key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                           ).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                            ).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        # logits = self.hm_output(torch.cat([left_out, right_out], dim=-1)).squeeze(1)
        logits = self.hm_output(torch.cat([left_out, right_out], dim=-1))

        if compute_loss:
            # hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            hm_loss = F.cross_entropy(logits, targets, reduction='none')
            return hm_loss
        else:
            return logits
