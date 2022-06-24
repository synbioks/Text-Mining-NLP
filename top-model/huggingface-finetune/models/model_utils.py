import torch
from torch._C import device, dtype
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput

def activations_mapper(activation):
    """Maps the string activation to the corresponding pytorch function"""
    ## Experiment with more activations from here:
    ## https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
    if activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "leaky_relu":
        return F.leaky_relu
    elif activation == "tanh":
        return F.tanh
    # no activation ("None")
    else:
        return nn.Identity()
    
class SelfAdjDiceLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 1.0,wt=None, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = -100
        self.wt = wt

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.detach().clone()
        active_loss = torch.ne(targets,self.ignore_index)
        mask = torch.ones_like(targets)
        mask[~active_loss]*=0
        targets[~active_loss] = 2
        probs = torch.softmax(logits, dim=-1)
        probs = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        wt = self.wt.clone()
        wt = wt.expand(*(logits.shape))
        wt = torch.gather(wt, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        probs_with_factor = ((1 - probs) ** self.alpha) * torch.log(probs)
#         loss = - (2 * probs_with_factor + self.gamma) / (probs_with_factor + 1 + self.gamma)
        loss = -1*wt*probs_with_factor
        loss = loss * mask

        if self.reduction == "mean":
            return loss.sum()/torch.sum(active_loss)
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
    
def get_token_classifier_output(model, logits, labels, attention_mask, return_dict, outputs):
    loss = None
    if labels is not None:
        
        if "softmax" in model.top_model["name"]:
            wt = torch.as_tensor(model.xargs['wt'], dtype=logits.dtype, device=logits.device)
            if "class_wt" in model.xargs.keys():
                loss_fct = nn.CrossEntropyLoss(weight=wt)
            else:
                loss_fct = nn.CrossEntropyLoss()
                
            if model.xargs.get('dce_loss'):
                loss_fct = SelfAdjDiceLoss(model.xargs.get('dce_loss_alpha'),model.xargs.get('dce_loss_gamma'),wt=wt)
            # Only keep active parts of the loss
            if attention_mask is not None:
                if model.xargs.get('random'):
                    attention_mask = attention_mask.detach().clone()
                    attention_mask = attention_mask.view(-1)
                    all_idx =  torch.arange(0,attention_mask.shape[0]).to(attention_mask.device)
                    masked_idx = torch.mask_select(all_idx,labels.view(-1)==2)
                    rand_idx = torch.randint(0,masked_idx.shape[0],
                                             (model.xargs.get('random')*masked_idx.shape[0])//100).to(attention_mask.device)
                    attention_mask[masked_idx[rand_idx]] = loss_fct.ignore_index
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.reshape(-1, model.num_labels)
                active_labels = torch.where(
                    active_loss, labels.reshape(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.reshape(-1, model.num_labels), labels.view(-1))
                
        elif "crf" in model.top_model["name"]:
            labels_copy = labels.detach().clone()
#             if self.xargs.get('skip_subset',False):
#                 labels[:,0] = 2
#                 for i in range(labels.shape[0]):
#                     #TODO
#             labels_copy[labels_copy == -100] = 2
            
            attention_mask_copy = attention_mask.detach().clone()
            if model.xargs.get('skip_subset',False):
                attention_mask_copy[labels_copy == -100]=0
                attention_mask_copy[:,0]=1
            labels_copy[labels_copy == -100] = 2
            if attention_mask is not None:
                loss = -model.crf.forward(logits, labels_copy, attention_mask_copy.type(torch.uint8), reduction="mean")
            else:
                loss = -model.crf.forward(logits, labels_copy)

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )