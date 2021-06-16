import torch
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
    
    
def get_token_classifier_output(model, logits, labels, attention_mask, return_dict, outputs):
    loss = None
    if labels is not None:
        
        if "softmax" in model.top_model["name"]:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
                
        elif "crf" in model.top_model["name"]:
            labels_copy = labels.detach().clone()
            labels_copy[labels_copy == -100] = 2
            if attention_mask is not None:
                loss = -model.crf.forward(logits, labels_copy, attention_mask.type(torch.uint8), reduction="mean")
                decoded_seq = model.crf.decode(logits, attention_mask.type(torch.uint8))
            else:
                loss = -model.crf.forward(logits, labels_copy)
                decoded_seq = model.crf.decode(logits)
            
            #  compute new logits for CRF as per final sequence
            tag_indices = torch.tensor(decoded_seq).unsqueeze(-1)
            src_matrix = torch.ones_like(logits)
            logits = torch.zeros_like(logits)
            logits.scatter_(-1,tag_indices,src_matrix)

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )