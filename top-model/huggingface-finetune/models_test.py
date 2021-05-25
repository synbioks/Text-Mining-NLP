"""
Similar to models.py. 

"""

import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from model_utils_test import activations_mapper, get_token_classifier_output
  
# different activation functions can be used. see model_utils_test.py
top_model = {"hidden_units_list": [500, 250, 125], 
             "activations_list": ["relu", "relu", "relu", "relu"]
            }

class FullyConnectedLayers(nn.Module):
    """
       hidden_units is a list containing the # units in each hidden layer
       The length of the list denotes the number of hidden layers 
    """
    def __init__(self, hidden_units_list, activations_list, input_embedding_size, num_classes):
        super(FullyConnectedLayers, self).__init__()
        self.layers = nn.ModuleList()
        self.activations_list = activations_list
        
        self.num_units_list = [input_embedding_size, *hidden_units_list, num_classes]

        for i in range(len(self.num_units_list)-1):
            units1 = self.num_units_list[i]
            units2 = self.num_units_list[i+1]
            layer = nn.Linear(units1, units2)
            self.layers.append(layer)
        
    def forward(self, x):
        # for softmax models, it is assumed that the last layer activation would always be 
        # identity since nn.CrossEntropyLoss applies softmax
        for i, activation_str in enumerate(self.activations_list):
            layer = self.layers[i]
            activation = activations_mapper(activation_str)
            x = activation(layer(x))    
        return x
            
        
class BertTopModelE2E(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        
        self.classifier = FullyConnectedLayers(top_model["hidden_units_list"], top_model["activations_list"], 
                                         config.hidden_size, config.num_labels)

        print("Initializing weights")
        self.init_weights()
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return get_token_classifier_output(self, logits, labels, attention_mask, return_dict, outputs)
    