import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

actfn_lst = {
    "relu": ["relu"] * 3 + ["identity"],
    "leaky_relu": ["leaky_relu"] * 3 + ["identity"],
    "tanh": ["tanh"] * 3 + ["identity"],
    "softmax": ["softmax"] * 3 + ["identity"],
    "mix": ["relu", "leaky_relu", "tanh", "identity"],
}

top_model = {
    "hidden_units_list": [500, 250, 125],
    "activations_list": actfn_lst["leaky_relu"]
}


activations_mapper = {
    "relu": F.relu,
    "elu": F.elu,
    "leaky_relu": F.leaky_relu,
    "tanh": F.tanh,
    "softmax": nn.functional.softmax,
    "identity": nn.Identity()
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

        # [config.hidden_size, 500, 250, 125, 3]
        self.num_units_list = [input_embedding_size, *hidden_units_list, num_classes]

        for i in range(len(self.num_units_list) - 1):
            units1 = self.num_units_list[i]
            units2 = self.num_units_list[i + 1]
            # hidden_size -> 500
            # 500 -> 250 
            # 250 -> 125
            # 125 -> 3
            layer = nn.Linear(units1, units2)
            self.layers.append(layer)

    def forward(self, x):
        # for softmax models, it is assumed that the last layer activation would always be 
        # identity since nn.CrossEntropyLoss applies softmax
        for i, activation_str in enumerate(self.activations_list):
            layer = self.layers[i]
            activation = activations_mapper[activation_str]
            x = activation(layer(x))  # no dropout layer.
        return x


class BertTopModelE2E(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):

        super().__init__(config)

        # config stores the configuration parameters of BertModel
        # config.num_labels -> 3
        # config.hidden_size -> 
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = FullyConnectedLayers(
            top_model["hidden_units_list"],
            top_model["activations_list"],
            config.hidden_size,
            config.num_labels
        )

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

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # return the output as a tuple.
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return the output as an instance of ModelOutput
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
