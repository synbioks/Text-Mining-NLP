import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF

from models.model_utils import get_token_classifier_output

actfn_lst = {
    "relu": ["relu"] * 3 + ["identity"],
    "leaky_relu": ["leaky_relu"] * 3 + ["identity"],
    "tanh": ["tanh"] * 3 + ["identity"],
    "softmax": ["softmax"] * 3 + ["identity"],
    "mix": ["relu", "leaky_relu", "tanh", "identity"],
    "id": ["identity"],
}

top_model = {
    "hidden_units_list": [500, 250, 125],
    "activations_list": actfn_lst["leaky_relu"]
}

top_model_crf = {
    "name": "dense_layer_crf",
    "hidden_units_list": [],
    "activations_list": actfn_lst["id"]
}

top_model_crf_1 = {
    "name": "dense_layer_crf",
    "hidden_units_list": [250],
    "activations_list": ["leaky_relu", "identity"]
}

top_model_crf_1_500 = {
    "name": "dense_layer_crf",
    "hidden_units_list": [500],
    "activations_list": ["leaky_relu", "identity"]
}

top_model_fcn_crf = {
    "name": "dense_layer_crf",
    "hidden_units_list": [500, 250, 125],
    "activations_list": actfn_lst["leaky_relu"]
}

activations_mapper = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(),
    "identity": nn.Identity()
}

top_model = {
    0: top_model_crf,
    1: top_model_crf_1,
    3: top_model_fcn_crf,
    1500:top_model_crf_1_500
}

class FullyConnectedLayers(nn.Module):
    """
       hidden_units is a list containing the # units in each hidden layer
       The length of the list denotes the number of hidden layers 
    """

    def __init__(self, hidden_units_list, activations_list, input_embedding_size, num_classes):
        super(FullyConnectedLayers, self).__init__()
        layers = []
        self.activations_list = activations_list

        self.num_units_list = [input_embedding_size,
                               *hidden_units_list, num_classes]

        for i in range(len(self.num_units_list) - 1):
            units1 = self.num_units_list[i]
            units2 = self.num_units_list[i + 1]
            layer = nn.Linear(units1, units2)
            layers.append(layer)
            layers.append(activations_mapper[activations_list[i]])
            if i < len(self.num_units_list) - 2:
                layers.append(nn.LayerNorm(units2))
                layers.append(nn.Dropout(p=0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # for softmax models, it is assumed that the last layer activation would always be
        # identity since nn.CrossEntropyLoss applies softmax
        x = self.layers(x)

        return x


class BertNERCRF(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, xargs):
        super(BertNERCRF,self).__init__(config)
        self.xargs = xargs
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        top_layers = []
        top_layers.append(nn.Dropout(config.hidden_dropout_prob))
        self.top_model = top_model[xargs.get("top_model",0)]
        fcn = FullyConnectedLayers(self.top_model["hidden_units_list"], self.top_model["activations_list"],
                                   config.hidden_size, config.num_labels)
        top_layers.append(fcn)
        self.top_layers = nn.Sequential(*top_layers)

        if self.top_model["name"] == "dense_layer_crf":
            self.crf = CRF(config.num_labels, batch_first=True)

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

        logits = self.top_layers(sequence_output)

        return get_token_classifier_output(self, logits, labels, attention_mask, return_dict, outputs)


class BertNERCRFFCN(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, xargs):
        super(BertNERCRFFCN,self).__init__(config)
        self.num_labels = config.num_labels
        self.xargs = xargs
        self.bert = BertModel(config, add_pooling_layer=False)
        top_layers = []
        top_layers.append(nn.Dropout(config.hidden_dropout_prob))
        self.top_model = top_model[xargs.get("top_model",0)]
        fcn = FullyConnectedLayers(self.top_model["hidden_units_list"], self.top_model["activations_list"],
                                   config.hidden_size, config.num_labels)
        top_layers.append(fcn)
        self.top_layers = nn.Sequential(*top_layers)

        if self.top_model["name"] == "dense_layer_crf":
            self.crf = CRF(config.num_labels, batch_first=True)
        elif self.top_model["name"] == "bilstm_crf":
            pass

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
        
        if self.xargs.get('hmask',False):
            labels_copy = labels.detach().clone()
            labels_copy[labels_copy != -100] = 1
            labels_copy[labels_copy == -100] = 0
            labels_copy = labels_copy.unsqueeze(-1)
            tar_size = sequence_output.shape
            labels_copy.expand(*tar_size)
            sequence_output = sequence_output*labels_copy
            

        logits = self.top_layers(sequence_output)

        return get_token_classifier_output(self, logits, labels, attention_mask, return_dict, outputs)