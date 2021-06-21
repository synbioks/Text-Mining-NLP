from .models.models_enum import ModelsType
from .models.models import BertNERTopModel
from .models.models_fcn import BertTopModelE2E
from transformers import BertForTokenClassification


def get_model(model_args, config, model_type: ModelsType = ModelsType.BASELINE):
    if model_type == ModelsType.FCN:
        model = BertTopModelE2E.from_pretrained(
            model_args['model_name_or_path'],
            config=config,
            cache_dir=model_args['cache_dir'],
        )
    elif model_type == ModelsType.CRF:
        model = BertNERTopModel.from_pretrained(
            model_args['model_name_or_path'],
            config=config,
            cache_dir=model_args['cache_dir'],
        )
    else:
        model = BertForTokenClassification.from_pretrained(
            model_args['model_name_or_path'],
            config=config,
            cache_dir=model_args['cache_dir']
        )
    return model
