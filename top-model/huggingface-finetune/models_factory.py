from models.models_enum import ModelsType
from models.models import BertNERTopModel
from models.models_fcn import BertTopModelE2E
from transformers import BertForTokenClassification


def get_model(model_path, cache_dir, config, model_type: ModelsType = ModelsType.BASELINE):
    if model_type == ModelsType.FCN:
        model = BertTopModelE2E.from_pretrained(
            model_path,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_type == ModelsType.CRF:
        model = BertNERTopModel.from_pretrained(
            model_path,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        model = BertForTokenClassification.from_pretrained(
            model_path,
            config=config,
            cache_dir=cache_dir,
        )
    return model
