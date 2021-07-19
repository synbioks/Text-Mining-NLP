from models.models_enum import ModelsType
from models.models import BertNERCRFFCN, BertNERCRF
from models.models_fcn import BertNERTopModel, BertNERTopModelFCN


def get_model(model_path, cache_dir, config, model_type: ModelsType = ModelsType.BASELINE, xargs={}):
    if model_type == ModelsType.FCN:
        model = BertNERTopModelFCN.from_pretrained(
            model_path,
            xargs = xargs,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_type == ModelsType.CRF:
        model = BertNERCRF.from_pretrained(
            model_path,
            xargs = xargs,
            config=config,
            cache_dir=cache_dir,
        )
    elif model_type == ModelsType.FCN_CRF:
        model = BertNERCRFFCN.from_pretrained(
            model_path,
            xargs = xargs,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        model = BertNERTopModel.from_pretrained(
            model_path,
            xargs = xargs,
            config=config,
            cache_dir=cache_dir,
        )
    return model
