import logging

from data import download_data

import pedl

model_type = pedl.get_data_config().get("model_type")

if model_type == "single_output":
    from model_def import MNistTrial
    from data import make_data_loaders
elif model_type == "multi_output":
    from model_def_multi_output import MultiMNistTrial
    from data import make_multi_data_loaders as make_data_loaders
else:
    logging.critical(f"Unrecognized model_type: '{model_type}'")
