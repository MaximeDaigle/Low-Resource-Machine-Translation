import typing
import datetime
import json

from models.dummy_model import DummyModel
import tensorflow as tf


class ModelFactory():
    """
    This is a model factory: it's responsible of building Models from based on the information that will be provided 
    at evaluation time. From this, the model should be ready to train. 

    Args:
        config: configuration dictionary holding any extra parameters that might be required by the user. These
            parameters are loaded automatically if the user provided a JSON file in their submission. Submitting
            such a JSON file is completely optional, and this argument can be ignored if not needed.
    """

    def __init__(self,
                 config: typing.Dict[typing.AnyStr, typing.Any],
    ):
        self.config = config

        # Declare new models here and map builder function to it.
        self.models = {
            'DummyModel': DummyModel,
        }

    def build(self, modelName):
        return self.models[modelName]()

    def load_model_from_config(self):
        modelName = self.config.get("model_name") or "DummyModel"
        print(f"Loading {modelName}...")
        model = self.models[modelName]()
        return model.load_config(model, self.config)
    

