import tensorflow as tf
from tensorflow.keras import layers, models


class CNNModel(models.Sequential):
    def __init__(self, input_shape, num_classes, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        self.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation="relu"))
        self.add(layers.Dense(num_classes, activation="softmax"))

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop("input_shape")
        num_classes = config.pop("num_classes")
        # Remove extra keys that may interfere
        config.pop("layers", None)
        return cls(input_shape=input_shape, num_classes=num_classes, **config)

    def get_config(self):
        config = super(CNNModel, self).get_config()
        config.update(
            {
                "input_shape": self.layers[0].input_shape[1:],
                "num_classes": self.layers[-1].units,
            }
        )
        return config
