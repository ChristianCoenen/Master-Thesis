from typing import List, Tuple, Union, Optional
from tensorflow.keras.layers import Layer, Input, Flatten, Dense, LeakyReLU, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from pathlib import Path
from epn.custom_layers import DenseTranspose


class EPNetwork:
    def __init__(self, weight_sharing: bool, encoder_dims: List[int], discriminator_dims: List[int]):
        self.weight_sharing = weight_sharing
        self.encoder_dims = encoder_dims
        self.discriminator_dims = discriminator_dims

    def _build_encoder(
        self,
        input_shape: Union[Tuple[int, ...], int],
        output_layers: List[Layer],
        model_name: Optional[str] = "encoder",
    ) -> Model:
        """This class creates an encoder model with x encoder layers and y output layers."""
        # Define encoder layers
        encoder_layers = []
        for idx, encoder_layer in enumerate(self.encoder_dims):
            encoder_layers.append(Dense(self.encoder_dims[idx], activation=LeakyReLU(alpha=0.2), name=f"encoder_{idx}"))

        # Build encoder model
        encoder_inputs = Input(shape=input_shape, name="encoder_input")
        x = Flatten()(encoder_inputs) if type(input_shape) == tuple else encoder_inputs
        for encoder_layer in encoder_layers:
            x = encoder_layer(x)

        # Create an output layer based on the last encoder layer for each passed layer
        built_output_layers = [output_layer(x) for output_layer in output_layers]

        return Model(encoder_inputs, outputs=built_output_layers, name=model_name)

    def _build_decoder(
        self,
        encoder: Model,
        model_name: Optional[str] = "decoder",
    ) -> Model:
        """This class creates an decoder model that clones the encoder architecture."""
        hidden_encoder_layers = [
            layer for layer in encoder.layers if layer.name not in encoder.output_names and hasattr(layer, "kernel")
        ]
        output_encoder_layers = encoder.layers[-(len(encoder.outputs)) :]

        # Build decoder model
        decoder_inputs = Input(shape=sum([tensor.shape[-1] for tensor in encoder.outputs]), name="decoder_input")
        x = decoder_inputs

        if self.weight_sharing:
            # output_encoder_layers[::-1]: Order seems to matter. Interestingly, inverse order performs better.
            # Might be because of the 'transpose_b=True' in DenseTranspose class, inverse order might be correct
            x = DenseTranspose(
                dense_layers=output_encoder_layers[::-1], activation=LeakyReLU(alpha=0.2), name=f"decoder_0"
            )(decoder_inputs)
            for idx, encoder_layer in enumerate(reversed(hidden_encoder_layers)):
                x = DenseTranspose(
                    dense_layers=[encoder_layer],
                    activation="sigmoid" if idx == len(hidden_encoder_layers) - 1 else LeakyReLU(alpha=0.2),
                    name=f"decoder_{idx + 1}",
                )(x)
        else:
            for idx, encoder_layer in enumerate(reversed(hidden_encoder_layers)):
                x = Dense(
                    units=encoder_layer.input_shape[-1],
                    activation="sigmoid" if idx == len(hidden_encoder_layers) - 1 else LeakyReLU(alpha=0.2),
                    name=f"decoder_{idx}",
                )(x)

        outputs = Reshape(encoder.input_shape[1:])(x) if len(encoder.input_shape[1:]) > 1 else x
        return Model(decoder_inputs, outputs=outputs, name=model_name)

    def build_autoencoder(
        self,
        encoder_input_shape: Union[Tuple[int, ...], int],
        encoder_output_layers: List[Layer],
        ae_ignored_output_layer_names: Optional[List[str]] = None,
        model_name: Optional[str] = "autoencoder",
    ) -> Tuple[Model, Model, Model]:
        # Build autoencoder
        encoder = self._build_encoder(encoder_input_shape, encoder_output_layers)
        encoded = encoder(encoder.input)

        decoder = self._build_decoder(encoder)
        decoded = decoder(concatenate(encoded) if type(encoded) is list else encoded)

        # Ensure that encoded is a list (if encoder has only 1 output, encoded is a Tensor instead of a List of Tensors)
        encoded = [encoded] if type(encoded) is not list else encoded
        # Only use encoder outputs as autoencoder outputs that are not specified in 'ae_ignored_output_layer_names'
        autoencoder_outputs = (
            [output for output in encoded if output.name.split("/")[1] not in ae_ignored_output_layer_names]
            if ae_ignored_output_layer_names
            else encoded
        )
        autoencoder = Model(encoder.input, outputs=[*autoencoder_outputs, decoded], name=model_name)
        return encoder, decoder, autoencoder

    def build_discriminator(
        self,
        input_shape: Union[Tuple[int, ...], int],
        output_layers: List[Layer],
        model_name: Optional[str] = "discriminator",
    ):
        """Creates a discriminator model.

        Leaky ReLU is recommended for Discriminator networks.
        'Within the discriminator we found the leaky rectified activation to work well ...'
            — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015.

        :return: Discriminator model
        """
        inputs = Input(shape=input_shape, name="discriminator_inputs")
        x = Flatten()(inputs) if type(input_shape) == tuple else inputs

        for layer_dim in self.discriminator_dims:
            x = Dense(layer_dim, activation=LeakyReLU(alpha=0.2))(x)
            x = Dropout(0.3)(x)

        # Create an output layer based on the last encoder layer for each passed layer
        built_output_layers = [output_layer(x) for output_layer in output_layers]

        return Model(inputs, outputs=built_output_layers, name=model_name)

    def build_gan(
        self,
        generator: Model,
        discriminator: Model,
        ignored_layer_names: Optional[List[str]] = None,
        model_name: Optional[str] = "gan",
    ) -> Model:
        """Defines a GAN consisting of a generator and a discriminator model. GAN is used to train the generator.

        :return:
        """
        inputs = Input(shape=generator.input_shape[1:], name="gan_inputs")
        generated = generator(inputs)
        # Only use generated outputs as discriminator inputs that are not specified in 'ignored_layer_names'
        discriminator_inputs = (
            [output for output in generated if output.name.split("/")[1] not in ignored_layer_names]
            if ignored_layer_names
            else generated
        )
        discriminated = discriminator(discriminator_inputs)
        return Model(inputs, discriminated, name=model_name)

    def train_autoencoder(self):
        pass

    def train(self):
        pass

    def save_model_architecture_images(self, models: List[Model], path: str):
        """Saves all passed models as PNGs into a defined subfolder.

        :param models:
        :param path: str
            Relative path from the execution directory
        :return:
            None
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        [plot_model(m, f"{path}/{m.name}_architecture.png", show_shapes=True, expand_nested=True) for m in models]
