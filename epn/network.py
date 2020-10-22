import abc
import tensorflow as tf
from typing import List, Tuple, Optional
from tensorflow.keras.layers import Layer, Input, Flatten, Dense, LeakyReLU, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow import Tensor
from pathlib import Path
from epn.custom_layers import DenseTranspose


class EPNetwork:
    """ The entropy propagation base class. It defines the base architecture and is highly customizable. """

    def __init__(self, weight_sharing: bool, encoder_dims: List[int], discriminator_dims: List[int]):
        """
        :param weight_sharing:
            If set to true, the decoder will used the weights created on the encoder side using DenseTranspose layers
        :param encoder_dims:
            Each value (x) represents one hidden encoder layer with x neurons.
        :param discriminator_dims:
            Each value (x) represents one hidden layer with x neurons. By default, the discriminator network will
            mimic the structure of the hidden encoder layers (since the generator has the same structure as the encoder,
            the discriminator will and decoder are of the same size which is mostly good in an adversarial setting).
        """
        self.weight_sharing = weight_sharing
        self.encoder_dims = encoder_dims
        self.discriminator_dims = discriminator_dims

    def build_encoder(
        self,
        input_tensors: List[Tensor],
        output_layers: List[Layer],
        model_name: str = "encoder",
    ) -> Model:
        """
        This class creates an encoder model with x encoder layers and y output layers.

        :param input_tensors
            A list of keras tensors that are attached as input tensors to the encoder.
        :param output_layers
            Takes a list of valid keras layers and attaches them as output layers to the encoder.
        :param model_name
            The name of the model (useful for plots)

        :returns: A model object.

        """
        # Define encoder layers
        encoder_layers = []
        for idx, encoder_layer in enumerate(self.encoder_dims):
            encoder_layers.append(Dense(self.encoder_dims[idx], activation=LeakyReLU(alpha=0.2), name=f"encoder_{idx}"))

        """ Build encoder model """
        # Concatenate all input layers (if more than 1) and Flatten multidimensional inputs beforehand
        flattened_input_tensors = []
        for input_tensor in input_tensors:
            flattened_input_tensors.append(Flatten()(input_tensor) if len(input_tensor.shape[1:]) > 1 else input_tensor)
        encoder_inputs = flattened_input_tensors[0] if len(input_tensors) < 2 else concatenate(flattened_input_tensors)

        x = encoder_inputs
        for encoder_layer in encoder_layers:
            x = encoder_layer(x)

        # Create an output layer based on the last encoder layer for each passed layer
        built_output_layers = [output_layer(x) for output_layer in output_layers]

        return Model(input_tensors, outputs=built_output_layers, name=model_name)

    def build_decoder(
        self,
        encoder: Model,
        model_name: str = "decoder",
    ) -> Model:
        """This class creates an decoder model that clones the encoder architecture.

        :param encoder
            A model object that represents the encoder.
        :param model_name
            The name of the model (useful for plots)

        :returns: A model object.

        """
        hidden_encoder_layers = [
            layer for layer in encoder.layers if layer.name not in encoder.output_names and hasattr(layer, "kernel")
        ]
        output_encoder_layers = encoder.layers[-(len(encoder.outputs)) :]

        # Build decoder model
        decoder_inputs = [Input(shape=tensor.shape[-1], name=tensor.name.split("/")[0]) for tensor in encoder.outputs]
        x = decoder_inputs[0] if len(decoder_inputs) < 2 else concatenate(decoder_inputs)

        # Hidden layers
        if self.weight_sharing:
            x = DenseTranspose(dense_layers=output_encoder_layers, activation=LeakyReLU(alpha=0.2), name=f"decoder_0")(
                x
            )
            for idx, encoder_layer in enumerate(reversed(hidden_encoder_layers[1:])):
                x = DenseTranspose(
                    dense_layers=[encoder_layer],
                    activation=LeakyReLU(alpha=0.2),
                    name=f"decoder_{idx + 1}",
                )(x)
        else:
            for idx, encoder_layer in enumerate(reversed(hidden_encoder_layers[1:])):
                x = Dense(
                    units=encoder_layer.input_shape[-1],
                    activation=LeakyReLU(alpha=0.2),
                    name=f"decoder_{idx}",
                )(x)

        # Output layers
        # TODO: this part is not optimal and is a bit messy / hard to understand, but I don't see another solution.
        # What's happening here is that decoder outputs are basically the same as encoder inputs, so we iterate through
        # the encoder inputs and create decoder layers. Because encoder inputs are concatenated and there is no direct
        # split in tensorflow, the weights are sliced to ensure that weight sharing is still working
        outputs = []
        encoder_layer = hidden_encoder_layers[0]
        counter = 0
        for input_tensor in encoder.inputs:
            neurons = Flatten()(input_tensor).shape[-1]
            if self.weight_sharing:
                output = DenseTranspose(
                    dense_layers=[encoder_layer],
                    activation="sigmoid",
                    custom_weights=tf.Variable(encoder_layer.weights[0][counter : counter + neurons]),
                    name=f"{input_tensor.name.split(':')[0]}",
                )(x)
            else:
                output = Dense(
                    units=Flatten()(input_tensor).shape[-1],
                    activation="sigmoid",
                    name=f"{input_tensor.name.split(':')[0]}",
                )(x)
            outputs.append(Reshape(input_tensor.shape[1:])(output) if len(input_tensor.shape[1:]) > 1 else output)
            counter += neurons

        return Model(decoder_inputs, outputs=outputs, name=model_name)

    def build_autoencoder(
        self,
        encoder_input_tensors: List[Tensor],
        encoder_output_layers: List[Layer],
        ae_ignored_output_layer_names: Optional[List[str]] = None,
        model_name: str = "autoencoder",
    ) -> Tuple[Model, Model, Model]:
        """
        Creates an autoencoder by calling the build_encoder & build_decoder method and concatenating the returned models

        :param encoder_input_tensors
            A list of keras tensors that are attached as input tensors to the encoder.
        :param encoder_output_layers
            A list of valid keras layers that are attached as output layers to the encoder.
        :param ae_ignored_output_layer_names
            A list of layer names that are ignored as autoencoder outputs. If empty, the decoder will use all output
            layers of the encoder + the decoder's output, which is always used.
        :param model_name
            The name of the model (useful for plots)

        :returns: Three model objects (encoder, decoder, autoencoder).

        """
        # Build autoencoder
        encoder = self.build_encoder(encoder_input_tensors, encoder_output_layers)
        autoencoder_inputs = [
            Input(shape=tensor.shape[1:], name=tensor.name.split(":")[0]) for tensor in encoder.inputs
        ]
        encoded = encoder(autoencoder_inputs)
        decoder = self.build_decoder(encoder)
        decoded = decoder(encoded)
        # Ensure that encoded is a list (if encoder has only 1 output, encoded is a Tensor instead of a List of Tensors)
        encoded = [encoded] if type(encoded) is not list else encoded
        # Only use encoder outputs as autoencoder outputs that are not specified in 'ae_ignored_output_layer_names'
        autoencoder_outputs = (
            [output for output in encoded if output.name.split("/")[1] not in ae_ignored_output_layer_names]
            if ae_ignored_output_layer_names
            else encoded
        )
        autoencoder = Model(autoencoder_inputs, outputs=[*autoencoder_outputs, decoded], name=model_name)
        return encoder, decoder, autoencoder

    def build_discriminator(
        self,
        input_tensors: List[Tensor],
        output_layers: List[Layer],
        model_name: str = "discriminator",
        use_dropout_layers: bool = True,
    ):
        """
        Creates a discriminator model.

        Leaky ReLU is recommended for Discriminator networks.
        'Within the discriminator we found the leaky rectified activation to work well ...'
            — Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2015.

        Additionally, each layer is followed by a dropout layer of strength 0.3. This is a commonly used addition for
        discriminator networks to prevent them from outperforming the generator during training.

        :param input_tensors
            A list of keras tensors that are attached as input tensors to the discriminator.
        :param output_layers
            Takes a list of valid keras layers and attaches them as output layers to the discriminator.
        :param model_name
            The name of the model (useful for plots)
        :param use_dropout_layers
            Whether dropout layers are added after each layer in the discriminator.

        :return: A model object.
        """

        """ Build discriminator model """
        # Concatenate all input layers (if more than 1) and Flatten multidimensional inputs beforehand
        flattened_input_tensors = []
        for input_tensor in input_tensors:
            flattened_input_tensors.append(Flatten()(input_tensor) if len(input_tensor.shape[1:]) > 1 else input_tensor)

        discriminator_inputs = (
            flattened_input_tensors[0] if len(input_tensors) < 2 else concatenate(flattened_input_tensors)
        )
        x = discriminator_inputs

        for layer_dim in self.discriminator_dims:
            x = Dense(layer_dim, activation=LeakyReLU(alpha=0.2))(x)
            if use_dropout_layers:
                x = Dropout(0.3)(x)

        # Create an output layer based on the last encoder layer for each passed layer
        built_output_layers = [output_layer(x) for output_layer in output_layers]

        return Model(input_tensors, outputs=built_output_layers, name=model_name)

    def build_gan(
        self,
        generator: Model,
        discriminator: Model,
        ignored_layer_names: Optional[List[str]] = None,
        model_name: str = "gan",
    ) -> Model:
        """
        Defines a GAN consisting of a generator and a discriminator model. The GAN is used to train the generator.

        :param generator
            A model object that represents a generator / decoder.
        :param discriminator
            A model object that represents a discriminator.
        :param ignored_layer_names
            A list of layer names that are ignored as gan outputs. If empty, the gan will use all output
            layers of the generator / decoder + the discriminator's output, which is always used.
        :param model_name
            The name of the model (useful for plots)

        :return: A model object.
        """
        inputs = [Input(shape=tensor.shape[-1], name=tensor.name.split(":")[0]) for tensor in generator.inputs]
        generated = generator(inputs)
        # Only use generated outputs as discriminator inputs that are not specified in 'ignored_layer_names'
        discriminator_inputs = (
            [output for output in generated if output.name.split("/")[1] not in ignored_layer_names]
            if ignored_layer_names
            else generated
        )
        discriminated = discriminator(discriminator_inputs)
        return Model(inputs, discriminated, name=model_name)

    @abc.abstractmethod
    def train_autoencoder(self, **kwargs):
        """ When using an epn architecture, it should always be possible to train the autoencoder separately """
        pass

    @abc.abstractmethod
    def train(self, epochs: int, batch_size: int, steps_per_epoch: int, train_encoder: bool):
        """ When using a epn architecture, a train function has to be provided """
        pass

    def save_model_architecture_images(self, models: List[Model], path: str, fmt: str = "png"):
        """
        Saves all passed models as PNGs into a defined subfolder. A common use case for this method is to call it
        in the respective subclasses with the defined models.

        :param models:
            A list of models whose architecture should be visualized and saved.
        :param path: str
            Relative path from the execution directory.
        :param fmt: str
            Format of the resulting file. Valid options are "png" and "svg".
        :return:
            None
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        [plot_model(m, f"{path}/{m.name}_architecture.{fmt}", show_shapes=True, expand_nested=True) for m in models]
