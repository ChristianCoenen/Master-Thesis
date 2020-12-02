from abc import ABC

from manim import *
from manim import color as c
from itertools import product
import itertools as it


class NetworkMobjectSimple(VGroup, ABC):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_stroke_color": c.BLUE,
        "neuron_stroke_width": 2,
        "neuron_fill_color": c.GREEN,
        "bias_radius": 0.15,
        "bias_stroke_color": c.BLUE,
        "bias_stroke_width": 2,
        "bias_fill_color": c.GREEN,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "edge_color": c.LIGHT_GREY,
        "edge_stroke_width": 2,
        "edge_propagation_color": c.YELLOW,
        "edge_propagation_time": 1,
        "nn_position": ORIGIN,
        "include_bias": False,
    }

    def __init__(self, sizes, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.layer_sizes = sizes
        self._add_neurons()
        self._add_edges()

    def _add_neurons(self):
        layers = VGroup(*[self._get_layer(n_neurons, idx) for idx, n_neurons in enumerate(self.layer_sizes)])
        layers.arrange(RIGHT, buff=self.layer_to_layer_buff)
        # When bias is shown, the last layer is aligned with all neurons of the previous layer (including its bias)
        # so we shift the last layer down so that it aligns with the neurons of the previous layer (excluding the bias)
        layers[-1].shift(DOWN * self.neuron_to_neuron_buff) if self.include_bias else None
        layers.shift(self.nn_position)
        self.layers = layers
        self.add(self.layers)

    def _get_layer(self, n_neurons, layer_idx):
        # Creates neuron objects and adds them to a VGroup called 'layer'
        layer = VGroup()
        neurons = [
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.neuron_stroke_color,
                stroke_width=self.neuron_stroke_width,
                fill_color=self.neuron_fill_color,
                fill_opacity=0,
            )
            for _ in range(n_neurons)
        ]

        if self.include_bias and layer_idx != len(self.layer_sizes) - 1:
            bias = Circle(
                radius=self.bias_radius,
                stroke_color=self.bias_stroke_color,
                stroke_width=self.bias_stroke_width,
                fill_color=self.bias_fill_color,
                fill_opacity=0,
            )
            neurons = VGroup(bias, *neurons)
        else:
            neurons = VGroup(*neurons)

        neurons.arrange(DOWN, buff=self.neuron_to_neuron_buff)

        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.bias = neurons[0] if self.include_bias and layer_idx != len(self.layer_sizes) - 1 else None
        layer.neurons = neurons[1:] if self.include_bias and layer_idx != len(self.layer_sizes) - 1 else neurons
        layer.add(neurons)
        return layer

    def _add_edges(self, forward=True):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            if self.include_bias:
                for b1, n2 in product(l1.bias, l2.neurons):
                    if forward:
                        edge = self._get_edge(b1, n2)
                    else:
                        edge = self._get_edge(n2, b1)
                    edge_group.add(edge)
                    b1.edges_out.add(edge)
                    n2.edges_in.add(edge)
            for n1, n2 in product(l1.neurons, l2.neurons):
                if forward:
                    edge = self._get_edge(n1, n2)
                else:
                    edge = self._get_edge(n2, n1)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    def _get_edge(self, neuron1, neuron2):
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    def get_edge_propagation_animations(self, index, lag_ratio=0, forward=True, width=1.0):
        if not forward:
            self._add_neurons()
            self._add_edges(forward)

        edge_group_copy = self.edge_groups[index].copy()
        edge_group_copy.set_stroke(self.edge_propagation_color, width=width * self.edge_stroke_width)
        return [ShowCreationThenDestruction(edge_group_copy, run_time=self.edge_propagation_time, lag_ratio=lag_ratio)]

    def toggle_bias(self, make_visible):
        for idx, layer in enumerate(self.layers):
            if layer.bias:
                layer.bias.set_stroke(opacity=make_visible)
                layer.bias.edges_out.set_stroke(opacity=make_visible)

    def add_random_values_to_all_neurons(self, scale_factor=1):
        all_neurons = VGroup(*it.chain(*[layer.neurons for layer in self.layers]))
        self.neuron_values = VGroup()
        for neuron in all_neurons:
            neuron.value = DecimalNumber(np.random.uniform(0, 1))
            neuron.value.move_to(neuron)
            neuron.value.scale(scale_factor)
            self.neuron_values.add(neuron.value)
        self.add(self.neuron_values)

    def add_output_values(self, values, scale_factor=1):
        self.remove_values_from_output_neurons()
        self.output_values = VGroup()
        for idx, neuron in enumerate(self.layers[-1].neurons):
            neuron.value = DecimalNumber(values[idx])
            neuron.value.move_to(neuron)
            neuron.value.scale(scale_factor)
            self.output_values.add(neuron.value)
        self.add(self.output_values)

    def remove_values_from_output_neurons(self):
        if hasattr(self, "output_values"):
            self.remove(self.output_values)

    def remove_values_from_all_neurons(self):
        if hasattr(self, "neuron_values"):
            self.remove(self.neuron_values)

    def add_values_to_bias(self, scale_factor=1):
        all_bias_neurons = VGroup(*it.chain(*[layer.bias for layer in self.layers[:-1]]))
        self.bias_values = VGroup()
        for bias_neuron in all_bias_neurons:
            bias_neuron.value = DecimalNumber(1, num_decimal_places=0)
            bias_neuron.value.move_to(bias_neuron)
            bias_neuron.value.scale(scale_factor)
            self.bias_values.add(bias_neuron.value)
        self.add(self.bias_values)

    def remove_values_from_bias(self):
        if hasattr(self, "bias_values"):
            self.remove(self.bias_values)
