import deepstruct
import deepstruct.sparse
from deepstruct.transform import GraphTransform
import torch
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def show_network_graph():
    shape_input = (50,)
    layers = [100, 50, 100]
    output_size = 10

    model = deepstruct.sparse.MaskedDeepFFN(shape_input, output_size, layers)

    for layer in deepstruct.sparse.maskable_layers(model):
        layer.weight[:, :] += 1

    functor = GraphTransform(torch.randn((1,) + shape_input))

    result = functor.transform(model)

    assert len(result.nodes) == shape_input[0] + sum(layers) + output_size
    assert (
        len(result.edges)
        == shape_input[0] * layers[0]
        + sum(l1 * l2 for l1, l2 in zip(layers[0:-1], layers[1:]))
        + layers[-1] * output_size
    )

    pos = graphviz_layout(result, prog='dot')
    nx.draw(result, pos, with_labels=True, arrows=True)
    plt.show()


