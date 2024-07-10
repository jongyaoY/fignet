# MIT License
#
# Copyright (c) [2024] [Zongyao Yi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import matplotlib.pyplot as plt
import numpy as np


def init_fig():
    # plt.rcParams['xtick.major.pad'] = 8
    px = 1 / plt.rcParams["figure.dpi"]
    fig, ax = plt.subplots(figsize=(2000 * px, 700 * px))
    fig.legend(
        [
            plt.Line2D([0], [0], color="c", lw=4),
            plt.Line2D([0], [0], color="b", lw=4),
            plt.Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    # fig.set_size_inches(20, 10)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient")
    # ax.set_aspect(0.1)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    return fig


def simplify_layer_name(layer_name):
    names_to_remove = ["_encode_process_decode", "_processor"]
    out = layer_name.split(".")
    for n in names_to_remove:
        if n in out:
            out.remove(n)
    return ".".join(out)


def plot_grad_flow(named_parameters, fig: plt.Figure):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layer = simplify_layer_name(n)
            layers.append(layer)

            try:
                ave_grads.append(p.grad.detach().cpu().abs().mean())
                max_grads.append(p.grad.detach().cpu().abs().max())
            except AttributeError:
                ave_grads.append(0.0)
                max_grads.append(0.0)
                pass
    ax = fig.axes[0]
    if len(ax.get_xticks()) != len(ave_grads):
        ax.set_xticks(
            range(0, len(ave_grads), 1),
            layers,
            fontsize=6,
            rotation="vertical",
        )
        # ax.tick_params(axis='x', width=2)
        ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(
            bottom=-0.001, top=0.02
        )  # zoom in on the lower gradient regions
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.1, lw=1, color="b")


def plot_graph(
    graph,
):
    mesh_nodes = graph["pos"]["mesh"][-1, ...]
    obj_nodes = graph["pos"]["object"][-1, ...]
    mm_edges = graph["index"]["mm"]
    mo_edges = graph["index"]["mo"]
    ff_edges = graph["index"]["ff"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.scatter(*mesh_nodes.T, alpha=0.2, s=100, color="blue")
    ax.scatter(*obj_nodes.T, alpha=0.2, s=100, color="red")
    for mm_index in mm_edges.T:
        edge = np.array([mesh_nodes[mm_index[0]], mesh_nodes[mm_index[1]]])
        ax.plot(*edge.T, color="gray")
    for mo_index in mo_edges.T:
        edge = np.array([mesh_nodes[mo_index[0]], obj_nodes[mo_index[1]]])
        ax.plot(*edge.T, color="green")
    for edge_id in range(ff_edges.shape[1]):
        ff_index = ff_edges[:, edge_id, ...].T
        ax.plot_trisurf(
            *mesh_nodes[ff_index[:, 0]].T, linewidth=0.2, antialiased=True
        )
        ax.plot_trisurf(
            *mesh_nodes[ff_index[:, 1]].T, linewidth=0.2, antialiased=True
        )
        for i in range(3):
            edge = np.array(
                [mesh_nodes[ff_index[i, 0]], mesh_nodes[ff_index[i, 1]]]
            )
            ax.plot(*edge.T, color="red")

    ax.grid(False)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
