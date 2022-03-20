import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px


def _save_graph(graph, path, cmap=None):
    nx.draw(graph, with_labels=True, node_color=cmap)
    plt.savefig(path)
    plt.clf()


def _save_line_graph(x, y, path, x_label, y_label, title=""):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, "--")
    plt.savefig(path)
    plt.clf()


def _save_dataframe(data, path):
    data.to_csv(path)


def _save_scatter(data, path, x_label, y_label, title=""):
    data = pd.DataFrame(data=data, columns=[x_label, y_label])
    fig = px.scatter(data, x=x_label, y=y_label, title=title)
    fig.write_image(path)
