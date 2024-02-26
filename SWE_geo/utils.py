# coding=ISO-8859-1
import os
import sys
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import pygraphviz as pgv
import seaborn as sns

"""
Glossary (Svenska - English)
----------------------------
Ändringstyp - Type of change
Datum ikraftträdande - Date of entry into force
Gammal kod - Old code
Ny kod - New code 
"""


def read_geo_changes(fp_chg):
    """ 
    Reads data from municipality changes 
    downloaded from http://regina.scb.se/indelningsandringar
    
    Parameters
    ----------
    fp_chg : str
        Directory of the input file
    
    Returns
    -------
    df_change : pd.DataFrame
        Pandas dataframe containing change information
    """
    
    # Load data
    df_change = pd.read_csv(fp_chg, encoding="ISO-8859-1", sep="\t",
                            parse_dates=True, infer_datetime_format=True,
                            dtype={"Gammal kod":str, "Ny kod":str})
    
    # Change datatype for date to datetime 
    df_change["Datum ikraftträdande"] = (
        pd.to_datetime(df_change["Datum ikraftträdande"])
    )
    
    # Sort by ascending date
    df_change = (
        df_change
            .sort_values(["Datum ikraftträdande"])
            .reset_index(drop=True)
    )
    
    return df_change


def create_nx_graph(df, print_info=True, **kwargs):
    """
    Create an NetworkX graph object from a pandas dataframe
    containing sources, targets and other attribution, if available,
    and (optionally) print some information about the object.
    See https://w.wiki/74$ and 
    http://mathworld.wolfram.com/WeaklyConnectedDigraph.html
    for definition of weakly connected component
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe frame 
    print_info : bool, optional
        Should information about the graph object be printed?    
    kwargs : dict, optional
        Further parameters to other functions
          
    Returns
    -------
    graph : NetworkX graph object of the specified kind
        Graph generated from the dataframe
    """
    
    # Use dataframe to generate graph object
    graph = nx.from_pandas_edgelist(df=df, **kwargs)
    
    # Print info
    if(print_info):
        print(nx.info(G=graph))
        n_comp = nx.number_weakly_connected_components(G=graph)
        print("Number of weakly connected components: %s" % (n_comp))
    
    return graph


def create_children_graph(graph, node, date,
                          print_info=True):
    """
    Create the subgraph induced by all children (descendants) of the given 
    node and after the given date
    
    Parameters
    ----------
    graph : NetworkX graph
        Original graph
    node : str
        Query node name
    date : np.datetime64
        Query date
    print_info : bool, optional
        Should information about the graph object be printed?
    
    Returns
    -------
    subgraph : NetworkX graph, optional
        Induced graph
    """

    # Compute distance from the given node to its children
    # and extract the names of the children nodes
    children_name = np.array(
            list(
            nx.single_source_shortest_path_length(
                G=graph, source=node
            ).keys()
        )
    )
    
    # Filter out edges older than the given date 
    future_edges = [
            (k) for k, v in (
                nx.get_edge_attributes(
                    G=graph.subgraph(nodes=children_name),
                    name="Datum ikraftträdande"
                ).items()
            )
            if v >= date
    ]
    if (len(future_edges) == 0):  # No edges found
        if(print_info):
            print("No changes found for geographic unit %s after %s"
                  % (node, date)
            )
    
    # Create subgraph
    subgraph = nx.edge_subgraph(
            G=graph,
            edges=future_edges
    )
    
    # Print info
    if(print_info):
        print(nx.info(G=subgraph))
        n_comp = (
            nx.number_weakly_connected_components(G=subgraph)
        )
        print("Number of weakly connected components: %s" % (n_comp))
    
    return subgraph


def find_current_geo(graph, node, date, all_current,
                     print_info=True):
    """
    Find the geographic units that the given node is part of by taking the 
    intersection of all its children
    
    Parameters
    ----------
    graph : NetworkX graph
        Original graph
    node : str
        Query node name
    date : np.datetime64
        Query date
    all_current : np.ndarray
        Current municipality codes 
    print_info : bool, optional
        Should information about the graph object be printed?
        
    Returns
    -------
    current : np.1darray
        1D array of current code(s)
    """

    # Create subgraph induced by the given node and date
    # If no future edges are found, be sure to include the node itself
    try:
        subgraph = create_children_graph(graph, node, date,
                                         print_info=print_info)
        subgraph_nodes = np.array(list(subgraph))
        if (subgraph_nodes.size == 0):
            subgraph_nodes = np.array([node])
    except nx.NodeNotFound:
        if node in all_current:
            subgraph_nodes = np.array([node])
        else:
            raise ValueError("Geographic unit %s doesn't exist." % (node))
    current = np.intersect1d(
        ar1=subgraph_nodes,
        ar2=all_current
    )
    
    if(print_info):
        print("Geographic unit", node, "is now part of unit(s)", current) 
    
    return current


def plot_graph(graph, save_dot=None,
               plot_path=None, prog="fdp",
               date=None, old_style="dashed", edge_color="date",
               node_attrs=None, edge_attrs=None, graph_attrs=None,
               node=None, plot_subgraph=None,
               **kwargs):
    """
    Plot the given graph and (optionally) the subgraph induced by the given node
    
    Parameters
    ----------
    graph : NetworkX graph
        Graph to be plotted
    save_dot : str, optional
        Path to save the DOT code of the graph
    plot_path : str, optional
        Path to save the plot of the full graph
    prog : str, optional
        Graphviz layout method
        (see https://graphviz.gitlab.io/_pages/pdf/dot.1.pdf for more details)
    date : np.datetime64, optional
        Date of interest
    old_style : str, optional
        Should edges whose date is older than the given date be plotted using a
        different style? If not, set this argument to `None`.
    edge_color : str, optional
        Coloring scheme for the edges. Currently only support `"date"`, which
        gives edges with different dates different colors. If not, set this
        argument to `None`.
        Should the subgraph induced by the node 
    node_attrs : dict, optional
        Global node attributes
    edge_attrs : dict, optional
        Global edge attributes
    graph_attr : dict, optional
        Global graph attributes
    node : str, bool
        Query node name
    plot_subgraph : str, optional
        Path to save the plot of the subgraph induced by node
    kwargs : dict, optional
        Additional keyword arguments for construction of the color palette
        
    Returns
    -------
    agraph : pygraphviz graph
        Pygraphviz object generated from the NetworkX object for the full graph
    sub_agraph : pygraphviz graph, optional
        Pygraphviz object generated from the NetworkX object for the subraph
        induced by the given node
    """
    
    # Color and style for edges
    # Get times of change
    edge_time = nx.get_edge_attributes(
        G=graph,
        name="Datum ikraftträdande"
    )
    change_times = np.unique(
        np.array(
            [v for _, v in edge_time.items()]
        )
    )
    if (edge_color == "date"):  # both color and style
        # One color for each change time
        colors = sns.cubehelix_palette(
            n_colors=change_times.size,
            **kwargs
        ).as_hex()
        time_color = dict(zip(change_times, colors))
        edge_color_style = {
            key:{"color":time_color[val],
                 "style":(old_style if (date is not None and
                                        val < date and
                                        old_style is not None)
                          else "bold")
                } for key, val in edge_time.items()
        }
    elif (edge_color is None):  # style only
        edge_color_style = {
            key:{"style":(old_style if (date is not None and
                                        val < date
                                        and old_style is not None)
                          else "bold")
                } for key, val in edge_time.items()
        }
    else:
        raise ValueError("edge_color can currently only be date")
    # Add the attributes above to the graph
    nx.set_edge_attributes(
        G=graph,
        values=edge_color_style
    )
    
    # Global (i.e. applicable indiscriminately to all) attributes
    graph.graph['node'] = node_attrs
    graph.graph['edge'] = edge_attrs
    graph.graph['graph'] = graph_attrs
    
    # Convert to pygraphviz graph
    agraph = nx.drawing.nx_agraph.to_agraph(N=graph)
    
    # Save DOT file and plot
    if (save_dot is not None): 
        agraph.write(path=save_dot)
    if (plot_path is not None):
        agraph.draw(path=plot_path, prog=prog)
        _plot_path = str.split(plot_path, sep=".", maxsplit=1)
        legend_path = _plot_path[0] + "_legend" + ".pdf"
        
    # Make legend
    time_color_style = {
        key.strftime("%Y-%m-%d"):{"value":1, "color":val}
        for key, val in time_color.items()
    }
    df_legend = (
        pd.DataFrame
        .from_dict(data=time_color_style, orient="index")
        .reset_index()
        .rename(columns={"index":"time"})
    )
    with sns.axes_style("white", {"font.family":"serif"}):
        # Make plot
        f, ax = plt.subplots(1, 1, figsize=(3, 12))
        p = sns.barplot(x="value", y="time", data=df_legend,
                        palette=df_legend["color"])
        # Axes and ticks
        ax.tick_params(left=False)
        ax.get_yaxis().label.set_visible(False)
        ax.get_xaxis().set_visible(False)
        sns.despine(left=True, bottom=True)
        # Save
        p.get_figure().savefig(legend_path, bbox_inches="tight")
    
    # Generate and plot subgraph
    if (node is not None):
        subgraph = create_children_graph(
            graph, node, date, print_info=True
        )
        nx.set_edge_attributes(
            G=subgraph,
            values=edge_color_style
        )
        subgraph.graph['node'] = node_attrs
        subgraph.graph['edge'] = edge_attrs
        subgraph.graph['graph'] = graph_attrs
        sub_agraph = nx.drawing.nx_agraph.to_agraph(N=subgraph)
        if (plot_subgraph is not None):
            sub_agraph.draw(path=plot_subgraph, prog=prog)
        
        return agraph, sub_agraph
    else:
        return agraph
    

def main():
    """
    Main function for testing
    """
    
    # Command line arguments 
    p = ap.ArgumentParser()
    p.add_argument("-g", "--geo", type=str, required=True,
                   help="geographic code")
    p.add_argument("-d", "--date", type=str, required=True,
                   help="date of record")
    p.add_argument("-n", "--namebase", type=str, required=False,
                   nargs="?", default=os.getcwd(),
                   help="working directory")
    p.add_argument("-f", "--plot_full", action="store_const", const=True,
                   help="plot full graph")
    p.add_argument("-s", "--plot_sub", action="store_const", const=True,
                   help="plot subgraph")
    args = p.parse_args()
    namebase = args.namebase
    node = args.geo
    date = np.datetime64(args.date)
    plot_full = args.plot_full
    plot_sub = args.plot_sub
    
    """
    The code block below can be used to find current geographic units within the
    python ecosystem by specifying assigning appropriate values to the
    variables `namebase`, `node`, `date`, `plot_full`, `plot_sub`
    """ 
    # Current geographic units and changes
    fp_current_kommun = namebase + "/kommuner_in_shp.txt"
    fp_current_laen = namebase + "/laen_in_shp.txt"
    if (len(node) == 4):  # Kommun
        fp_current = fp_current_kommun
        fp_chg = namebase + "/kommun_code_changes.txt"
    elif (len(node) == 2):  # Län
        fp_current = fp_current_laen
        fp_chg = namebase + "/laen_code_changes.txt"
    else:
        raise ValueError(
            "Geographic code can only be 2 digits or 4 digits long"
        )
    geo_in_shp = np.loadtxt(fname=fp_current, dtype=str)
    df_change = read_geo_changes(fp_chg)
    
    # Convert to multidigraph
    graph = create_nx_graph(
        df=df_change,
        print_info=True,
        source="Gammal kod",
        target="Ny kod",
        edge_attr=["Datum ikraftträdande", "Ändringstyp"],
        create_using=nx.MultiDiGraph()
    )
    
    # Find current geographic units
    current_codes = find_current_geo(
        graph=graph,
        node=node, date=date,
        all_current=geo_in_shp,
        print_info=True
    )

    # Make plots
    if (plot_full):
        # Global attributes
        # See https://www.graphviz.org/doc/info/ for further details about
        # node, edge and graph attributes
        node_attrs = {"shape":"none",
                      "width":0,
                      "height":0,
                      "margin":0.1}
        edge_attrs = {"arrowsize":0.4,
                      "penwidth":0.8,
                      "splines":"curved"}
        graph_attrs = {}
        # For seaborn color wheel used for color-coding the edges
        color_options = {"rot":0.75, "start":1.5}
    
        if (plot_sub):
            agraph, sub_agraph = plot_graph(
                graph=graph,
                save_dot=namebase + "/full_graph.dot",
                plot_path=namebase + "/full_graph.pdf",
                prog="fdp", date=date,
                old_style="dashed", edge_color="date",
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                graph_attrs=graph_attrs,
                node=node,
                plot_subgraph=namebase + "/subgraph_%s_%s.pdf" % (node, date),
                **color_options
            )
        else:
            sub_agraph = plot_graph(
                graph=graph,
                save_dot=namebase + "/full_graph.dot",
                plot_path=namebase + "/full_graph.pdf",
                prog="fdp", date=date,
                old_style="dashed", edge_color="date",
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                graph_attrs=graph_attrs,
                node=node,             
                **color_options
            )

    
if __name__ == "__main__":
    main()
