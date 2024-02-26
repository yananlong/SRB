# coding=ISO-8859-1
from utils import *
import numpy as np


def main():
    # Current geographic units and changes    
    fp_current_kommun = "kommuner_in_shp.txt"
    fp_current_laen = "laen_in_shp.txt"
    fp_chg_kommun = "kommun_code_changes.txt"
    fp_chg_laen = "laen_code_changes.txt"
    kommuner_in_shp = np.loadtxt(fname=fp_current_kommun, dtype=str)
    laen_in_shp = np.loadtxt(fname=fp_current_laen, dtype=str)
    df_change_kommun = read_geo_changes(fp_chg_kommun)
    df_change_laen = read_geo_changes(fp_chg_laen)
    
    # Geographic unit/year pairs
    fp_geo_year = "/geo_year.csv"
    ar_geo_year = np.loadtxt(
        fname="geo_year.csv", dtype=str, delimiter=",",
        converters={
            0: lambda s: s.zfill(len(s) + 1 if len(s) % 2 == 1 else len(s)),
        }
    )

    # Convert to multidigraph
    graph_kommun = nx.from_pandas_edgelist(
        df=df_change_kommun,
        source="Gammal kod",
        target="Ny kod",
        edge_attr=["Datum ikraftträdande", "Ändringstyp"],
        create_using=nx.MultiDiGraph()
    )
    graph_laen = nx.from_pandas_edgelist(
        df=df_change_laen,
        source="Gammal kod",
        target="Ny kod",
        edge_attr=["Datum ikraftträdande", "Ändringstyp"],
        create_using=nx.MultiDiGraph()
    )

    # Find current geographic units
    with open("geo_year_current.txt", "w") as f:
        for node, year in ar_geo_year:
            # Use the correct objects for kommuner and län
            if (len(node) == 4):
                graph = graph_kommun
                geo_in_shp = kommuner_in_shp
            elif (len(node) == 2):
                graph = graph_laen
                geo_in_shp = laen_in_shp
            else:
                continue
            
            try:
                current = find_current_geo(
                    graph=graph,
                    node=node, date=np.datetime64(year),
                    all_current=geo_in_shp,
                    print_info=True
                )
            except ValueError:
                current = np.array([])
                
            print(node, year, current,
                  file=f)

if __name__ == "__main__":
    main()
