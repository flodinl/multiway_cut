# multiway_cut
Instructions to run:
1) You will need to install "pulp" and "networkx" to your Python3 Distribution (pip install pulp/networkx)
2) Running "python3 multiway.py" will run both algorithms on the included test graph.
3) If you want to use another graph, format it as a text file (use the included graph as a template, there should be 1 row of the cost matrix per line (comma separated), followed by a blank line, followed by a list of the terminals on the next line (also comma separated). Then in the main function, change the filepath to the desired filepath.
4) There is also support for directly saving graphs stored internally as lists rather than manually converting them to a text format, see the function documentation for details.
