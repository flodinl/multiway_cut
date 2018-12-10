import math
import networkx as nx
import pickle
import pulp
import random


def solve_lp(cost_matrix, terminals):
	'''
	Solves the LP described on p. 198 of Shmoys and Williamson given the cost_matrix and terminals,
	and returns a list of lists where the ith entry is a list representing the vector x_i
	'''
	n = len(cost_matrix)
	k = len(terminals)


	# x[u][i] = x_u^i
	x = [[None for i in range(k)] for j in range(n)]
	# print(x)
	assert(len(x) == n)
	assert(len(x[0]) == k)

	# z[u][v][i] = z_{(u, v)}^i
	z = [[[None for i1 in range(k)] for i2 in range(n)] for i3 in range(n)]
	assert(len(z) == n)
	assert(len(z[0]) == n)
	assert(len(z[0][0]) == k)
	for i in range(k):
		for u in range(n):
			# this handles the equality "constraints" that aren't really constraints
			# print("value of i is:", i)
			# print("value of u is:", u)
			# print(x)
			if u in terminals:
				# print("in if block")
				if i == terminals.index(u):
					x[u][i] = 1
				else:
					x[u][i] = 0
			else:
				# print("in else block")
				x[u][i] = pulp.LpVariable("x_" + str(u) + "^" + str(i), lowBound=0)
			for v in range(u):
				z[u][v][i] = pulp.LpVariable("z_" + str(u) + "," + str(v) + "^" + str(i))
	# print(x)

	lp = pulp.LpProblem("LP", pulp.LpMinimize)

	# add the objective
	obj = pulp.LpAffineExpression()
	for u in range(n):
		for v in range(u):
			for i in range(k):
				obj += cost_matrix[u][v] * z[u][v][i]
	lp += (1 / 2) * obj


	# add the constraints
	for u in range(n):
		lp += (pulp.lpSum(x[u]) == 1)


	for u in range(n):
		for v in range(u):
			for i in range(k):
				lp += (z[u][v][i] >= x[u][i] - x[v][i])
				lp += (z[u][v][i] >= x[v][i] - x[u][i])


	# for i in range(k):
	# 	lp += (x[terminals[i]] == 1)

	# solve the LP
	status = lp.solve()
	assert(pulp.LpStatus[lp.status] == "Optimal")

	# print(x)

	# turn the resulting variable values into a list
	x_vector_list = [[None for i in range(k)] for j in range(n)]
	for u in range(n):
		for i in range(k):
			if isinstance(x[u][i], int):
				x_vector_list[u][i] = x[u][i]
			else:
				x_vector_list[u][i] = x[u][i].varValue

	# print(x_vector_list)
	return x_vector_list

def l1_norm(v1, v2):
	'''
	Given lists v1, v2 of the same length, returns the l1 norm of v1 - v2
	'''
	assert(len(v1) == len(v2))
	norm = 0
	for i in range(len(v1)):
		norm += abs(v1[i] - v2[i])
	return norm

def lp_apx(graph_cost_matrix, terminals):
	'''
	Given graph_cost_matrix where entry (i, j) is the cost of edge (i, j) if it exists
	and 0 otherwise, and terminals is a list of which vertices are terminals, outputs the cost and edge set of
	a multiway cut between the terminals using a 3/2-approximation based on LP rounding.
	'''
	n, k = len(graph_cost_matrix), len(terminals)

	# x_vectors[u][i] is x_u^i
	x_vectors = solve_lp(graph_cost_matrix, terminals)
	C = [[] for i in range(k)]
	# print(C)
	r = random.random()
	rand_perm = list(range(k))
	random.shuffle(rand_perm)
	# print(rand_perm)
	X = set()
	for i in range(k - 1):
		ball = set()
		e_pi_i = [0] * k
		# print(e_pi_i)
		e_pi_i[rand_perm[i]] = 1
		for index, x in enumerate(x_vectors):
			if (1 / 2) * l1_norm(e_pi_i, x) <= r:
				ball.add(index)

		C[rand_perm[i]] = ball - X
		X = X | C[rand_perm[i]]
	V = set(range(n))
	C[rand_perm[k - 1]] = V - X
	# print(C)

	edge_set = set()
	for i, C_i in enumerate(C):
		for j, C_j in enumerate(C[0:i]):
			if i == j:
				pass
			else:
				for u in C_i:
					for v in C_j:
						# print("u, v are", u, v)
						if graph_cost_matrix[u][v] != 0:
							edge_set.add((u, v))


	cost = get_cost(graph_cost_matrix, edge_set)
	return cost, edge_set


def find_min_cut(G, s, t):
	'''
	Given GraphX graph G and integers s, t representing indices of vertices in the graph, return
	a set of edges that is the minimum cost cut between nodes s and t
	'''
	cut_value, node_partition = nx.minimum_cut(G, s, t)
	edges = set()
	for v1 in node_partition[0]:
		for v2 in node_partition[1]:
			if G.has_edge(v1, v2):
				edges.add((v1, v2))

	return edges

def get_cost(graph_cost_matrix, edge_set):
	'''
	Takes a matrix representing the costs of the edges and a set of edges, and returns the total
	cost of all the edges in the set.
	'''
	total_cost = 0
	for edge in edge_set:
		i, j = edge[0], edge[1]
		total_cost += graph_cost_matrix[i][j]
	return total_cost

def make_graphx_graph(cost_matrix):
	'''
	Takes a cost_matrix and returns a GraphX graph object representing the graph
	'''
	G = nx.Graph()
	G.add_nodes_from(range(len(cost_matrix)))
	for i in range(len(cost_matrix)):
		for j in range(len(cost_matrix[i])):
			if (i != j) and (cost_matrix[i][j] != 0):
				G.add_edge(i, j, capacity=cost_matrix[i][j])

	return G


def mincut_apx(graph_cost_matrix, terminals):
	'''
	Given graph_cost_matrix where entry (i, j) is the cost of edge (i, j), and terminals which
	is a list of indices of the terminal vertices of G, returns a list of edges (i, j)
	and their total cost using a min-cut based 2-approximation.
	'''

	n = len(graph_cost_matrix)
	
	edge_set = set()
	for terminal in terminals:
		new_cost_matrix = graph_cost_matrix.copy()
		# add sink node t to new_cost_matrix with infinite cost edges to all other terminals
		for terminal_row in new_cost_matrix:
			if terminal_row in terminals:
				terminal_row.append(math.inf)
			else:
				terminal_row.append(0)
		t_row = [0] * (n + 1)
		for index in terminals:
			if index != terminal:
				t_row[index] = math.inf
		new_cost_matrix.append(t_row)

		G = make_graphx_graph(new_cost_matrix)

		min_cut_edges = find_min_cut(G, terminal, n) # n is the index of the sink t
		edge_set = edge_set | min_cut_edges

	cost = get_cost(graph_cost_matrix, edge_set)

	return cost, edge_set

def write_test_graph(filepath):
	'''
	Writes a file to "filepath" in the parent directory containing the compressed representation
	of the hard-coded graph below. This graph can then be opened by calling "read_saved_graph(filepath)".
	'''
	terminals = [0, 1, 2]
	cost_matrix = [[0, 0, 0, 1.9, 0, 0], [0, 0, 0, 0, 1.9, 0], [0, 0, 0, 0, 0, 1.9], [1.9, 0, 0, 0, 1, 1], [0, 1.9, 0, 1, 0, 1], [0, 0, 1.9, 1, 1, 0]]
	write_graph_to_file(cost_matrix, terminals, filepath)

def read_saved_graph(filepath):
	'''
	Reads in a graph from "filepath" in the parent directory that was written to filepath using
	the write_graph_to_file function.
	'''
	f = open(filepath, "rb")
	cost_matrix, terminals = pickle.load(f)
	f.close()
	return cost_matrix, terminals

def read_graph_from_text_file(filepath):
	'''
	Reads in a graph from "filepath" in the parent directory that is a textfile consisting of
	the rows of the cost matrix, where entries are separated by commas and rows are separated
	by new lines, followed by a new line, followed by a comma-separated list of which
	rows correspond to terminals. Example of a triangle with two vertices as terminals and unit edge costs
	in this format: 
	
	0, 1, 1
	1, 0, 1
	1, 1, 0

	0, 2
	'''
	f = open(filepath, "r")
	cost_matrix = []
	for line in f:

		if line == "\n":
			terminals_line = f.readline()
		else:
			# print(line)
			cost_matrix.append(list(map(float, line.split(","))))
	terminals = list(map(int, terminals_line.split(",")))
	return cost_matrix, terminals

def write_graph_to_file(cost_matrix, terminals, filepath):
	'''
	Writes a file to "filepath" in the parent directory consisting of a compressed representation of
	cost_matrix and terminals which can then be opened using "read_saved_graph(filepath)".
	'''
	f = open(filepath, "wb")
	pickle.dump((cost_matrix, terminals), f)
	f.close()


def main():
	filepath = "test_graph.pickle"
	# cost_matrix, terminals = read_graph_from_text_file("test_graph.txt")
	# write_test_graph(filepath)
	cost_matrix, terminals = read_saved_graph(filepath)
	mincut_cost, mincut_edge_set = mincut_apx(cost_matrix, terminals)
	lp_cost, lp_edge_set = lp_apx(cost_matrix, terminals)
	print("cost of mincut 2-approximation is", mincut_cost, "with edge set", mincut_edge_set)
	print("cost of LP rounding (3/2)-approximation is", lp_cost, "with edge set", lp_edge_set)


if __name__ == '__main__':
	main()