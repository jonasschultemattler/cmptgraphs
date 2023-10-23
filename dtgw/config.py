CONFIG = {
	# Gurobi primal/dual gap limit:
	"gaplimit": 0.001,
	# Gurobi time limit:
	"timelimit": float('inf'),
	# Use ILP instead of QP:
	"linearize": False,
	"verbosity": 0,
	# one of "diagonal warping", "optimistic warping", "sigma*", "optimistic matching":
	"heuristic initialization": "diagonal warping",
	"max iterations": float("inf"),
}
