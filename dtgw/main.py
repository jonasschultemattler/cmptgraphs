import argparse
import time
import textwrap
import sys

from common import read_temporal_graph
from tools import LoggingObserver, OutputSink
from config import CONFIG
from vertex_signatures import SIGNATURE_PROVIDERS



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("tgraph1", type=argparse.FileType("r"))
	parser.add_argument("tgraph2", type=argparse.FileType("r"))
	parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity. Pass twice to increase even more.")
	parser.add_argument("--log", type=argparse.FileType("w"), help="Logfile")
	parser.add_argument("--log-cost", action="store_true", help="Include detailed costs in logfile. Requires --log.")
	parser.add_argument("--ilp", action="store_true", help="Use ILP instead of heuristic.")
	parser.add_argument("--signature", choices=SIGNATURE_PROVIDERS.keys(), default="degrees")
	parser.add_argument("--normalize", choices=("layers", "vertices", "both"), help=textwrap.dedent("""\
		Normalze the final result by dividing by
		* the maximum number of layers in any of the two graphs,
		* the minimum number of vertices in any of the two graphs,
		* or both of the above."""
	))
	parser.add_argument("--max-iterations", metavar="N", type=int, help="(Heuristic only) Stop after at most N iterations.")
	parser.add_argument("--initialize", 
		choices = ("diagonal warping", "optimistic warping", "sigma*", "optimistic matching"),
		default = "diagonal warping",
		help = "Initialization to  use for the heuristic"
	)
	args = parser.parse_args()

	CONFIG["verbosity"] = int(args.verbose)

	if args.max_iterations is not None:
		CONFIG["max iterations"] = args.max_iterations

	CONFIG["heuristic initialization"] = args.initialize

	if args.ilp:
		import dtgw_ilp
		compute_dtgw = dtgw_ilp.compute_dtgw
	else:
		import dtgw_alternating
		compute_dtgw = dtgw_alternating.compute_dtgw


	observer = LoggingObserver()
	if CONFIG["verbosity"] >= 1:
		observer.add_sink(OutputSink(sys.stdout, {"progress"}))
	if args.log:
		flags = {"result", "progress"}
		if args.log_cost:
			flags.add("detailed cost")
		if CONFIG["verbosity"] >= 2:
			flags.add("full progress")
		observer.add_sink(OutputSink(args.log, flags))

	ltg1 = read_temporal_graph(args.tgraph1)
	ltg2 = read_temporal_graph(args.tgraph2)
	observer.update_progress("Graphs read")

	signature_provider = SIGNATURE_PROVIDERS[args.signature](
		# hardcoded values for now
		betweenness_weight = 1000,
		component_weight = 0.1,
		max_degree = 10
	)
	result = dtgw(
		ltg1,
		ltg2,
		compute_dtgw = compute_dtgw,
		signature_provider = signature_provider,
		observer = observer,
		normalize_vertices = args.normalize in ("vertices", "both"),
		normalize_layers = args.normalize in ("layers", "both")
	)
	print("Result:", result)



def dtgw(ltg1, ltg2, compute_dtgw, signature_provider, observer=LoggingObserver(), normalize_vertices=False, normalize_layers=False):
	"""Compute the dtgw distance between two temporal graphs
	and prints the result to stdout.

	Arguments
	---------
	ltg1, ltg2: LabeledTemporalGraph
		The temporal graphs to compare
	compute_dtgw:
		one of [dtgw_ilp.compute_dtgw, dtgw_alternating.compute_dtgw]
		or any function with the same interface
	signature_provider: VertexSignatureProvider
		vertex signatures to use
	logfile: (optional)
		Logfile to write to
	"""
	
	# observer.set_output_maps(
	# 	vertex_map_1 = ltg1.vertex_index.lookup,
	# 	vertex_map_2 = ltg2.vertex_index.lookup,
	# 	layer_map_1 = ltg1.layer_index.lookup,
	# 	layer_map_2 = ltg2.layer_index.lookup,
	# )

	features1 = signature_provider.signatures(ltg1)
	features2 = signature_provider.signatures(ltg2)

	distance = compute_dtgw(
		features1,
		features2,
		eps = signature_provider.eps,
		metric = signature_provider.metric,
		observer = observer
	)
	if normalize_vertices:
		distance /= min(ltg1.tgraph.num_vertices(), ltg2.tgraph.num_vertices())
	if normalize_layers:
		distance /= max(ltg1.tgraph.lifetime, ltg2.tgraph.lifetime)
	return distance


if __name__ == "__main__":
	main()