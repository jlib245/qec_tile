"""qec_tile — tile codes on a square lattice with boundary."""
from ._core import add, parity
from .circuit import circuit_failure_rate, memory_z_circuit
from .decode import (DECODERS, failure_rate, logical_error_rate, make_decoder,
                     sample_residuals)
from .distance import (distance_bruteforce, distance_ilp,
                       distance_upper_bound)
from .pheno import spacetime_channel, spacetime_matrices
from .tile import TILES, TileCode, build_tile_code, paper_code

__all__ = ["add", "parity", "TILES", "TileCode", "build_tile_code",
           "paper_code", "circuit_failure_rate", "distance_bruteforce",
           "distance_ilp", "distance_upper_bound", "DECODERS", "failure_rate",
           "logical_error_rate", "make_decoder", "memory_z_circuit",
           "sample_residuals", "spacetime_channel", "spacetime_matrices"]
