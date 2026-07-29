"""qec_tile — tile codes on a square lattice with boundary."""
# Must come first: config publishes the BLAS/OpenMP thread caps, which are
# only read when those libraries load (i.e. at `import numpy` below).
from . import config  # noqa: F401  isort:skip

from ._core import add, parity
from .bp import METHODS, BpIteration, bp, bp_trace, tanner_edges
from .circuit import circuit_failure_rate, memory_z_base, memory_z_circuit
from .noise_model import NoiseModel
from .decode import (DECODERS, failure_rate, logical_error_rate, make_decoder,
                     sample_residuals)
from .directional import (PAPER_CODES, build_directional_code,
                          displacement_vectors, parse_directional_word,
                          satisfies_parity_condition, tile_from_word,
                          walk_edges)
from .distance import (distance_bruteforce, distance_ilp,
                       distance_upper_bound)
from .pheno import spacetime_channel, spacetime_matrices
from .sinter_sampling import SINTER_DECODERS, collect
from .tile import TILES, TileCode, build_tile_code, paper_code

__all__ = ["add", "parity", "config", "BpIteration", "bp", "bp_trace",
           "METHODS", "tanner_edges", "TILES", "TileCode", "build_tile_code",
           "paper_code", "circuit_failure_rate", "distance_bruteforce",
           "distance_ilp", "distance_upper_bound", "DECODERS", "failure_rate",
           "logical_error_rate", "make_decoder", "memory_z_base",
           "memory_z_circuit", "NoiseModel", "sample_residuals",
           "SINTER_DECODERS", "collect", "spacetime_channel",
           "spacetime_matrices", "PAPER_CODES", "build_directional_code",
           "displacement_vectors", "parse_directional_word",
           "satisfies_parity_condition", "tile_from_word", "walk_edges"]
