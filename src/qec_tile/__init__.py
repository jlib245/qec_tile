"""qec_tile — 경계가 있는 정사각 격자 위의 tile code."""
# 맨 먼저 와야 한다: config가 BLAS/OpenMP thread cap을 심는데, 그 값은
# 라이브러리가 로드될 때만 읽힌다 (즉 아래 numpy import 시점).
from . import config  # noqa: F401  isort:skip

from . import cycles  # 분석용 진단; 코어 API 밖이라 모듈째로 둔다
from ._core import add, parity
from .bp import METHODS, BpIteration, bp, bp_trace, tanner_edges
from .circuit import circuit_failure_rate, memory_z_base, memory_z_circuit
from .noise_model import NoiseModel
from .decode import (DECODERS, block_failure_rate, code_capacity_block_rate,
                     make_decoder, sample_residuals)
from .directional import (PAPER_CODES, build_directional_code,
                          displacement_vectors, parse_directional_word,
                          satisfies_parity_condition, tile_from_word,
                          walk_edges)
from .distance import (distance_bruteforce, distance_ilp,
                       distance_upper_bound)
from .pheno import spacetime_channel, spacetime_matrices
from .sinter_sampling import SINTER_DECODERS, collect
from .tile import TILES, TileCode, build_tile_code, paper_code
from .walk2 import (CHECK_X, CHECK_Z, DATA, ROUTING, Qubit, check_starts,
                    dormant, flow_step, partial_sums, reverse_steps,
                    walk_crossings, walk_layout, walk_memory_z_base,
                    walk_round, walk_schedule)

__all__ = ["add", "parity", "config", "cycles", "BpIteration", "bp",
           "bp_trace", "METHODS", "tanner_edges", "TILES", "TileCode",
           "build_tile_code", "paper_code", "circuit_failure_rate",
           "distance_bruteforce", "distance_ilp", "distance_upper_bound",
           "DECODERS", "block_failure_rate", "code_capacity_block_rate",
           "make_decoder", "memory_z_base",
           "memory_z_circuit", "NoiseModel", "sample_residuals",
           "SINTER_DECODERS", "collect", "spacetime_channel",
           "spacetime_matrices", "PAPER_CODES", "build_directional_code",
           "displacement_vectors", "parse_directional_word",
           "satisfies_parity_condition", "tile_from_word", "walk_edges",
           "CHECK_X", "CHECK_Z", "DATA", "ROUTING", "Qubit",
           "check_starts", "dormant", "flow_step", "partial_sums",
           "reverse_steps", "walk_crossings", "walk_layout",
           "walk_memory_z_base", "walk_round", "walk_schedule"]
