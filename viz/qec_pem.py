"""
qec_pcm.py
==========
Surface code / Bivariate Bicycle (BB) code / Tile code 의 CSS parity check matrix 생성기.

모든 코드는 CSS 이므로 (H_X, H_Z) 쌍으로 표현되며 H_X @ H_Z.T % 2 == 0 을 만족한다.
  - H_X : (m_X, n) 이진행렬, 각 행 = X-type 스태빌라이저의 support
  - H_Z : (m_Z, n) 이진행렬, 각 행 = Z-type 스태빌라이저의 support
  - k = n - rank(H_X) - rank(H_Z)

References
----------
[1] Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory",
    Nature 627, 778 (2024).                              -> BB codes
[2] Steffan, Choe, Breuckmann, Pereira, Eberhardt,
    "Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary",
    arXiv:2504.09171 / PRL (2025).                       -> tile codes
"""

from itertools import product
import numpy as np

try:                                    # pip install ldpc  -> 훨씬 빠른 GF(2) 연산
    from ldpc import mod2 as _mod2
    from scipy.sparse import csr_matrix as _csr
    _HAS_LDPC = True
except Exception:                       # 없으면 순수 numpy 폴백
    _HAS_LDPC = False


# ============================================================================
# 1. GF(2) 유틸리티
# ============================================================================

def rref_gf2(M):
    """GF(2) 위에서의 RREF. (R, pivot_columns) 반환."""
    R = np.array(M, dtype=np.uint8) % 2
    rows, cols = R.shape
    pivots, r = [], 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if R[i, c]:
                piv = i
                break
        if piv is None:
            continue
        R[[r, piv]] = R[[piv, r]]
        mask = R[:, c].copy().astype(bool)
        mask[r] = False
        R[mask] ^= R[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return R, pivots


def rank_gf2(M):
    M = np.asarray(M, dtype=np.uint8)
    if M.size == 0:
        return 0
    if _HAS_LDPC:
        return int(_mod2.rank(_csr(M)))
    return len(rref_gf2(M)[1])


def nullspace_gf2(M):
    """M x = 0 의 해공간 기저를 행으로 갖는 행렬 반환."""
    M = np.array(M, dtype=np.uint8) % 2
    n = M.shape[1]
    R, piv = rref_gf2(M)
    free = [c for c in range(n) if c not in piv]
    basis = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, p in enumerate(piv):
            basis[i, p] = R[r, f]
    return basis


def css_commute(hx, hz):
    """CSS 조건 H_X H_Z^T = 0 (mod 2) 검사."""
    return not ((hx.astype(np.int64) @ hz.astype(np.int64).T) % 2).any()


def css_k(hx, hz):
    n = hx.shape[1]
    return n - rank_gf2(hx) - rank_gf2(hz)


def css_logicals(hx, hz):
    """
    (Lx, Lz): X-logical / Z-logical 대표원 기저.
    Lx: ker(H_Z) 중 rowspace(H_X) 에 없는 것,  Lz: ker(H_X) / rowspace(H_Z).
    """
    def _quotient(kernel_of, modulo):
        ker = nullspace_gf2(kernel_of)
        base = modulo.copy()
        out = []
        cur = rank_gf2(base) if base.size else 0
        for v in ker:
            trial = np.vstack([base, v]) if base.size else v[None, :]
            r = rank_gf2(trial)
            if r > cur:
                out.append(v)
                base, cur = trial, r
        return np.array(out, dtype=np.uint8)

    return _quotient(hz, hx), _quotient(hx, hz)


def _echelon(M):
    """열 순서를 보존하는 GF(2) full row-echelon."""
    M = np.asarray(M, dtype=np.uint8)
    if _HAS_LDPC:
        R = _mod2.row_echelon(_csr(M), True)[0]
        return np.asarray(R.todense() if hasattr(R, "todense") else R, dtype=np.uint8)
    return rref_gf2(M)[0]


def _dir_distance(H, Hs, trials, rng):
    """ker(H) 안에 있으나 rowspace(Hs) 에는 없는 최소무게 벡터 (randomized 상한)."""
    ker = nullspace_gf2(H)
    n, rs = ker.shape[1], rank_gf2(Hs)
    best = np.inf
    for _ in range(trials):
        perm = rng.permutation(n)
        R = _echelon(ker[:, perm])
        back = np.zeros_like(R)
        back[:, perm] = R                      # 열 순열 되돌리기
        for row in back:
            w = int(row.sum())
            if 0 < w < best and rank_gf2(np.vstack([Hs, row])) > rs:
                best = w
    return best


def distance_upper_bound(hx, hz, trials=30, seed=0):
    """
    randomized information-set 방식의 CSS distance '상한'.
    trials 를 늘릴수록 조여지며, 작은 코드에서는 보통 참값에 도달한다.
    엄밀한 값이 필요하면 ILP 기반 도구(qdistrnd, GAP/QDistRnd 등)를 사용할 것.
    """
    rng = np.random.default_rng(seed)
    d = min(_dir_distance(hz, hx, trials, rng),   # X-logical
            _dir_distance(hx, hz, trials, rng))   # Z-logical
    return int(d) if np.isfinite(d) else None


def report(name, hx, hz, dist_trials=0):
    n = hx.shape[1]
    ok = css_commute(hx, hz)
    k = css_k(hx, hz)
    wx = hx.sum(1).max() if hx.size else 0
    wz = hz.sum(1).max() if hz.size else 0
    qx = hx.sum(0).max() if hx.size else 0
    qz = hz.sum(0).max() if hz.size else 0
    d = distance_upper_bound(hx, hz, trials=dist_trials) if dist_trials else None
    dstr = f", d<={d}" if d else ""
    print(f"{name:38s} [[{n},{k}{dstr}]]  H_X:{hx.shape} H_Z:{hz.shape} "
          f"max stab wt=({wx},{wz}) qubit deg=({qx},{qz}) CSS={ok}")
    return dict(n=n, k=k, d=d, commute=ok)


# ============================================================================
# 2. 고전 부호 블록 & Hypergraph product
# ============================================================================

def repetition_H(n, cyclic=False):
    """반복부호 패리티체크. cyclic=True -> n x n 순환행렬(고리), False -> (n-1) x n."""
    if cyclic:
        H = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            H[i, i] = 1
            H[i, (i + 1) % n] = 1
    else:
        H = np.zeros((n - 1, n), dtype=np.uint8)
        for i in range(n - 1):
            H[i, i] = 1
            H[i, i + 1] = 1
    return H


def hypergraph_product(H1, H2):
    """
    HGP:  qubits = n1*n2 + m1*m2
      H_X = [ H1 (x) I_n2 | I_m1 (x) H2^T ]
      H_Z = [ I_n1 (x) H2 | H1^T (x) I_m2 ]
    """
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    hx = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)),
                    np.kron(np.eye(m1, dtype=np.uint8), H2.T)]) % 2
    hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2),
                    np.kron(H1.T, np.eye(m2, dtype=np.uint8))]) % 2
    return hx.astype(np.uint8), hz.astype(np.uint8)


# ============================================================================
# 3. Surface code
# ============================================================================

def toric_code(d):
    """Toric code [[2d^2, 2, d]] = HGP(cyclic rep, cyclic rep)."""
    H = repetition_H(d, cyclic=True)
    return hypergraph_product(H, H)


def unrotated_surface_code(d):
    """Unrotated planar surface code [[d^2+(d-1)^2, 1, d]] = HGP(rep, rep)."""
    H = repetition_H(d, cyclic=False)
    return hypergraph_product(H, H)


def rotated_surface_code(d):
    """
    Rotated surface code [[d^2, 1, d]] (d 홀수).
    데이터 큐빗은 d x d 격자 위, index q = i*d + j (i=행, j=열).
      - 내부 face (i,j), 0<=i,j<=d-2 : 4-body,  (i+j) 짝수 -> X, 홀수 -> Z
      - 위/아래 경계 : weight-2 Z,   좌/우 경계 : weight-2 X
    """
    assert d >= 2
    n = d * d
    idx = lambda i, j: i * d + j
    X, Z = [], []

    # 각 타입을 한 번에 훑되 경계 체크를 자기 줄 안, 자기 j 자리에 넣는다.
    # 뒤에 몰아 붙이면 행이 격자 순서를 벗어나 PCM의 밴드가 그 지점에서 끊긴다.
    # X는 줄(i)마다 왼쪽(j=0) 또는 오른쪽(j=d-1) 경계를 하나씩 갖고, Z는 열(j)
    # 마다 위(i=0) 또는 아래(i=d-1)를 하나씩 가지므로 그룹 크기가 균일해진다.
    for i in range(d - 1):
        if i % 2:                                # 왼쪽 경계는 j=0
            X.append([idx(i, 0), idx(i + 1, 0)])
        for j in range(d - 1):
            if (i + j) % 2 == 0:
                X.append([idx(i, j), idx(i, j + 1),
                          idx(i + 1, j), idx(i + 1, j + 1)])
        if i % 2 == 0:                           # 오른쪽 경계는 j=d-1
            X.append([idx(i, d - 1), idx(i + 1, d - 1)])

    for j in range(d - 1):
        if j % 2 == 0:                           # 위 경계는 i=0
            Z.append([idx(0, j), idx(0, j + 1)])
        for i in range(d - 1):
            if (i + j) % 2:
                Z.append([idx(i, j), idx(i, j + 1),
                          idx(i + 1, j), idx(i + 1, j + 1)])
        if j % 2:                                # 아래 경계는 i=d-1
            Z.append([idx(d - 1, j), idx(d - 1, j + 1)])

    def to_mat(rows):
        M = np.zeros((len(rows), n), dtype=np.uint8)
        for r, sup in enumerate(rows):
            M[r, sup] = 1
        return M

    return to_mat(X), to_mat(Z)


# ============================================================================
# 4. Bivariate Bicycle (BB) code
# ============================================================================

def _shift(n):
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def bb_monomial(l, m, a, b):
    """x^a y^b,  x = S_l (x) I_m,  y = I_l (x) S_m.  (l*m x l*m 행렬)"""
    x = np.kron(_shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), _shift(m))
    return (np.linalg.matrix_power(x.astype(np.int64), a) @
            np.linalg.matrix_power(y.astype(np.int64), b) % 2).astype(np.uint8)


def bb_code(l, m, A_terms, B_terms):
    """
    Bivariate Bicycle code [1].
      A = sum_{(a,b) in A_terms} x^a y^b,   B = 마찬가지
      H_X = [A | B],   H_Z = [B^T | A^T],   n = 2*l*m
    A, B 가 가환 다항식이므로 H_X H_Z^T = AB + BA = 0 (mod 2) 가 자동 성립.

    예) gross code [[144,12,12]] : l=12, m=6,
        A = x^3 + y   + y^2   -> [(3,0),(0,1),(0,2)]
        B = y^3 + x   + x^2   -> [(0,3),(1,0),(2,0)]
    """
    A = np.zeros((l * m, l * m), dtype=np.uint8)
    B = np.zeros_like(A)
    for a, b in A_terms:
        A ^= bb_monomial(l, m, a, b)
    for a, b in B_terms:
        B ^= bb_monomial(l, m, a, b)
    hx = np.hstack([A, B])
    hz = np.hstack([B.T, A.T])
    return hx, hz


# 논문 [1] Table 3 의 대표 파라미터
BB_CATALOG = {
    "[[72,12,6]]":    dict(l=6,  m=6,  A=[(3, 0), (0, 1), (0, 2)],  B=[(0, 3), (1, 0), (2, 0)]),
    "[[90,8,10]]":    dict(l=15, m=3,  A=[(9, 0), (0, 1), (0, 2)],  B=[(0, 0), (2, 0), (7, 0)]),
    "[[108,8,10]]":   dict(l=9,  m=6,  A=[(3, 0), (0, 1), (0, 2)],  B=[(0, 3), (1, 0), (2, 0)]),
    "[[144,12,12]]":  dict(l=12, m=6,  A=[(3, 0), (0, 1), (0, 2)],  B=[(0, 3), (1, 0), (2, 0)]),
    "[[288,12,18]]":  dict(l=12, m=12, A=[(3, 0), (0, 2), (0, 7)],  B=[(0, 3), (1, 0), (2, 0)]),
}


# ============================================================================
# 5. Tile code  (arXiv:2504.09171)
# ============================================================================
#
# 무한 정사각 격자 Z^2, 큐빗은 edge 위:
#   h(x,y) : (x,y) -> (x+1,y) 수평 edge
#   v(x,y) : (x,y) -> (x,y+1) 수직 edge
#
# tile = B x B 상자 안의 offset 집합.  X-tile 을 (A_h, A_v) 로 주면
# 가환성(T2)을 위해 Z-tile 은 180도 회전 + h<->v 교환으로 결정된다:
#   C_h = (B-1,B-1) - A_v ,   C_v = (B-1,B-1) - A_h
#
# 증명: anchor 차이 d 에서의 겹침 수는
#   [x^d] ( A_h(x) C_h(x^-1) + A_v(x) C_v(x^-1) ) 이고,
#   위 규칙이면 = M * (A_h A_v + A_v A_h) = 0 (mod 2).
#
# layout (unrotated square):
#   bulk anchor  P = [0,L1-1] x [0,L2-1]
#   큐빗         (x,y) in [0,L1+B-2] x [0,L2+B-2]  ->  n = 2 * l * m
#   X anchor     x in [0,L1-1],       y in [-(B-1), L2+B-2]   (위/아래로 B-1 층)
#   Z anchor     x in [-(B-1), L1+B-2], y in [0,L2-1]         (좌/우로 B-1 층)
#   support 는 큐빗 집합으로 truncate, 빈 스태빌라이저는 제거.

def tile_code(B, A_h, A_v, L1, L2, prune=True, return_layout=False):
    """
    Parameters
    ----------
    B      : 상자 크기
    A_h    : X-tile 의 수평 edge offset 리스트, 각 원소 (dx,dy) in [0,B-1]^2
    A_v    : X-tile 의 수직 edge offset 리스트
    L1, L2 : bulk 스태빌라이저 앵커 격자 크기
    prune  : True 면 X 또는 Z 어느 한쪽에도 안 걸리는 큐빗 제거 후 빈 체크 제거

    Returns (H_X, H_Z) [, layout dict]
    """
    A_h = [tuple(t) for t in A_h]
    A_v = [tuple(t) for t in A_v]
    C_h = [(B - 1 - dx, B - 1 - dy) for (dx, dy) in A_v]
    C_v = [(B - 1 - dx, B - 1 - dy) for (dx, dy) in A_h]

    l, m = L1 + B - 1, L2 + B - 1
    qubits = {}
    for x in range(l):
        for y in range(m):
            qubits[('h', x, y)] = len(qubits)
            qubits[('v', x, y)] = len(qubits)

    x_anchors = [(p, q) for p in range(L1) for q in range(-(B - 1), L2 + B - 1)]
    z_anchors = [(p, q) for p in range(-(B - 1), L1 + B - 1) for q in range(L2)]

    def supp(anchor, off_h, off_v):
        p, q = anchor
        s = set()
        for (dx, dy) in off_h:
            key = ('h', p + dx, q + dy)
            if key in qubits:
                s.add(qubits[key])
        for (dx, dy) in off_v:
            key = ('v', p + dx, q + dy)
            if key in qubits:
                s.add(qubits[key])
        return s

    xs = [supp(a, A_h, A_v) for a in x_anchors]
    zs = [supp(a, C_h, C_v) for a in z_anchors]

    n = len(qubits)
    keep = list(range(n))
    if prune:
        covX = set().union(*xs) if xs else set()
        covZ = set().union(*zs) if zs else set()
        keep = sorted(covX & covZ)
        remap = {q: i for i, q in enumerate(keep)}
        xs = [{remap[q] for q in s if q in remap} for s in xs]
        zs = [{remap[q] for q in s if q in remap} for s in zs]
        x_anchors = [a for a, s in zip(x_anchors, xs) if s]
        z_anchors = [a for a, s in zip(z_anchors, zs) if s]
        xs = [s for s in xs if s]
        zs = [s for s in zs if s]

    nn = len(keep)
    hx = np.zeros((len(xs), nn), dtype=np.uint8)
    hz = np.zeros((len(zs), nn), dtype=np.uint8)
    for r, s in enumerate(xs):
        hx[r, sorted(s)] = 1
    for r, s in enumerate(zs):
        hz[r, sorted(s)] = 1

    if return_layout:
        inv = {v: k for k, v in qubits.items()}
        layout = dict(qubit_coords=[inv[q] for q in keep],
                      x_anchors=x_anchors, z_anchors=z_anchors,
                      X_tile=(A_h, A_v), Z_tile=(C_h, C_v), l=l, m=m)
        return hx, hz, layout
    return hx, hz


# 편의: 다항식 a(x), b(y) 로 HGP 형 tile 만들기 (논문 Appendix A)
def hgp_tile(a_degs, b_degs, B):
    """a(x)=sum x^i (i in a_degs) -> 최상단 수평, b(y)=sum y^j -> 최우측 수직."""
    A_h = [(i, B - 1) for i in a_degs]
    A_v = [(B - 1, j) for j in b_degs]
    return A_h, A_v


TILE_CATALOG = {
    # 표면부호 타일 (B=2): X = star, Z = plaquette.  L=d-1 -> [[d^2+(d-1)^2, 1, d]]
    "surface(B=2,w=4)": dict(B=2, A_h=[(0, 1), (1, 1)], A_v=[(1, 0), (1, 1)]),
    # HGP 형 (논문 Appendix A): a(x)=1+x+x^2, b(y)=1+y+y^2.  L=10 -> [[244,4]]
    "hgp(B=3,w=6)":     dict(B=3, A_h=[(0, 2), (1, 2), (2, 2)],
                             A_v=[(2, 0), (2, 1), (2, 2)]),
    # weight-6, 3x3 box.  L1=L2=10 -> [[288, 8, 12]]  (논문 Table I 재현)
    "w6(B=3)->288_8_12": dict(B=3, A_h=[(0, 0), (1, 2), (2, 2)],
                              A_v=[(0, 2), (2, 0), (2, 1)]),
    # weight-8, 4x4 box.  L=9 -> [[288,18,*]],  L=13 -> [[512,18,*]]
    "w8(B=4)->512_18":   dict(B=4, A_h=[(0, 0), (1, 3), (2, 3), (3, 3)],
                              A_v=[(0, 3), (3, 0), (3, 1), (3, 2)]),
}


# ============================================================================
# 6. 데모
# ============================================================================

if __name__ == "__main__":
    print("=== Surface code ===")
    for d in (3, 5, 7):
        report(f"rotated_surface_code(d={d})", *rotated_surface_code(d), dist_trials=30)
    for d in (3, 4):
        report(f"unrotated_surface_code(d={d})", *unrotated_surface_code(d), dist_trials=30)
        report(f"toric_code(d={d})", *toric_code(d), dist_trials=30)

    print("\n=== Bivariate Bicycle code ===")
    for name, p in BB_CATALOG.items():
        hx, hz = bb_code(p["l"], p["m"], p["A"], p["B"])
        report(f"BB {name}", hx, hz, dist_trials=30)

    print("\n=== Tile code ===")
    t = TILE_CATALOG["surface(B=2,w=4)"]
    report("tile surface B=2, L=4", *tile_code(t["B"], t["A_h"], t["A_v"], 4, 4), dist_trials=30)
    t = TILE_CATALOG["hgp(B=3,w=6)"]
    report("tile hgp B=3, L=10", *tile_code(t["B"], t["A_h"], t["A_v"], 10, 10))
    t = TILE_CATALOG["w6(B=3)->288_8_12"]
    report("tile w6 B=3, L=10", *tile_code(t["B"], t["A_h"], t["A_v"], 10, 10), dist_trials=40)
    t = TILE_CATALOG["w8(B=4)->512_18"]
    report("tile w8 B=4, L=9",  *tile_code(t["B"], t["A_h"], t["A_v"], 9, 9))
    report("tile w8 B=4, L=13", *tile_code(t["B"], t["A_h"], t["A_v"], 13, 13))


# ============================================================================
# 7. 시각화 (matplotlib)
# ============================================================================

def plot_pcm(hx, hz, title="", dividers=(), ax=None, cmap=("#2a78d6", "#eb6834")):
    """H_X 와 H_Z 의 sparsity pattern 을 위아래로 쌓아 그린다."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n, mx, mz = hx.shape[1], hx.shape[0], hz.shape[0]
    if ax is None:
        h = 3.2 * (mx + mz) / max(n, 1) + 1.2
        _, ax = plt.subplots(figsize=(7, max(2.2, h)))

    img = np.zeros((mx + mz + 2, n, 4))
    def rgba(hexs, a=1.0):
        hexs = hexs.lstrip("#")
        return [int(hexs[i:i+2], 16) / 255 for i in (0, 2, 4)] + [a]
    img[:mx][hx.astype(bool)] = rgba(cmap[0])
    img[mx + 2:][hz.astype(bool)] = rgba(cmap[1])

    ax.imshow(img, interpolation="nearest", aspect="auto")
    for d in dividers:
        ax.axvline(d - 0.5, color="0.55", lw=0.8, ls="--")
    ax.axhline(mx + 0.5, color="0.85", lw=1.0)
    ax.set_yticks([mx / 2, mx + 2 + mz / 2])
    ax.set_yticklabels(["$H_X$", "$H_Z$"])
    ax.set_xlabel(f"qubit index (n={n})")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return ax


def plot_gallery(path="pcm_gallery.png", dpi=170):
    """대표 코드들의 패리티체크 행렬 갤러리를 파일로 저장."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = []
    items.append(("Rotated surface code, d=7  [[49,1,7]]", *rotated_surface_code(7), ()))
    items.append(("Unrotated surface code, d=5  [[41,1,5]]", *unrotated_surface_code(5), (25,)))
    items.append(("Toric code, L=5  [[50,2,5]]", *toric_code(5), (25,)))
    p = BB_CATALOG["[[144,12,12]]"]
    items.append(("BB gross code  [[144,12,12]]  H_X=[A|B]", *bb_code(p["l"], p["m"], p["A"], p["B"]), (72,)))
    t = TILE_CATALOG["w6(B=3)->288_8_12"]
    items.append(("Tile code, B=3, w=6, L=10  [[288,8,12]]", *tile_code(t["B"], t["A_h"], t["A_v"], 10, 10), ()))
    t = TILE_CATALOG["w8(B=4)->512_18"]
    items.append(("Tile code, B=4, w=8, L=13  [[512,18]]", *tile_code(t["B"], t["A_h"], t["A_v"], 13, 13), ()))

    fig, axes = plt.subplots(len(items), 1, figsize=(7.2, 2.3 * len(items)))
    for ax, (title, hx, hz, div) in zip(np.atleast_1d(axes), items):
        plot_pcm(hx, hz, title=title, dividers=div, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return path