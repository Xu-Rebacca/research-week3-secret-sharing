"""Brickell 秘密共享方案（基于向量内积的线性方案）。

原理：把秘密嵌入 GF(p) 上的 k 维随机向量 s = (s_1, ..., s_k)，
      其中 s_1 = 秘密，其余分量为随机数。每个参与者 i 持有公开向量
      v_i = (1, x_i, x_i^2, ..., x_i^{k-1})（Vandermonde 行），
      份额 y_i = <s, v_i> mod p。

门限性质：任意 k 个参与者向量 v_i 构成 k×k Vandermonde 矩阵的行，
      因 x_i 互不相同且 p 为素数，行列式 det = Π_{j<i}(x_i - x_j) ≠ 0 (mod p)，
      矩阵可逆，线性方程组 V·s = y 有唯一解，可恢复 s（从而得到秘密 s_1）；
      少于 k 个参与者时方程数 < 未知数，解不唯一，s_1 无法确定。
"""
import random


def public_vectors(k, n, p):
    """生成 n 个公开参与者向量 v_i = (1, x_i, ..., x_i^{k-1})，x_i = 1..n。

    该表与秘密无关，可预计算（初始化开销单独统计）。
    """
    vs = []
    for i in range(1, n + 1):
        row = [pow(i, j, p) for j in range(k)]
        vs.append((i, row))
    return vs


def split_secret(secret, k, n, p, vectors=None):
    """把 secret 分成 n 份（向量内积方案）。

    返回 [(i, y_i)]，y_i = <s, v_i> mod p。
    """
    assert 2 <= k <= n
    assert 0 <= secret < p
    if vectors is None:
        vectors = public_vectors(k, n, p)
    # 秘密向量：第一分量为秘密，其余 k-1 个分量随机
    s_vec = [secret % p] + [random.randrange(p) for _ in range(k - 1)]
    shares = []
    for i, v in vectors:
        y = sum(s_vec[j] * v[j] for j in range(k)) % p
        shares.append((i, y))
    return shares


def solve_mod(aug, p):
    """模 p 高斯消元解 k 元线性方程组。

    aug: k×(k+1) 增广矩阵（每行 [A_i, b_i]）。
    返回解向量；无唯一解时返回 None。
    """
    m = len(aug)
    a = [row[:] for row in aug]
    for col in range(m):
        pivot = None
        for r in range(col, m):
            if a[r][col] % p != 0:
                pivot = r
                break
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inv = pow(a[col][col], p - 2, p)
        a[col] = [(v * inv) % p for v in a[col]]
        for r in range(m):
            if r != col and a[r][col] % p != 0:
                f = a[r][col]
                a[r] = [(a[r][c] - f * a[col][c]) % p for c in range(m + 1)]
    return [a[r][m] % p for r in range(m)]


def recover(shares, k, p):
    """由 k 份份额 (i, y_i) 解方程组 V·s = y，返回 s 的第一分量（秘密）。

    shares: [(x_i, y_i)]，长度须为 k。
    """
    k = len(shares)
    aug = []
    for i, y in shares:
        row = [pow(i, j, p) for j in range(k)] + [y % p]
        aug.append(row)
    sol = solve_mod(aug, p)
    if sol is None:
        return None
    return sol[0]
