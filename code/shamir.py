"""Shamir 秘密共享方案（有限域 GF(p)）。

分片：以秘密为常数项，随机选取 k-1 个系数构造 k-1 次多项式
      f(x) = a_0 + a_1 x + ... + a_{k-1} x^{k-1}，
      在 n 个互不相同的非零横坐标 x_i = 1..n 处取值得到 n 个份额。
恢复：任意 k 份份额通过拉格朗日插值恢复 f(0) = 秘密；
      少于 k 份时多项式不唯一，秘密无法确定。
"""
import random


def split_secret(secret, k, n, p):
    """把 secret 分成 n 份，任意 k 份可恢复。

    secret: 非负整数秘密；k: 门限；n: 份额总数；p: 素数模数。
    返回 [(x_i, y_i)]，x_i = 1..n。
    """
    assert 2 <= k <= n
    assert 0 <= secret < p
    # 常数项为秘密，其余 k-1 个系数在 GF(p) 上随机
    coeffs = [secret % p] + [random.randrange(p) for _ in range(k - 1)]
    shares = []
    for i in range(1, n + 1):
        y = 0
        for j, c in enumerate(coeffs):
            y = (y + c * pow(i, j, p)) % p
        shares.append((i, y))
    return shares


def recover(shares, k, p):
    """拉格朗日插值：由 k 份份额恢复 f(0) = 秘密。

    shares: [(x_i, y_i)]，长度须为 k；k 仅用于接口统一（实际取 len(shares)）。
    返回 GF(p) 上的秘密值。
    """
    k = len(shares)
    secret = 0
    for i in range(k):
        xi, yi = shares[i]
        num = 1
        den = 1
        for j in range(k):
            if j != i:
                xj, _ = shares[j]
                num = num * (-xj) % p
                den = den * (xi - xj) % p
        # 费马小定理求逆元
        secret = (secret + yi * num * pow(den, p - 2, p)) % p
    return secret
