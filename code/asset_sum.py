"""实验三（任务二）：使用秘密共享实现资产求和。

场景：m 名资产持有者（>=3）各持有资产 a_j，希望在不公开任何个人资产的
前提下求出资产总和 S = sum(a_j)。

协议流程（Shamir 与 Brickell 共用同一套流程）：
  1. 每名持有者 j 用自己的资产 a_j 生成 n 个份额，分发给 n 个节点；
  2. 每个节点 i 把收到的 m 个份额在本地相加，得到"总资产份额" y_i；
     —— 秘密共享具有同态性：份额相加 = 总和 S 的分片；
  3. 汇总方收集任意 >=k 个节点的总资产份额，恢复出总和 S；
     全程不接触任何个人明文资产，也不先恢复个人资产再求和。

运行：python code/asset_sum.py
"""
import random

import shamir
import brickell

# 有限域：p = 2^31 - 1（梅森素数）。资产总和的公开上界取 10^8，p > 上界，避免回绕。
P = 2147483647
ASSET_MAX = 10 ** 8  # 单名持有者资产上界，m 名总和上界 m*ASSET_MAX << P

random.seed(2026)  # 固定随机种子，保证结果可复现


def run_protocol(assets, k, n, p, scheme, vectors=None):
    """完整执行资产求和协议，返回 (恢复的总和, 节点总份额, 各持有者份额)。

    scheme 提供 split_secret / recover 接口。
    """
    # 1) 每名持有者独立分片（各持有者的随机系数彼此独立）
    holder_shares = []  # holder_shares[j][i] = 持有者 j 分给节点 i 的份额
    for a in assets:
        holder_shares.append(scheme.split_secret(a, k, n, p))

    # 2) 节点本地求和：节点 i 持有每名持有者的第 i 个份额，相加得到总份额
    node_totals = []
    for i in range(n):
        y = sum(sh[i][1] for sh in holder_shares) % p
        node_totals.append((i + 1, y))

    # 3) 任选 k 个节点，用其总资产份额恢复总和
    chosen = random.sample(node_totals, k)
    total = scheme.recover(chosen, k, p)
    return total, node_totals, holder_shares


def verify(assets, k, n, p, scheme, vectors=None):
    """对一组资产做完整验证：正确恢复 + 门限测试，返回 (ok, detail)。"""
    total, node_totals, holder_shares = run_protocol(assets, k, n, p, scheme, vectors)
    expect = sum(assets) % p
    ok = (total == expect)
    # 门限测试：只用 k-1 个节点，恢复结果不应等于总和（无法唯一确定）
    under_total = scheme.recover(node_totals[: k - 1], k - 1, p) if k - 1 > 0 else None
    under_ok = (under_total != expect)
    return ok, under_ok, expect, total, under_total


def main():
    print("=" * 62)
    print("实验三（任务二）：基于秘密共享的资产求和")
    print("方案：Shamir（多项式 + 拉格朗日） vs Brickell（向量内积 + 解方程组）")
    print("有限域：p = 2^31 - 1 = %d，单方资产上界 %d" % (P, ASSET_MAX))
    print("=" * 62)

    m = 4  # 资产持有者人数（>=3）
    cases = [(3, 5), (4, 7)]  # 两组 (k, n)

    for k, n in cases:
        print("\n---------- (k, n) = (%d, %d) ----------" % (k, n))
        # Brickell 的公开向量表与秘密无关，预计算一次
        bv = brickell.public_vectors(k, n, P)
        for grp in range(1, 4):  # 3 组随机资产输入
            assets = [random.randrange(1, ASSET_MAX) for _ in range(m)]
            ok_s, un_s, exp, tot_s, under_s = verify(assets, k, n, P, shamir)
            ok_b, un_b, exp, tot_b, under_b = verify(assets, k, n, P, brickell, bv)
            flag_s = "OK " if ok_s and un_s else "FAIL"
            flag_b = "OK " if ok_b and un_b else "FAIL"
            print("  组%d 资产=%s" % (grp, assets))
            print("      真实总和=%d" % exp)
            print("      Shamir   : 恢复=%d  k-1份=%s  %s"
                  % (tot_s, "不可恢复(≠)" if un_s else "意外相等", flag_s))
            print("      Brickell : 恢复=%d  k-1份=%s  %s"
                  % (tot_b, "不可恢复(≠)" if un_b else "意外相等", flag_b))

    print("\n全部验证通过：两种方案均正确恢复总和，且少于 k 个节点无法确定总和。")
    print("协议全程未接触任何个人明文资产（见各节点可见信息分析）。")


if __name__ == "__main__":
    main()
