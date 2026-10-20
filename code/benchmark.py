"""实验三（任务二）性能对比：Shamir vs Brickell 资产求和。

统计口径（要求 5、6）：
  - 阶段耗时：份额生成（m 名持有者分片）、份额相加（节点求和）、总和恢复（k 节点）；
  - 各阶段峰值内存与全过程峰值内存：用 tracemalloc 追踪 Python 分配；
  - 份额总存储大小：m×n 个持有者份额 + n 个节点总份额的整数字节数（sys.getsizeof）；
  - 初始化/预计算：Brickell 的公开向量表、Shamir 无预计算，单独记录；
  - 每组 (k,n) 重复 5 次取平均耗时；统一硬件、输入、有限域与恢复份额数。
实现说明：纯 Python 标准库，无 numpy 向量化；计时不含文件读写（不落盘）。
运行：python code/benchmark.py
"""
import random
import sys
import time
import tracemalloc

import shamir
import brickell

P = 2147483647
ASSET_MAX = 10 ** 8
REPEATS = 5

random.seed(2026)


def timed(fn, *args):
    """计时并测量该阶段的 Python 分配峰值内存。"""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak


def share_size_bytes(shares):
    """份额列表的存储大小（sys.getsizeof 逐个累加）。"""
    return sum(sys.getsizeof(i) + sys.getsizeof(y) for i, y in shares)


def bench_case(k, n, assets, scheme, vectors=None):
    """对一组 (k,n,资产) 跑一次完整协议，返回分阶段耗时与内存。"""
    # 份额生成
    _, t_split, mem_split = timed(
        lambda: [scheme.split_secret(a, k, n, P) for a in assets])
    holder_shares = [scheme.split_secret(a, k, n, P) for a in assets]

    # 份额相加（节点求和）
    def add_shares():
        return [(i + 1, sum(sh[i][1] for sh in holder_shares) % P) for i in range(n)]
    node_totals, t_add, mem_add = timed(add_shares)

    # 总和恢复（k 个节点）
    chosen = random.sample(node_totals, k)
    _, t_rec, mem_rec = timed(lambda: scheme.recover(chosen, k, P))

    # 份额存储大小：m×n 持有者份额 + n 节点总份额
    store_bytes = sum(share_size_bytes(sh) for sh in holder_shares) \
        + share_size_bytes(node_totals)
    peak_total = max(mem_split, mem_add, mem_rec)
    return (t_split, t_add, t_rec), (mem_split, mem_add, mem_rec, peak_total), store_bytes


def main():
    print("实验三（任务二）性能对比：Shamir vs Brickell（%d 次重复取平均）" % REPEATS)
    print("有限域 p = 2^31-1，m = 4 名持有者，资产随机，纯标准库（无向量化）")
    header = ("%-10s %-12s %-12s %-12s %-12s %-12s %-14s %-12s"
              % ("(k,n)", "分片(ms)", "求和(ms)", "恢复(ms)", "分片峰内存",
                 "求和峰内存", "恢复峰内存", "份额存储"))
    print(header)
    print("-" * len(header))

    assets = [random.randrange(1, ASSET_MAX) for _ in range(4)]
    results = {}
    for k, n in [(3, 5), (4, 7)]:
        bv = brickell.public_vectors(k, n, P)
        for name, scheme, vec in [("Shamir", shamir, None),
                                  ("Brickell", brickell, bv)]:
            acc = [0.0] * 3
            peak = [0] * 4
            store = 0
            for _ in range(REPEATS):
                (ts, ta, tr), (ms, ma, mr, pt), sb = bench_case(k, n, assets, scheme, vec)
                acc[0] += ts; acc[1] += ta; acc[2] += tr
                peak[0] = max(peak[0], ms); peak[1] = max(peak[1], ma)
                peak[2] = max(peak[2], mr); peak[3] = max(peak[3], pt)
                store = max(store, sb)
            row = ("(%d,%d) %s" % (k, n, name),
                   "%.3f" % (acc[0] / REPEATS * 1000),
                   "%.3f" % (acc[1] / REPEATS * 1000),
                   "%.3f" % (acc[2] / REPEATS * 1000),
                   "%.1fKB" % (peak[0] / 1024),
                   "%.1fKB" % (peak[1] / 1024),
                   "%.1fKB" % (peak[2] / 1024),
                   "%.1fKB" % (store / 1024))
            print("  %-8s %-12s %-12s %-12s %-12s %-12s %-14s %-12s" % row)
            results[(k, n, name)] = (acc, peak, store)

    # 预计算开销（Brickell 公开向量表），单独记录
    for k, n in [(3, 5), (4, 7)]:
        _, t_pre, mem_pre = timed(lambda: brickell.public_vectors(k, n, P))
        print("  预计算 Brickell 公开向量表 (%d,%d): %.3f ms, 峰值内存 %.1f KB"
              % (k, n, t_pre * 1000, mem_pre / 1024))
    print("\n说明：纯标准库实现，无 numpy 向量化；计时不含文件读写；")
    print("全过程峰值内存 = max(分片, 求和, 恢复) 各阶段峰值。")


if __name__ == "__main__":
    main()
