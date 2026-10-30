"""Computer science, systems, security, ML and applied-computing families."""

from __future__ import annotations

import heapq
import ipaddress
import itertools
import math
import sqlite3
from collections import Counter, deque
from fractions import Fraction

from .generators import Q, family, fmt, scale, sci, yesno

SIG = "(3 significant figures)"


# ===================================================================== fundamentals

@family("cs.binary", "Binary, hexadecimal and data sizes", {"cs": (1, 6)},
        "Binary uses powers of two: 1011₂ = 8 + 2 + 1 = 11. Hexadecimal uses 16 symbols 0–9, A–F (A = 10 … F = 15); each hex digit is 4 bits. 1 byte = 8 bits; 1 KiB = 1024 bytes. "
        "In n-bit two's complement, −x is written as 2ⁿ − x. ASCII: 'A' = 65, 'a' = 97, '0' = 48.",
        "Reading binary digits left to right as if they were decimal.")
def binary(rng, t):
    kinds = ["b2d", "d2b", "bytes", "ascii"] + (["hex", "twos"] if t > .3 else [])
    k = rng.choice(kinds)
    n = rng.randint(3, 2 ** scale(t, 4, 8) - 1)
    if k == "b2d":
        return Q("numeric", f"Convert the binary number {bin(n)[2:]} to decimal.", n, f"{bin(n)[2:]}₂ = {n}")
    if k == "d2b":
        return Q("cloze", f"Write {n} in binary (digits only).", bin(n)[2:], f"{n} = {bin(n)[2:]}₂", exact=True)
    if k == "bytes":
        kib = rng.randint(1, 16)
        return Q("numeric", f"How many bits are in {kib} KiB (1 KiB = 1024 bytes)?", kib * 1024 * 8, f"{kib}×1024×8")
    if k == "ascii":
        ch = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return Q("numeric", f"What is the ASCII code (decimal) of the character '{ch}'? (A = 65, a = 97)", ord(ch), "count from 'A' or 'a'")
    if k == "hex":
        return Q("numeric", f"Convert the hexadecimal number {n:X} to decimal.", n, f"{n:X}₁₆ = {n}")
    x = rng.randint(1, 100)
    return Q("cloze", f"Write −{x} as an 8-bit two's complement binary number (8 digits).", format(256 - x, "08b"), f"256 − {x} = {256 - x} = {format(256 - x, '08b')}", exact=True)


@family("cs.bits", "Bitwise operations", {"cs": (6, 9), "cybersecurity": (3, 6)},
        "AND (&) keeps bits set in both; OR (|) keeps bits set in either; XOR (^) keeps bits that differ. x << k multiplies by 2ᵏ; x >> k divides by 2ᵏ (floor). XOR with the same value twice gives back the original.",
        "Mixing up & and |.")
def bits(rng, t):
    a, b, k = rng.randint(1, 255), rng.randint(1, 255), rng.randint(1, 4)
    op = rng.choice(["&", "|", "^", "<<", ">>"])
    v = {"&": a & b, "|": a | b, "^": a ^ b, "<<": a << k, ">>": a >> k}[op]
    rhs = k if op in ("<<", ">>") else b
    return Q("numeric", f"What is {a} {op} {rhs} in Python (decimal result)?", v, f"{a:08b} {op} {rhs if op in ('<<', '>>') else format(b, '08b')}")


@family("cs.complexity", "Counting steps and Big-O", {"cs": (5, 9), "algorithms": (1, 6)},
        "Count how many times the innermost statement runs. A single loop to n → n. A loop inside a loop → n². An inner loop that runs i times → 1 + 2 + … + (n−1) = n(n−1)/2. "
        "A loop that doubles its counter runs log₂ n times. Binary search on n sorted items needs at most ⌊log₂ n⌋ + 1 comparisons. If the cost grows like n², tripling n multiplies the time by 9.",
        "Treating n(n−1)/2 as n.")
def complexity(rng, t):
    k = rng.random()
    if k < .3:
        n = rng.randint(4, 30)
        return Q("numeric", f"How many times does `count += 1` execute?\n```python\nfor i in range({n}):\n    for j in range(i):\n        count += 1\n```", n * (n - 1) // 2, "0 + 1 + … + (n−1)")
    if k < .55:
        e = rng.randint(3, 12)
        return Q("numeric", f"How many times does the loop body run?\n```python\ni = 1\nwhile i < {2 ** e}:\n    i *= 2\n```", e, f"i takes the values 1, 2, 4, … up to 2^{e - 1}")
    if k < .8:
        n = rng.choice([7, 15, 31, 63, 127, 255, 1023])
        return Q("numeric", f"What is the maximum number of comparisons binary search needs on a sorted list of {n} items?", int(math.log2(n + 1)), "⌊log₂ n⌋ + 1")
    n0, tt, m = rng.choice([1000, 500, 100]), rng.choice([2, 4, 5]), rng.choice([2, 3, 4])
    return Q("numeric", f"An O(n²) algorithm takes {tt} s for n = {n0}. About how long (s) will n = {n0 * m} take?", tt * m * m, "time ∝ n²")


@family("cs.arch", "Computer architecture", {"cs": (4, 8)},
        "n address bits can name 2ⁿ bytes. Average memory access time AMAT = hit time + miss rate × miss penalty. CPU time = instructions × CPI ÷ clock rate.",
        "Using the hit rate where the formula needs the miss rate.")
def arch(rng, t):
    k = rng.random()
    if k < .3:
        b = rng.choice([10, 16, 20, 30, 32])
        return Q("numeric", f"How many bytes can be addressed with {b} address bits?", 2 ** b, "2ⁿ")
    if k < .65:
        h, m, p = rng.choice([1, 2, 4]), rng.choice([0.02, 0.05, 0.1]), rng.choice([50, 100, 200])
        return Q("numeric", f"A cache has hit time {h} ns, miss rate {m:.0%} and miss penalty {p} ns. What is the AMAT (ns)? {SIG}", fmt(h + m * p), "hit + miss × penalty")
    ins, cpi, ghz = rng.choice([1e9, 2e9, 5e9]), rng.choice([1, 1.5, 2]), rng.choice([2, 3, 4])
    return Q("numeric", f"A program runs {sci(ins, 1)} instructions at CPI {cpi} on a {ghz} GHz CPU. How long does it take (s)? {SIG}", fmt(ins * cpi / (ghz * 1e9)), "instructions × CPI ÷ clock")


@family("cs.os", "Operating systems", {"operating_systems": (1, 12), "cs": (6, 8)},
        "FCFS runs processes in arrival order; shortest-job-first (SJF) runs the shortest burst first. A process's waiting time = the time before it starts (when all arrive at time 0). "
        "Paging: a page of 2ᵏ bytes has a k-bit offset; the page number uses the remaining address bits. Page replacement counts faults when a needed page is not in memory.",
        "Counting the process's own burst time as waiting.")
def os_(rng, t):
    k = rng.random()
    if k < .35:
        bursts = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
        sjf = rng.random() < .5
        order = sorted(bursts) if sjf else bursts
        wait = [sum(order[:i]) for i in range(len(order))]
        return Q("numeric", f"Processes with burst times {bursts} (all arrive at time 0) run {'shortest-job-first' if sjf else 'first-come-first-served in the listed order'}. What is the average waiting time? (2 decimals)",
                 f"{sum(wait) / len(wait):.2f}", f"waits: {wait}")
    if k < .6:
        a, p = rng.choice([32, 48, 64]), rng.choice([10, 12, 13, 16])
        if rng.random() < .5:
            return Q("numeric", f"A system uses {a}-bit virtual addresses and {2 ** p}-byte pages. How many bits are in the page number?", a - p, f"{a} − {p}")
        return Q("numeric", f"What is the page offset size in bits for {2 ** p}-byte pages?", p, "log₂(page size)")
    refs = [rng.randint(1, 5) for _ in range(rng.randint(8, 12))]
    frames = rng.choice([3, 3, 4])
    lru = rng.random() < .5
    mem, faults = [], 0
    for r in refs:
        if r in mem:
            if lru:
                mem.remove(r)
                mem.append(r)
        else:
            faults += 1
            if len(mem) == frames:
                mem.pop(0)
            mem.append(r)
    return Q("numeric", f"With {frames} frames and {'LRU' if lru else 'FIFO'} replacement, how many page faults occur for the reference string {refs}?", faults, "simulate")


@family("cs.network", "Computer networks", {"networking": (1, 12), "cs": (7, 9)},
        "A /n IPv4 subnet has 2^(32−n) addresses, of which 2^(32−n) − 2 are usable hosts. The network address is the address with the host bits set to 0. Transfer time = bits ÷ bandwidth. "
        "Bandwidth–delay product = bandwidth × round-trip time. In TCP slow start the congestion window doubles every round trip.",
        "Forgetting that a byte is 8 bits when converting file sizes to bits.")
def network(rng, t):
    k = rng.random()
    if k < .25:
        n = rng.randint(20, 30)
        return Q("numeric", f"How many usable host addresses does a /{n} IPv4 subnet have?", 2 ** (32 - n) - 2, "2^(32−n) − 2")
    if k < .5:
        o = rng.choice([64, 128, 192])
        n = rng.choice([25, 26, 27])
        ip = f"192.168.{rng.randint(0, 5)}.{rng.randint(1, 250)}"
        net = ipaddress.ip_network(f"{ip}/{n}", strict=False)
        return Q("cloze", f"What is the network address of {ip}/{n}? (dotted decimal)", str(net.network_address), "zero the host bits", exact=True)
    if k < .75:
        mb, mbps = rng.choice([1, 5, 10, 100]), rng.choice([8, 10, 100, 1000])
        return Q("numeric", f"How many seconds does it take to send a {mb} MB file over a {mbps} Mbit/s link? (1 MB = 8 Mbit; ignore overhead) {SIG}", fmt(mb * 8 / mbps), "size in bits ÷ rate")
    rtts = rng.randint(2, 8)
    return Q("numeric", f"In TCP slow start the window begins at 1 segment and doubles each round trip. How many segments is it after {rtts} round trips?", 2 ** rtts, "2^k")


@family("cs.sql", "SQL queries", {"databases": (1, 12), "data_science": (3, 8)},
        "SELECT picks columns, WHERE filters rows, GROUP BY makes one row per group (use COUNT, SUM, AVG, MAX, MIN per group), HAVING filters groups, JOIN combines tables on a key, and a subquery in brackets can supply a value such as the average.",
        "Using WHERE to filter on an aggregate (use HAVING).")
def sql(rng, t):
    depts = ["Eng", "Ops", "Sales"]
    rows = [(f"E{i}", rng.choice(depts), rng.randint(3, 9) * 10, rng.randint(22, 55)) for i in range(1, rng.randint(8, 10))]
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE emp(name TEXT, dept TEXT, salary INT, age INT)")
    db.executemany("INSERT INTO emp VALUES (?,?,?,?)", rows)
    db.execute("CREATE TABLE dept(dept TEXT, budget INT)")
    budgets = [(d, rng.randint(1, 5) * 100) for d in depts]
    db.executemany("INSERT INTO dept VALUES (?,?)", budgets)
    d, s, a = rng.choice(sorted({r[1] for r in rows})), rng.randint(4, 7) * 10, rng.randint(30, 45)
    qs = [
        (0, f"SELECT COUNT(*) FROM emp WHERE dept = '{d}'"),
        (0, f"SELECT COUNT(*) FROM emp WHERE salary >= {s}"),
        (1, f"SELECT SUM(salary) FROM emp WHERE dept = '{d}'"),
        (1, "SELECT MAX(salary) - MIN(salary) FROM emp"),
        (1, f"SELECT COUNT(*) FROM emp WHERE age > {a} AND salary > {s}"),
        (2, "SELECT COUNT(DISTINCT dept) FROM emp WHERE salary > 40"),
        (2, "SELECT COUNT(*) FROM emp WHERE salary > (SELECT AVG(salary) FROM emp)"),
        (3, "SELECT COUNT(*) FROM (SELECT dept FROM emp GROUP BY dept HAVING COUNT(*) >= 3)"),
        (3, f"SELECT SUM(e.salary) FROM emp e JOIN dept d ON e.dept = d.dept WHERE d.budget >= {rng.randint(1, 4) * 100}"),
        (4, "SELECT MAX(c) FROM (SELECT COUNT(*) AS c FROM emp GROUP BY dept)"),
        (4, "SELECT COUNT(*) FROM emp e WHERE salary = (SELECT MAX(salary) FROM emp WHERE dept = e.dept)"),
    ]
    pool = [q for lvl, q in qs if lvl <= int(t * 4) + 1]
    q = rng.choice(pool)
    val = db.execute(q).fetchone()[0]
    table = "emp(name, dept, salary, age): " + "; ".join(f"({a_},{b_},{c_},{d_})" for a_, b_, c_, d_ in rows)
    table += "\ndept(dept, budget): " + "; ".join(f"({a_},{b_})" for a_, b_ in budgets)
    val = 0 if val is None else val
    return Q("numeric", f"Tables:\n{table}\nWhat single number does this query return?\n```sql\n{q}\n```", int(val), "run the query by hand")


@family("cs.automata", "Automata and formal languages", {"cs": (8, 11)},
        "A DFA reads a string one symbol at a time, moving between states; it accepts if it ends in an accepting state. The number of binary strings of length n with no two adjacent 1s follows Fibonacci: 2, 3, 5, 8, 13, …",
        "Forgetting the empty string.")
def automata(rng, t):
    if rng.random() < .5:
        n = rng.randint(3, 10)
        cnt = sum("11" not in "".join(p) for p in itertools.product("01", repeat=n))
        return Q("numeric", f"How many binary strings of length {n} contain no two consecutive 1s?", cnt, "Fibonacci numbers")
    s = "".join(rng.choice("01") for _ in range(rng.randint(4, 8)))
    kind = rng.choice(["even ones", "ends with 01", "contains 00", "value divisible by 3"])
    ok = {"even ones": s.count("1") % 2 == 0, "ends with 01": s.endswith("01"), "contains 00": "00" in s, "value divisible by 3": int(s, 2) % 3 == 0}[kind]
    return yesno(rng, f"A DFA accepts exactly the binary strings with: {kind}. Does it accept '{s}'?", ok, "simulate the DFA")


@family("cs.info", "Information theory and coding", {"cs": (9, 12), "cybersecurity": (6, 9)},
        "Entropy H = −Σ pᵢ log₂ pᵢ bits. Huffman coding merges the two least likely symbols repeatedly; the average code length is Σ pᵢ·lenᵢ. Hamming distance = number of differing bit positions. "
        "A Hamming code for m data bits needs r parity bits with 2ʳ ≥ m + r + 1.",
        "Using natural log instead of log base 2.")
def info(rng, t):
    k = rng.random()
    if k < .3:
        ps = rng.choice([[0.5, 0.25, 0.25], [0.5, 0.25, 0.125, 0.125], [0.25] * 4, [0.7, 0.1, 0.1, 0.1]])
        return Q("numeric", f"What is the entropy (bits) of a source with probabilities {ps}? {SIG}", fmt(-sum(p * math.log2(p) for p in ps)), "−Σ p log₂ p")
    if k < .55:
        ps = rng.choice([[0.4, 0.3, 0.2, 0.1], [0.5, 0.25, 0.15, 0.1], [0.35, 0.35, 0.2, 0.1]])
        h = list(ps)
        heapq.heapify(h)
        total = 0.0
        while len(h) > 1:
            a, b = heapq.heappop(h), heapq.heappop(h)
            total += a + b
            heapq.heappush(h, a + b)
        return Q("numeric", f"A Huffman code is built for symbol probabilities {ps}. What is its average code length (bits/symbol)? {SIG}", fmt(total), "sum of merged probabilities")
    if k < .75:
        a = "".join(rng.choice("01") for _ in range(8))
        b = "".join(rng.choice("01") for _ in range(8))
        return Q("numeric", f"What is the Hamming distance between {a} and {b}?", sum(x != y for x, y in zip(a, b)), "count differing positions")
    p, q = rng.choice([(11, 13), (17, 19), (23, 29)])
    n, phi = p * q, (p - 1) * (q - 1)
    e = rng.choice([x for x in (3, 5, 7, 11, 13) if math.gcd(x, phi) == 1])
    m = rng.randint(2, 40)
    if rng.random() < .5:
        return Q("numeric", f"Textbook RSA with p = {p}, q = {q}, e = {e}. What is the ciphertext of m = {m}? (c = m^e mod n)", pow(m, e, n), f"n = {n}")
    return Q("numeric", f"Textbook RSA with p = {p}, q = {q}, e = {e}. What is the private exponent d (e·d ≡ 1 mod φ(n))?", pow(e, -1, phi), f"φ = {phi}")


@family("cs.security", "Security fundamentals (defensive)", {"cybersecurity": (1, 12)},
        "A password from an alphabet of N symbols and length L has N^L possibilities and L·log₂N bits of entropy — longer beats fancier. A Caesar cipher shifts each letter by k (decrypt by shifting back). "
        "Diffie–Hellman: both sides compute g^(ab) mod p. The birthday bound: collisions among n-bit hashes become likely (50%) after about 1.18·2^(n/2) values.",
        "Believing a complex short password beats a long one.")
def security(rng, t):
    k = rng.random()
    if k < .3:
        word = rng.choice(["secure", "network", "password", "defense", "firewall", "protect", "patch", "update", "encrypt"])
        shift = rng.randint(1, 25)
        enc = "".join(chr((ord(c) - 97 + shift) % 26 + 97) for c in word)
        return Q("cloze", f"A Caesar cipher shifted every letter forward by {shift}. The ciphertext is '{enc}'. What is the plaintext word?", word, f"shift each letter back by {shift}")
    if k < .55:
        n, L = rng.choice([(26, 8), (62, 8), (94, 10), (10, 6), (26, 12)])
        return Q("numeric", f"How many bits of entropy does a random {L}-character password have if each character comes from {n} possible symbols? {SIG}", fmt(L * math.log2(n)), "L·log₂N")
    if k < .75:
        p, g, a, b = rng.choice([(23, 5), (29, 2), (37, 5)]) + (rng.randint(2, 10), rng.randint(2, 10))
        return Q("numeric", f"Diffie–Hellman with p = {p}, g = {g}; Alice's secret is {a}, Bob's is {b}. What shared key do they compute?", pow(pow(g, a, p), b, p), "g^(ab) mod p")
    bitsn = rng.choice([32, 48, 64, 128])
    return Q("numeric", f"About how many random {bitsn}-bit hash values must be generated before a collision becomes 50% likely? (use 1.18·2^(n/2)) {SIG}", fmt(1.18 * 2 ** (bitsn / 2)), "birthday bound")


# ===================================================================== data structures

@family("cs.structures", "Stacks, queues, heaps, trees and hashing", {"data_structures": (1, 12)},
        "Stack: last in, first out (push/pop at the top). Queue: first in, first out. A complete binary tree of height h has at most 2^(h+1) − 1 nodes. In a 0-indexed array heap, the children of i are 2i + 1 and 2i + 2 and the parent is (i − 1)//2. "
        "A BST inserts smaller keys left and larger keys right; its height is the number of edges on the longest root-to-leaf path.",
        "Counting nodes instead of edges for the height.")
def structures(rng, t):
    k = rng.random()
    if k < .25:
        ops, st = [], []
        for _ in range(rng.randint(5, 9)):
            if st and rng.random() < .4:
                st.pop()
                ops.append("pop")
            else:
                v = rng.randint(1, 9)
                st.append(v)
                ops.append(f"push {v}")
        if not st:
            st.append(1)
            ops.append("push 1")
        return Q("numeric", f"A stack starts empty. Operations: {', '.join(ops)}. What is the top element afterwards?", st[-1], f"stack = {st}")
    if k < .45:
        h = rng.randint(2, 8)
        return Q("numeric", f"What is the maximum number of nodes in a binary tree of height {h} (a single node has height 0)?", 2 ** (h + 1) - 1, "2^(h+1) − 1")
    if k < .6:
        i = rng.randint(3, 20)
        return Q("numeric", f"In a 0-indexed array-based binary heap, what is the index of the left child of the node at index {i}?", 2 * i + 1, "2i + 1")
    if k < .85:
        keys = rng.sample(range(1, 30), rng.randint(5, 8))
        def insert(root, key):
            if root is None:
                return [key, None, None]
            side = 1 if key < root[0] else 2
            root[side] = insert(root[side], key)
            return root

        def height(n):
            return -1 if n is None else 1 + max(height(n[1]), height(n[2]))

        root = None
        for x in keys:
            root = insert(root, x)
        return Q("numeric", f"Keys {keys} are inserted in order into an empty binary search tree. What is the tree's height (edges on the longest path)?", height(root), "build the BST")
    m = rng.choice([5, 7, 10])
    keys = [rng.randint(10, 99) for _ in range(rng.randint(5, 8))]
    cnt = Counter(x % m for x in keys)
    return Q("numeric", f"Keys {keys} are stored in a hash table with {m} buckets using key mod {m}. What is the largest number of keys in any one bucket?", max(cnt.values()), "count per bucket")


@family("cs.graphs", "Graph algorithms by hand", {"data_structures": (5, 12), "algorithms": (3, 12), "discrete_math": (7, 11)},
        "BFS visits nodes level by level using a queue; DFS goes deep using a stack. Dijkstra picks the closest unvisited node each time. Kruskal's MST sorts edges by weight and adds each edge that does not create a cycle.",
        "Adding an edge that closes a cycle in Kruskal's algorithm.")
def graphs(rng, t):
    n = rng.randint(5, 6)
    edges = set()
    nodes = list(range(n))
    for i in range(1, n):
        edges.add((rng.randrange(i), i, rng.randint(1, 9)))
    for _ in range(rng.randint(2, 4)):
        a, b = sorted(rng.sample(nodes, 2))
        if not any((x, y) == (a, b) for x, y, _ in edges):
            edges.add((a, b, rng.randint(1, 9)))
    edges = sorted(edges)
    lab = "ABCDEF"
    txt = ", ".join(f"{lab[a]}–{lab[b]}:{w}" for a, b, w in edges)
    k = rng.random()
    if k < .4:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        total = 0
        for a, b, w in sorted(edges, key=lambda e: e[2]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                total += w
        return Q("numeric", f"An undirected weighted graph has edges {txt}. What is the total weight of its minimum spanning tree?", total, "Kruskal")
    if k < .75:
        adj = {i: [] for i in nodes}
        for a, b, w in edges:
            adj[a].append((b, w))
            adj[b].append((a, w))
        dist, pq = {}, [(0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in dist:
                continue
            dist[u] = d
            for v, w in adj[u]:
                heapq.heappush(pq, (d + w, v))
        goal = n - 1
        return Q("numeric", f"An undirected weighted graph has edges {txt}. What is the shortest distance from A to {lab[goal]}?", dist[goal], "Dijkstra")
    adj = {i: [] for i in nodes}
    for a, b, _ in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, order, q = {0}, [], deque([0])
    while q:
        u = q.popleft()
        order.append(lab[u])
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return Q("cloze", f"An undirected graph has edges {txt} (weights ignored). Give the BFS visiting order from A, always visiting neighbours in alphabetical order (letters only, e.g. ABCD).", "".join(order), "queue-based BFS", exact=True)


# ===================================================================== machine learning & AI

@family("ml.metrics", "Evaluating classifiers", {"machine_learning": (1, 6), "data_science": (5, 12), "ai_safety": (1, 12)},
        "From a confusion matrix: accuracy = (TP+TN)/all; precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R). Overfitting means doing well on training data but poorly on new data — always measure on held-out data.",
        "Quoting accuracy on imbalanced data.")
def ml_metrics(rng, t):
    tp, fp, fn, tn = rng.randint(10, 80), rng.randint(2, 30), rng.randint(2, 30), rng.randint(20, 100)
    k = rng.choice(["accuracy", "precision", "recall", "F1"])
    p, r = tp / (tp + fp), tp / (tp + fn)
    v = {"accuracy": (tp + tn) / (tp + tn + fp + fn), "precision": p, "recall": r, "F1": 2 * p * r / (p + r)}[k]
    return Q("numeric", f"A classifier has TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}. What is its {k}? (3 decimals)", f"{v:.3f}", k)


@family("ml.optim", "Optimisation and linear models", {"machine_learning": (3, 9), "optimization": (1, 5), "deep_learning": (1, 4)},
        "Gradient descent: w ← w − η·∇L(w). For L = (w − a)², the gradient is 2(w − a), so each step multiplies the distance from a by (1 − 2η). Linear model ŷ = w·x + b; MSE = mean of squared errors. Sigmoid σ(z) = 1/(1 + e^(−z)).",
        "Using a learning rate that is too large so the steps overshoot.")
def ml_optim(rng, t):
    k = rng.random()
    if k < .4:
        a, w0, eta, steps = rng.randint(1, 5), rng.randint(6, 12), rng.choice([0.1, 0.2, 0.25, 0.4]), rng.randint(1, 4)
        w = w0
        for _ in range(steps):
            w -= eta * 2 * (w - a)
        return Q("numeric", f"Minimise L(w) = (w − {a})² by gradient descent from w = {w0} with learning rate {eta}. What is w after {steps} step(s)? {SIG}", fmt(w), "w ← w − η·2(w − a)")
    if k < .65:
        z = rng.choice([-2, -1, 0, 1, 2, 3])
        return Q("numeric", f"What is the sigmoid σ(z) = 1/(1 + e^(−z)) at z = {z}? (3 decimals)", f"{1 / (1 + math.exp(-z)):.3f}", "formula")
    ys = [rng.randint(1, 9) for _ in range(4)]
    ps = [y + rng.randint(-3, 3) for y in ys]
    return Q("numeric", f"Targets {ys} and predictions {ps}. What is the mean squared error? (2 decimals)", f"{sum((a - b) ** 2 for a, b in zip(ys, ps)) / 4:.2f}", "mean of squared differences")


@family("ml.nn", "Neural networks", {"deep_learning": (3, 12), "machine_learning": (8, 12), "computer_vision": (5, 9)},
        "A dense layer with n inputs and m outputs has (n + 1)·m parameters (weights + biases). ReLU(x) = max(0, x). Softmax pᵢ = e^zᵢ / Σe^zⱼ; cross-entropy loss = −ln p_correct. "
        "A conv layer with k×k kernels, c_in input and c_out output channels has (k·k·c_in + 1)·c_out parameters. Attention weights = softmax(q·kᵢ/√d).",
        "Forgetting the bias terms in parameter counts.")
def ml_nn(rng, t):
    k = rng.random()
    if k < .25:
        n, m = rng.choice([784, 100, 256, 64]), rng.choice([10, 32, 128])
        return Q("numeric", f"How many parameters (weights and biases) does a dense layer from {n} inputs to {m} outputs have?", (n + 1) * m, "(n+1)m")
    if k < .45:
        kk, ci, co = rng.choice([3, 5]), rng.choice([1, 3, 16]), rng.choice([8, 16, 64])
        return Q("numeric", f"How many parameters does a conv layer with {kk}×{kk} kernels, {ci} input channels and {co} output channels have (with biases)?", (kk * kk * ci + 1) * co, "(k²·c_in + 1)·c_out")
    if k < .65:
        z = [rng.randint(0, 3) for _ in range(3)]
        p = math.exp(z[0]) / sum(math.exp(x) for x in z)
        return Q("numeric", f"Logits are {z}. What is the softmax probability of the first class? (3 decimals)", f"{p:.3f}", "e^z₀ / Σe^z")
    if k < .8:
        x = [rng.randint(-3, 3) for _ in range(2)]
        w1, w2 = [[rng.randint(-2, 2) for _ in range(2)] for _ in range(2)], [rng.randint(-2, 2) for _ in range(2)]
        h = [max(0, sum(w * xi for w, xi in zip(row, x))) for row in w1]
        out = sum(a * b for a, b in zip(w2, h))
        return Q("numeric", f"A network has input x = {x}, hidden weights W₁ = {w1} (row i gives hidden unit i, no bias) with ReLU, and output weights {w2} (no bias). What is the output?", out, f"hidden = {h}")
    d = rng.choice([1, 4, 16])
    s = [rng.randint(-2, 3) for _ in range(2)]
    e = [math.exp(v / math.sqrt(d)) for v in s]
    return Q("numeric", f"Query·key dot products for two keys are {s} and d = {d}. What is the attention weight on the first key after scaling by 1/√d and softmax? (3 decimals)", f"{e[0] / sum(e):.3f}", "softmax(s/√d)")


@family("rl.basics", "Reinforcement learning", {"reinforcement_learning": (1, 12)},
        "Discounted return G = r₀ + γr₁ + γ²r₂ + …. Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s′,·) − Q(s,a)]. With ε-greedy, the greedy action is chosen with probability 1 − ε + ε/|A|.",
        "Forgetting to discount future rewards.")
def rl_basics(rng, t):
    k = rng.random()
    if k < .35:
        g, rs = rng.choice([0.5, 0.9, 0.8]), [rng.randint(0, 10) for _ in range(rng.randint(3, 5))]
        return Q("numeric", f"With discount γ = {g}, rewards {rs} are received at steps 0, 1, 2, …. What is the discounted return? {SIG}", fmt(sum(r * g ** i for i, r in enumerate(rs))), "Σ γⁱ rᵢ")
    if k < .7:
        q, r, qn, a, g = rng.randint(0, 10), rng.randint(0, 10), rng.randint(0, 10), rng.choice([0.1, 0.5]), rng.choice([0.9, 0.5])
        return Q("numeric", f"Q(s,a) = {q}, reward r = {r}, max Q(s′,·) = {qn}, α = {a}, γ = {g}. What is the updated Q(s,a)? {SIG}", fmt(q + a * (r + g * qn - q)), "Q-learning update")
    eps, na = rng.choice([0.1, 0.2, 0.4]), rng.choice([2, 4, 5])
    return Q("numeric", f"An ε-greedy policy with ε = {eps} and {na} actions picks the greedy action with what probability? (3 decimals)", f"{1 - eps + eps / na:.3f}", "1 − ε + ε/|A|")


@family("nlp.basics", "Language-model arithmetic", {"nlp": (1, 12), "data_science": (6, 9)},
        "Bigram probability P(w₂|w₁) = count(w₁ w₂)/count(w₁). Cosine similarity = (u·v)/(|u||v|). Perplexity of a uniform model over V words is V. TF-IDF = tf · log₁₀(N/df).",
        "Dividing by the total number of words instead of count(w₁).")
def nlp_basics(rng, t):
    k = rng.random()
    if k < .4:
        words = [rng.choice(["the", "cat", "sat", "on", "mat"]) for _ in range(14)]
        w1 = rng.choice(sorted(set(words[:-1])))
        c1 = sum(1 for i in range(len(words) - 1) if words[i] == w1)
        w2 = rng.choice(sorted({words[i + 1] for i in range(len(words) - 1) if words[i] == w1}))
        c12 = sum(1 for i in range(len(words) - 1) if words[i] == w1 and words[i + 1] == w2)
        return Q("numeric", f"Corpus: \"{' '.join(words)}\". What is P('{w2}' | '{w1}') using bigram counts? (fraction or decimal)", Fraction(c12, c1), f"{c12}/{c1}")
    if k < .7:
        u, v = [rng.randint(0, 4) for _ in range(3)], [rng.randint(0, 4) for _ in range(3)]
        if not any(u) or not any(v):
            u, v = [1, 2, 3], [3, 2, 1]
        cs = sum(a * b for a, b in zip(u, v)) / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))
        return Q("numeric", f"What is the cosine similarity of {u} and {v}? (3 decimals)", f"{cs:.3f}", "dot ÷ norms")
    n, df, tf = rng.choice([100, 1000, 10000]), rng.choice([1, 10, 100]), rng.randint(1, 8)
    return Q("numeric", f"A term appears {tf} times in a document and in {df} of {n} documents. What is its TF-IDF (tf · log₁₀(N/df))? {SIG}", fmt(tf * math.log10(n / df)), "tf × idf")


@family("cv.basics", "Computer-vision arithmetic", {"computer_vision": (1, 12)},
        "Output size of a convolution = ⌊(n + 2p − k)/s⌋ + 1 for input n, kernel k, padding p, stride s. IoU = overlap area ÷ union area. Max-pooling keeps the largest value in each window. L stacked 3×3 convolutions see a (2L + 1)×(2L + 1) neighbourhood.",
        "Forgetting padding in the output-size formula.")
def cv_basics(rng, t):
    k = rng.random()
    if k < .35:
        n, kk, p, s = rng.choice([32, 28, 224, 64]), rng.choice([3, 5, 7]), rng.choice([0, 1, 2]), rng.choice([1, 2])
        return Q("numeric", f"A {n}×{n} image goes through a {kk}×{kk} convolution with padding {p} and stride {s}. What is the output width?", (n + 2 * p - kk) // s + 1, "⌊(n+2p−k)/s⌋ + 1")
    if k < .65:
        a = (rng.randint(0, 3), rng.randint(0, 3))
        wa, ha = rng.randint(3, 6), rng.randint(3, 6)
        b = (a[0] + rng.randint(0, 3), a[1] + rng.randint(0, 3))
        wb, hb = rng.randint(3, 6), rng.randint(3, 6)
        ix = max(0, min(a[0] + wa, b[0] + wb) - max(a[0], b[0]))
        iy = max(0, min(a[1] + ha, b[1] + hb) - max(a[1], b[1]))
        inter = ix * iy
        iou = inter / (wa * ha + wb * hb - inter)
        return Q("numeric", f"Box A has corner {a}, width {wa}, height {ha}; box B has corner {b}, width {wb}, height {hb}. What is the IoU? (3 decimals)", f"{iou:.3f}", "intersection ÷ union")
    L = rng.randint(2, 8)
    return Q("numeric", f"What is the width of the receptive field after {L} stacked 3×3 convolutions (stride 1)?", 2 * L + 1, "2L + 1")


# ===================================================================== systems & engineering of software

@family("cs.distributed", "Distributed systems and the cloud", {"distributed_systems": (1, 12), "cloud_computing": (1, 12)},
        "Amdahl's law: speedup = 1/((1 − p) + p/s) when a fraction p is sped up by s. Little's law: L = λ·W. Quorums: reads and writes overlap (strong consistency) when R + W > N. Availability: series = a₁·a₂, parallel replicas = 1 − (1 − a)ⁿ. "
        "A 99.9% SLA allows 0.1% downtime ≈ 43.2 minutes in a 30-day month.",
        "Adding availabilities instead of multiplying.")
def distributed(rng, t):
    k = rng.random()
    if k < .2:
        p, s = rng.choice([0.5, 0.8, 0.9, 0.95]), rng.choice([2, 4, 8, 16])
        return Q("numeric", f"{p:.0%} of a program can be sped up by a factor of {s}. What is the overall speedup (Amdahl's law)? {SIG}", fmt(1 / ((1 - p) + p / s)), "1/((1−p)+p/s)")
    if k < .4:
        lam, w = rng.choice([10, 50, 200]), rng.choice([0.1, 0.5, 2])
        return Q("numeric", f"Requests arrive at {lam} per second and each spends {w} s in the system. By Little's law, how many are in the system on average?", fmt(lam * w), "L = λW")
    if k < .55:
        n, r = rng.choice([3, 5, 7]), rng.randint(1, 3)
        return Q("numeric", f"With N = {n} replicas and read quorum R = {r}, what is the smallest write quorum W giving strong consistency (R + W > N)?", n - r + 1, "W > N − R")
    if k < .75:
        a, n = rng.choice([0.9, 0.95, 0.99]), rng.choice([2, 3])
        if rng.random() < .5:
            return Q("numeric", f"{n} independent replicas each have availability {a}. The service is up if at least one is up. What is its availability? (5 decimals)", f"{1 - (1 - a) ** n:.5f}", "1 − (1 − a)ⁿ")
        return Q("numeric", f"{n} services in series each have availability {a}. What is the combined availability? (4 decimals)", f"{a ** n:.4f}", "aⁿ")
    if k < .9:
        load, cap = rng.choice([1000, 2500, 8000]), rng.choice([300, 400, 750])
        return Q("numeric", f"Traffic is {load} requests/s and one instance handles {cap}. How many instances are needed (whole number, rounded up)?", math.ceil(load / cap), "⌈load/capacity⌉")
    sla = rng.choice([99, 99.9, 99.99])
    return Q("numeric", f"How many minutes of downtime per 30-day month does a {sla}% SLA allow? {SIG}", fmt((100 - sla) / 100 * 30 * 24 * 60), "(1 − SLA) × 43200 min")


@family("cs.compilers", "Parsing and expressions", {"compilers": (1, 12), "cs": (7, 10)},
        "Shunting-yard turns infix into postfix: output operands immediately; push operators on a stack, popping operators of higher or equal precedence first; brackets group. Postfix is evaluated with a stack. A lexer splits source text into tokens such as identifiers, numbers and operators.",
        "Ignoring operator precedence when converting.")
def compilers(rng, t):
    def expr():
        toks = [str(rng.randint(1, 9))]
        for _ in range(rng.randint(2, 4)):
            toks += [rng.choice("+-*"), str(rng.randint(1, 9))]
        return toks

    k = rng.random()
    toks = expr()
    if k < .45:
        prec = {"+": 1, "-": 1, "*": 2}
        out, st = [], []
        for tk in toks:
            if tk in prec:
                while st and prec[st[-1]] >= prec[tk]:
                    out.append(st.pop())
                st.append(tk)
            else:
                out.append(tk)
        out += st[::-1]
        return Q("cloze", f"Convert the infix expression `{' '.join(toks)}` to postfix (tokens separated by spaces; all operators are left-associative, * binds tighter than + and -).", " ".join(out), "shunting-yard", exact=True)
    if k < .75:
        val = eval(" ".join(toks))
        return Q("numeric", f"What is the value of {' '.join(toks)}? (usual precedence)", val, "evaluate")
    src = f"x{rng.randint(1, 9)} = {rng.randint(10, 99)} + y * {rng.randint(1, 9)};"
    import re as _re
    ntok = len(_re.findall(r"[A-Za-z_]\w*|\d+|[=+*;]", src))
    return Q("numeric", f"How many tokens does a lexer produce for `{src}` (identifiers, numbers, and each of = + * ; count as one token)?", ntok, "split into tokens")


@family("se.metrics", "Software engineering numbers", {"software_engineering": (1, 12), "web_dev": (1, 12)},
        "Cyclomatic complexity = E − N + 2 for a control-flow graph (or decisions + 1). Test coverage = lines executed ÷ total lines. A team of n people has n(n − 1)/2 communication paths. Defect density = defects ÷ KLOC.",
        "Counting only the if statements and forgetting loops and cases.")
def se_metrics(rng, t):
    k = rng.random()
    if k < .3:
        e, n = rng.randint(8, 20), rng.randint(6, 14)
        return Q("numeric", f"A control-flow graph has {e} edges and {n} nodes (one connected component). What is its cyclomatic complexity?", e - n + 2, "E − N + 2")
    if k < .55:
        n = rng.randint(3, 15)
        return Q("numeric", f"How many pairwise communication paths exist in a team of {n} people?", n * (n - 1) // 2, "n(n−1)/2")
    if k < .8:
        tot, cov = rng.choice([200, 400, 500, 1000]), rng.choice([0.5, 0.75, 0.8, 0.9])
        return Q("numeric", f"Tests execute {int(tot * cov)} of {tot} lines. What is the line coverage (%)?", fmt(cov * 100), "executed ÷ total")
    pts, vel = rng.choice([120, 90, 200]), rng.choice([20, 25, 30])
    return Q("numeric", f"{pts} story points remain and the team completes {vel} per sprint. How many sprints are needed (rounded up)?", math.ceil(pts / vel), "⌈remaining/velocity⌉")


@family("robot.basics", "Robot kinematics and control", {"robotics": (1, 12)},
        "Two-link arm: x = l₁cosθ₁ + l₂cos(θ₁+θ₂), y = l₁sinθ₁ + l₂sin(θ₁+θ₂). Differential drive: forward speed v = r(ω_L + ω_R)/2, turn rate ω = r(ω_R − ω_L)/L. PID: u = Kp·e + Ki·Σe·Δt + Kd·(e − e_prev)/Δt. Distance from encoder ticks = ticks/ticks_per_rev × 2πr.",
        "Mixing degrees and radians.")
def robot_basics(rng, t):
    k = rng.random()
    if k < .3:
        l1, l2 = rng.choice([1, 2]), rng.choice([1, 2])
        a, b = rng.choice([0, 90, 180]), rng.choice([0, 90, -90])
        x = l1 * math.cos(math.radians(a)) + l2 * math.cos(math.radians(a + b))
        return Q("numeric", f"A 2-link arm has l₁ = {l1}, l₂ = {l2}, θ₁ = {a}°, θ₂ = {b}°. What is the end-point x coordinate? (2 decimals)", f"{x:.2f}", "forward kinematics")
    if k < .55:
        r, wl, wr = rng.choice([0.05, 0.1]), rng.randint(2, 10), rng.randint(2, 10)
        return Q("numeric", f"A differential-drive robot has wheel radius {r} m and wheel speeds {wl} and {wr} rad/s. What is its forward speed (m/s)? {SIG}", fmt(r * (wl + wr) / 2), "r(ω_L+ω_R)/2")
    if k < .8:
        kp, ki, kd, e, ep, s = rng.choice([1, 2]), rng.choice([0.1, 0.5]), rng.choice([0.2, 1]), rng.randint(1, 5), rng.randint(0, 5), rng.randint(2, 8)
        return Q("numeric", f"PID gains Kp = {kp}, Ki = {ki}, Kd = {kd}; error now {e}, previous error {ep}, accumulated error sum {s}, Δt = 1. What is the control output? {SIG}", fmt(kp * e + ki * s + kd * (e - ep)), "Kp e + Ki Σe + Kd Δe")
    ticks, tpr, r = rng.choice([1200, 2400, 600]), rng.choice([600, 1200]), rng.choice([0.03, 0.05])
    return Q("numeric", f"A wheel of radius {r} m has an encoder with {tpr} ticks per revolution and counts {ticks} ticks. How far did it roll (m)? {SIG}", fmt(ticks / tpr * 2 * math.pi * r), "revolutions × circumference")


@family("opt.methods", "Optimisation and numerical methods", {"optimization": (1, 12), "numerical_methods": (1, 12)},
        "Linear programs reach their optimum at a corner of the feasible region: evaluate the objective at every corner. Newton's method: x ← x − f(x)/f′(x); for √c use x ← (x + c/x)/2. Bisection halves the interval each step. "
        "Euler: y ← y + h·f(t, y). Trapezoid rule: ∫f ≈ h[f₀/2 + f₁ + … + f_{n−1} + f_n/2]. Lagrange: maximise xy with x + y = s at x = y = s/2.",
        "Taking the first corner you check as the optimum.")
def opt_methods(rng, t):
    k = rng.random()
    if k < .2:
        a, b = rng.randint(6, 12), rng.randint(2, 5)
        c1, c2 = rng.randint(1, 5), rng.randint(1, 5)
        corners = [(0, 0), (b, 0), (b, a - b), (0, a)]
        return Q("numeric", f"Maximise {c1}x + {c2}y subject to x + y ≤ {a}, x ≤ {b}, x ≥ 0, y ≥ 0. What is the maximum value?", max(c1 * x + c2 * y for x, y in corners), f"corners: {corners}")
    if k < .4:
        c, x0 = rng.choice([2, 5, 10]), rng.choice([1, 2, 3])
        return Q("numeric", f"Apply one Newton step for f(x) = x² − {c} starting at x₀ = {x0}. What is x₁? {SIG}", fmt((x0 + c / x0) / 2), "(x + c/x)/2")
    if k < .55:
        w, tol = rng.choice([1, 2, 4, 8]), rng.choice([0.1, 0.01, 0.001])
        return Q("numeric", f"Bisection starts on an interval of width {w}. How many halvings guarantee width ≤ {tol}?", math.ceil(math.log2(w / tol)), "⌈log₂(w/tol)⌉")
    if k < .7:
        y0, h = rng.randint(0, 3), rng.choice([0.1, 0.5, 1])
        return Q("numeric", f"Euler's method for y′ = t + y with y(0) = {y0} and step h = {h}: what is y₁? {SIG}", fmt(y0 + h * (0 + y0)), "y₀ + h·f(0, y₀)")
    if k < .85:
        n = rng.choice([2, 4])
        hh = 2 / n
        vals = [(i * hh) ** 2 for i in range(n + 1)]
        trap = hh * (vals[0] / 2 + sum(vals[1:-1]) + vals[-1] / 2)
        return Q("numeric", f"Use the trapezoid rule with {n} equal intervals to estimate the integral of x² from 0 to 2. {SIG}", fmt(trap), "trapezoid rule")
    s = rng.choice([10, 20, 30, 100])
    return Q("numeric", f"Maximise x·y subject to x + y = {s}. What is the maximum product?", s * s // 4, "x = y = s/2")


@family("qc.basics", "Quantum computing basics", {"quantum_computing": (1, 12)},
        "A qubit state α|0⟩ + β|1⟩ has |α|² + |β|² = 1; measuring gives 0 with probability |α|². The Hadamard gate sends |0⟩ → (|0⟩ + |1⟩)/√2. n qubits span 2ⁿ basis states. Grover search over N items needs about (π/4)√N iterations. "
        "The order of a mod N is the smallest r > 0 with aʳ ≡ 1 (mod N) — the heart of Shor's algorithm.",
        "Adding amplitudes instead of squaring them to get probabilities.")
def qc_basics(rng, t):
    k = rng.random()
    if k < .3:
        a, b = rng.choice([(3, 4), (5, 12), (8, 15)])
        n = a * a + b * b
        c = int(math.isqrt(n))
        return Q("numeric", f"A qubit is in state ({a}/{c})|0⟩ + ({b}/{c})|1⟩. What is the probability of measuring 0? (fraction or decimal)", Fraction(a * a, c * c), "|α|²")
    if k < .5:
        n = rng.randint(2, 10)
        return Q("numeric", f"How many basis states do {n} qubits have?", 2 ** n, "2ⁿ")
    if k < .75:
        n = rng.choice([4, 6, 8, 10, 12, 16, 20])
        return Q("numeric", f"About how many Grover iterations (rounded down to a whole number, (π/4)√N) are needed to search N = 2^{n} items?", int(math.pi / 4 * 2 ** (n / 2)), "(π/4)√N")
    N = rng.choice([15, 21, 35, 33])
    a = rng.choice([x for x in range(2, N) if math.gcd(x, N) == 1])
    r = next(i for i in range(1, N * N) if pow(a, i, N) == 1)
    return Q("numeric", f"What is the order of {a} modulo {N} (the smallest r > 0 with {a}^r ≡ 1 mod {N})?", r, "try r = 1, 2, 3, …")
