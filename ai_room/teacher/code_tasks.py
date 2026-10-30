"""Programming exercises with reference solutions.

Expected values in the tests are produced by running the reference solution, so
the answer key cannot disagree with the task. Students are graded by running
their code against those asserts (see grading.run_code_tests).
"""

from __future__ import annotations

import copy
import inspect
import random
import textwrap
from dataclasses import dataclass, field
from typing import Callable

from .generators import Q, family


@dataclass
class Task:
    name: str
    level: float
    tags: tuple[str, ...]
    prompt: str
    ref: Callable | None = None
    args: Callable[[random.Random], list[tuple]] | None = None
    ref_src: str = ""          # for class-based tasks
    tests: list[str] = field(default_factory=list)  # fixed tests
    signature: str = ""

    def source(self) -> str:
        if self.ref_src:
            return self.ref_src.strip()
        src = textwrap.dedent(inspect.getsource(self.ref))
        return src[src.index("def "):].strip()

    def sig(self) -> str:
        if self.signature:
            return self.signature
        return f"{self.name}{inspect.signature(self.ref)}"

    def build_tests(self, rng: random.Random) -> list[str]:
        if self.tests:
            return list(self.tests)
        out = []
        for a in self.args(rng):
            shown = ", ".join(repr(x) for x in a)
            expected = self.ref(*copy.deepcopy(a))
            out.append(f"assert {self.name}({shown}) == {expected!r}")
        return out


TASKS: list[Task] = []


def task(level, tags, prompt, args):
    def deco(fn):
        TASKS.append(Task(fn.__name__, level, tuple(tags), prompt, ref=fn, args=args))
        return fn
    return deco


def _ints(rng, n=5, lo=-20, hi=50):
    return [rng.randint(lo, hi) for _ in range(n)]


WORDS = "the quick brown fox jumps over a lazy dog while cats and birds sing near old green trees".split()


# ----------------------------------------------------------------- level 1-2

@task(1, ["basic"], "returns the sum of a and b", lambda r: [(r.randint(-9, 99), r.randint(-9, 99)) for _ in range(4)])
def add(a, b):
    return a + b


@task(1, ["basic"], "returns n multiplied by itself", lambda r: [(r.randint(-12, 30),) for _ in range(4)])
def square(n):
    return n * n


@task(1.5, ["basic"], "returns True if n is even and False otherwise", lambda r: [(r.randint(-30, 60),) for _ in range(5)])
def is_even(n):
    return n % 2 == 0


@task(1.5, ["basic"], "returns the largest of the three numbers a, b and c (do not use max)", lambda r: [tuple(_ints(r, 3)) for _ in range(5)])
def max_of_three(a, b, c):
    best = a
    if b > best:
        best = b
    if c > best:
        best = c
    return best


@task(1.5, ["basic"], "returns the string 'Hello, NAME!' with NAME replaced by the argument", lambda r: [(r.choice(["Ada", "Bob", "Zoe", "Max", "Ana"]),) for _ in range(3)])
def greet(name):
    return f"Hello, {name}!"


# ----------------------------------------------------------------- level 2-3

@task(2, ["basic", "list"], "returns the sum of all numbers in the list nums (0 for an empty list)", lambda r: [(_ints(r, r.randint(0, 6)),) for _ in range(5)])
def sum_list(nums):
    total = 0
    for n in nums:
        total += n
    return total


@task(2, ["basic", "string"], "counts how many vowels (a, e, i, o, u, case-insensitive) the string s contains",
      lambda r: [(" ".join(r.sample(WORDS, r.randint(1, 4))),) for _ in range(4)] + [("",), ("RHYTHM",)])
def count_vowels(s):
    return sum(1 for ch in s.lower() if ch in "aeiou")


@task(2, ["basic", "string"], "returns the string s reversed", lambda r: [(" ".join(r.sample(WORDS, r.randint(1, 3))),) for _ in range(4)] + [("",)])
def reverse_string(s):
    return s[::-1]


@task(2.5, ["basic", "recursion"], "returns n! (n factorial) for n >= 0, with 0! = 1", lambda r: [(0,), (1,), (r.randint(2, 12),), (r.randint(2, 12),)])
def factorial(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


@task(2.5, ["string"], "returns True if s reads the same forwards and backwards, ignoring case and spaces",
      lambda r: [("Never odd or even",), ("hello",), ("A man a plan a canal Panama",), ("",), ("abca",)])
def is_palindrome(s):
    t = s.replace(" ", "").lower()
    return t == t[::-1]


@task(3, ["basic", "list"], "returns the list of strings for 1..n where multiples of 3 become 'Fizz', multiples of 5 become 'Buzz', multiples of both become 'FizzBuzz', and other numbers become their str() form",
      lambda r: [(1,), (5,), (r.randint(10, 20),)])
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        out.append("FizzBuzz" if i % 15 == 0 else "Fizz" if i % 3 == 0 else "Buzz" if i % 5 == 0 else str(i))
    return out


def _avg_args(r):
    out = []
    for _ in range(4):
        n = r.randint(2, 6)
        xs = [r.randint(0, 50) for _ in range(n)]
        xs[-1] += (-sum(xs)) % n
        out.append((xs,))
    return out


@task(3, ["basic", "list"], "returns the average (mean) of a non-empty list of numbers nums as a float", _avg_args)
def average(nums):
    return sum(nums) / len(nums)


# ----------------------------------------------------------------- level 3-4

@task(3.5, ["recursion", "alg"], "returns the n-th Fibonacci number with fibonacci(0) = 0 and fibonacci(1) = 1", lambda r: [(0,), (1,), (2,), (r.randint(5, 25),), (r.randint(20, 30),)])
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@task(3.5, ["math", "alg"], "returns True if n is a prime number and False otherwise (n can be 0, 1 or negative)", lambda r: [(0,), (1,), (2,), (r.choice([97, 101, 7919]),), (r.choice([91, 100, 7917]),)])
def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


@task(3.5, ["math", "alg"], "returns the greatest common divisor of the positive integers a and b", lambda r: [(r.randint(1, 200), r.randint(1, 200)) for _ in range(4)] + [(12, 18)])
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@task(4, ["list", "ds"], "flattens a list whose items are numbers or lists of numbers by one level, e.g. [1, [2, 3], 4] -> [1, 2, 3, 4]",
      lambda r: [([1, [2, 3], 4],), ([],), ([[1], [2], 3, [4, 5]],), ([[r.randint(0, 9) for _ in range(r.randint(0, 3))], r.randint(0, 9)],)])
def flatten(lst):
    out = []
    for x in lst:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def _two_sum_args(r):
    out = []
    while len(out) < 4:
        nums = r.sample(range(-20, 40), r.randint(4, 8))
        i, j = sorted(r.sample(range(len(nums)), 2))
        target = nums[i] + nums[j]
        pairs = [(a, b) for a in range(len(nums)) for b in range(a + 1, len(nums)) if nums[a] + nums[b] == target]
        if len(pairs) == 1:
            out.append((nums, target))
    return out


@task(4, ["list", "ds", "alg"], "returns a tuple (i, j) with i < j of the two indices whose values in nums add up to target (exactly one such pair exists)", _two_sum_args)
def two_sum(nums, target):
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return (seen[target - x], j)
        seen[x] = j


@task(4, ["string", "ds"], "returns a dict counting how many times each word occurs in text (split on spaces, compare in lower case)",
      lambda r: [(" ".join(r.choice(WORDS[:6]).capitalize() if r.random() < .3 else r.choice(WORDS[:6]) for _ in range(r.randint(3, 9))),) for _ in range(4)])
def word_frequencies(text):
    out = {}
    for w in text.lower().split():
        out[w] = out.get(w, 0) + 1
    return out


@task(4, ["list"], "returns the second-largest distinct value in nums (nums has at least two distinct values)",
      lambda r: [([5, 3, 9, 9, 1],), ([r.randint(0, 20) for _ in range(6)] + [100, 99],), ([2, 1],), ([-5, -5, -7, -1],)])
def second_largest(nums):
    return sorted(set(nums))[-2]


@task(4, ["list"], "removes duplicates from lst but keeps the first occurrence of each value in its original order",
      lambda r: [([r.randint(0, 5) for _ in range(r.randint(0, 9))],) for _ in range(5)])
def remove_duplicates(lst):
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@task(4, ["list"], "returns lst rotated to the right by k positions (k may be larger than len(lst); an empty list stays empty)",
      lambda r: [([1, 2, 3, 4, 5], 2), ([], 3), ([r.randint(0, 9) for _ in range(5)], 7), ([1, 2, 3], 0)])
def rotate_list(lst, k):
    if not lst:
        return []
    k %= len(lst)
    return lst[-k:] + lst[:-k] if k else lst[:]


# ----------------------------------------------------------------- level 4-6

@task(5, ["ds", "string"], "returns True if the brackets ()[]{} in s are balanced and correctly nested, ignoring all other characters",
      lambda r: [("([]{})",), ("(]",), ("a(b[c]{d})e",), ("((",), ("",), ("{[()]}(",)])
def balanced_parens(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


@task(5, ["list", "alg"], "merges two already-sorted lists a and b into one sorted list", lambda r: [(sorted(_ints(r, r.randint(0, 5))), sorted(_ints(r, r.randint(0, 5)))) for _ in range(5)])
def merge_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


def _bs_args(r):
    out = []
    for _ in range(5):
        arr = sorted(r.sample(range(100), r.randint(0, 9)))
        out.append((arr, r.choice(arr) if arr and r.random() < .6 else r.randint(0, 99)))
    return out


@task(5, ["alg", "search"], "returns the index of x in the sorted list arr of distinct numbers, or -1 if x is absent (use binary search)", _bs_args)
def binary_search(arr, x):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


@task(5, ["string"], "returns True if the strings a and b are anagrams of each other (same letters, ignoring case and spaces)",
      lambda r: [("Listen", "Silent"), ("abc", "abd"), ("Dormitory", "dirty room"), ("", ""), ("aab", "abb")])
def are_anagrams(a, b):
    f = lambda s: sorted(s.replace(" ", "").lower())  # noqa: E731
    return f(a) == f(b)


def _roman(n):
    out = ""
    for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
        while n >= v:
            out += s
            n -= v
    return out


@task(5.5, ["string", "alg"], "converts a Roman numeral string (I, V, X, L, C, D, M with subtractive pairs like IV and CM) to an integer",
      lambda r: [(_roman(r.randint(1, 3999)),) for _ in range(5)] + [("IV",)])
def roman_to_int(s):
    val = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and val[ch] < val[s[i + 1]]:
            total -= val[ch]
        else:
            total += val[ch]
    return total


@task(5.5, ["string"], "returns the longest common prefix of the list of strings words (an empty string if there is none or the list is empty)",
      lambda r: [(["flower", "flow", "flight"],), (["dog", "car"],), ([],), (["interview", "internet", "interval"],), (["same"],)])
def longest_common_prefix(words):
    if not words:
        return ""
    pre = words[0]
    for w in words[1:]:
        while not w.startswith(pre):
            pre = pre[:-1]
    return pre


@task(6, ["ds", "alg"], "returns the k most frequent values in nums as a list, ordered by frequency (highest first) and by value (smallest first) to break ties",
      lambda r: [([r.randint(0, 5) for _ in range(r.randint(6, 12))], r.randint(1, 3)) for _ in range(5)])
def top_k_frequent(nums, k):
    from collections import Counter
    c = Counter(nums)
    return sorted(c, key=lambda v: (-c[v], v))[:k]


@task(6, ["alg", "dp"], "returns the largest sum of any non-empty contiguous sub-list of nums",
      lambda r: [(_ints(r, r.randint(1, 8), -9, 12),) for _ in range(5)] + [([-3, -1, -2],)])
def max_subarray(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


@task(6, ["list", "alg"], "rotates the square matrix m (list of lists) 90 degrees clockwise and returns the new matrix",
      lambda r: [([[r.randint(0, 9) for _ in range(n)] for _ in range(n)],) for n in (1, 2, 3, 3, 4)])
def rotate_matrix(m):
    return [list(row) for row in zip(*m[::-1])]


@task(6, ["alg", "math"], "returns the number of steps the Collatz process needs to reach 1 from n (halve if even, 3n+1 if odd); collatz_steps(1) = 0",
      lambda r: [(1,), (6,), (r.randint(7, 100),), (27,)])
def collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


# ----------------------------------------------------------------- level 6-8

def _intervals(r):
    out = []
    for _ in range(r.randint(0, 6)):
        a = r.randint(0, 15)
        out.append([a, a + r.randint(0, 5)])
    return out


@task(7, ["list", "alg"], "merges overlapping intervals (each is a [start, end] list; touching intervals such as [1,3] and [3,5] overlap) and returns the merged intervals sorted by start",
      lambda r: [(_intervals(r),) for _ in range(5)] + [([[1, 3], [2, 6], [8, 10], [15, 18]],)])
def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _graph_args(r):
    out = []
    for _ in range(5):
        n = r.randint(4, 8)
        g = {i: [] for i in range(n)}
        for _ in range(r.randint(n - 1, n + 3)):
            a, b = r.sample(range(n), 2)
            if b not in g[a]:
                g[a].append(b)
                g[b].append(a)
        out.append((g, 0, r.randrange(n)))
    return out


@task(7, ["graph", "alg", "ds"], "returns the minimum number of edges on a path from start to goal in the undirected graph (a dict mapping each node to a list of neighbours), or -1 if no path exists",
      _graph_args)
def shortest_path_length(graph, start, goal):
    from collections import deque
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            return dist[u]
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return -1


@task(7, ["alg", "dp"], "returns the length of the longest strictly increasing subsequence of nums (0 for an empty list)",
      lambda r: [(_ints(r, r.randint(0, 9), 0, 20),) for _ in range(5)] + [([10, 9, 2, 5, 3, 7, 101, 18],)])
def lis_length(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


@task(7, ["alg", "dp"], "returns the fewest coins from coins (unlimited supply of each) that sum to amount, or -1 if impossible; amount 0 needs 0 coins",
      lambda r: [(r.sample([1, 2, 5, 7, 10, 25], r.randint(1, 4)), r.randint(0, 40)) for _ in range(5)] + [([2], 3), ([1, 2, 5], 11)])
def coin_change(coins, amount):
    INF = float("inf")
    best = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    return -1 if best[amount] == INF else best[amount]


@task(7.5, ["string", "alg", "dp"], "returns the edit (Levenshtein) distance between strings a and b: the fewest single-character insertions, deletions and substitutions to turn a into b",
      lambda r: [("kitten", "sitting"), ("", "abc"), ("flaw", "lawn"), ("".join(r.choice("abc") for _ in range(r.randint(0, 6))), "".join(r.choice("abc") for _ in range(r.randint(0, 6))))])
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@task(7.5, ["alg", "dp"], "solves the 0/1 knapsack problem: returns the maximum total value of items (weights[i], values[i]) whose total weight is at most capacity, using each item at most once",
      lambda r: [([r.randint(1, 8) for _ in range(n)], [r.randint(1, 20) for _ in range(n)], r.randint(0, 15)) for n in (1, 3, 4, 5, 6)])
def knapsack(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]


def _grid(r):
    h, w = r.randint(1, 5), r.randint(1, 5)
    return [[1 if r.random() < .45 else 0 for _ in range(w)] for _ in range(h)]


@task(8, ["graph", "alg"], "counts the islands in a grid (list of lists of 0/1): an island is a maximal group of 1s connected up, down, left or right",
      lambda r: [(_grid(r),) for _ in range(5)] + [([[1, 1, 0], [0, 0, 1], [1, 0, 1]],)])
def count_islands(grid):
    seen, count = set(), 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1 and (i, j) not in seen:
                count += 1
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if (x, y) in seen or not (0 <= x < len(grid) and 0 <= y < len(grid[0])) or grid[x][y] != 1:
                        continue
                    seen.add((x, y))
                    stack += [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return count


def _rpn(r):
    out = []
    for _ in range(5):
        toks = [str(r.randint(1, 9))]
        for _ in range(r.randint(1, 4)):
            toks += [str(r.randint(1, 9)), r.choice("+-*")]
        out.append((toks,))
    return out


@task(8, ["ds", "alg"], "evaluates an expression in reverse Polish notation: tokens is a list of strings, each an integer or one of '+', '-', '*'; returns the integer result",
      _rpn)
def eval_rpn(tokens):
    st = []
    for t in tokens:
        if t in "+-*" and len(t) == 1:
            b, a = st.pop(), st.pop()
            st.append(a + b if t == "+" else a - b if t == "-" else a * b)
        else:
            st.append(int(t))
    return st[0]


# ----------------------------------------------------------------- level 8-10

def _wgraph(r):
    out = []
    for _ in range(4):
        n = r.randint(4, 7)
        g = {i: [] for i in range(n)}
        for _ in range(r.randint(n, 2 * n)):
            a, b = r.sample(range(n), 2)
            if all(v != b for v, _ in g[a]):
                g[a].append((b, r.randint(1, 9)))
        out.append((g, 0))
    return out


@task(9, ["graph", "alg"], "returns a dict of the shortest distances from source to every reachable node in a weighted directed graph (dict: node -> list of (neighbour, weight) with positive weights); unreachable nodes are left out",
      _wgraph)
def dijkstra(graph, source):
    import heapq
    dist = {}
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        for v, w in graph[u]:
            if v not in dist:
                heapq.heappush(pq, (d + w, v))
    return dist


def _dgraph(r):
    out = []
    for _ in range(5):
        n = r.randint(3, 6)
        g = {i: [] for i in range(n)}
        for _ in range(r.randint(2, n + 2)):
            a, b = r.sample(range(n), 2)
            if b not in g[a]:
                g[a].append(b)
        out.append((g,))
    return out


@task(9, ["graph", "alg"], "returns True if the directed graph (dict mapping each node to a list of successors) contains a cycle",
      _dgraph)
def has_cycle(graph):
    state = {}

    def dfs(u):
        state[u] = 1
        for v in graph[u]:
            if state.get(v) == 1 or (v not in state and dfs(v)):
                return True
        state[u] = 2
        return False

    return any(u not in state and dfs(u) for u in graph)


@task(9, ["string", "dp", "alg"], "returns the longest palindromic substring of s (the leftmost one if several have the same length; '' for an empty string)",
      lambda r: [("".join(r.choice("abc") for _ in range(r.randint(0, 9))),) for _ in range(5)] + [("babad",), ("cbbd",)])
def longest_palindromic_substring(s):
    best = ""
    for c in range(len(s)):
        for lo, hi in ((c, c), (c, c + 1)):
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo -= 1
                hi += 1
            if hi - lo - 1 > len(best):
                best = s[lo + 1:hi]
    return best


@task(9, ["alg", "recursion"], "returns the number of ways to place n non-attacking queens on an n x n chessboard (n between 1 and 7)",
      lambda r: [(1,), (4,), (5,), (6,), (r.randint(1, 7),)])
def n_queens_count(n):
    def place(row, cols, d1, d2):
        if row == n:
            return 1
        total = 0
        for c in range(n):
            if c not in cols and row - c not in d1 and row + c not in d2:
                total += place(row + 1, cols | {c}, d1 | {row - c}, d2 | {row + c})
        return total
    return place(0, frozenset(), frozenset(), frozenset())


@task(9.5, ["alg", "dp", "recursion"], "returns how many subsets of the list nums (positive integers; subsets are chosen by position) add up to target",
      lambda r: [([r.randint(1, 9) for _ in range(r.randint(1, 8))], r.randint(1, 20)) for _ in range(5)])
def count_subsets(nums, target):
    ways = [1] + [0] * target
    for x in nums:
        for s in range(target, x - 1, -1):
            ways[s] += ways[s - x]
    return ways[target]


def _dag(r):
    out = []
    for _ in range(5):
        n = r.randint(3, 7)
        order = r.sample(range(n), n)
        g = {i: [] for i in range(n)}
        for _ in range(r.randint(2, n + 2)):
            a, b = sorted(r.sample(range(n), 2))
            u, v = order[a], order[b]
            if v not in g[u]:
                g[u].append(v)
        out.append((g,))
    return out


@task(9.5, ["graph", "alg"], "returns a topological ordering of the nodes of a directed acyclic graph (dict: node -> list of successors) as a list; when several nodes are available, always take the smallest one first",
      _dag)
def topological_order(graph):
    import heapq
    indeg = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            indeg[v] += 1
    heap = [u for u in graph if indeg[u] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    return out


@task(10, ["string", "dp"], "returns the length of the longest common subsequence of the strings a and b",
      lambda r: [("".join(r.choice("abcd") for _ in range(r.randint(0, 8))), "".join(r.choice("abcd") for _ in range(r.randint(0, 8)))) for _ in range(5)] + [("abcde", "ace")])
def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


@task(10, ["string", "alg"], "returns the sorted list of all start indices where the pattern pat occurs in text (overlapping matches count; an empty pattern matches nowhere)",
      lambda r: [("".join(r.choice("ab") for _ in range(r.randint(0, 12))), "".join(r.choice("ab") for _ in range(r.randint(0, 3)))) for _ in range(6)] + [("aaaa", "aa")])
def find_all(text, pat):
    if not pat:
        return []
    return [i for i in range(len(text) - len(pat) + 1) if text.startswith(pat, i)]


def _regex_args(r):
    out = []
    for _ in range(6):
        s = "".join(r.choice("ab") for _ in range(r.randint(0, 5)))
        p = "".join(r.choice(["a", "b", ".", "a*", "b*", ".*"]) for _ in range(r.randint(0, 4)))
        out.append((s, p))
    return out + [("aa", "a*"), ("ab", ".*c")]


@task(10.5, ["string", "dp", "recursion"], "implements regular-expression matching of the WHOLE string s against pattern p, where '.' matches any single character and 'x*' matches zero or more of x (x is a letter or '.')",
      _regex_args)
def regex_match(s, p):
    from functools import lru_cache

    @lru_cache(None)
    def m(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return m(i, j + 2) or (first and m(i + 1, j))
        return first and m(i + 1, j + 1)

    return m(0, 0)


def _expr_args(r):
    out = []
    for _ in range(5):
        toks = [str(r.randint(1, 20))]
        for _ in range(r.randint(1, 4)):
            toks += [r.choice("+-*"), str(r.randint(1, 20))]
        s = " ".join(toks)
        if r.random() < .5:
            s = f"({s}) * {r.randint(1, 5)}"
        out.append((s,))
    return out


@task(10.5, ["string", "alg", "ds"], "evaluates an arithmetic expression string with non-negative integers, + - *, spaces and parentheses, respecting precedence (e.g. '2 + 3 * (4 - 1)' -> 11) and returns the integer",
      _expr_args)
def eval_expression(s):
    toks = []
    i = 0
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(int(s[i:j]))
            i = j
        else:
            if s[i] != " ":
                toks.append(s[i])
            i += 1
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def atom():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            v = expr()
            pos += 1
            return v
        return t

    def term():
        nonlocal pos
        v = atom()
        while peek() == "*":
            pos += 1
            v *= atom()
        return v

    def expr():
        nonlocal pos
        v = term()
        while peek() in ("+", "-"):
            op = toks[pos]
            pos += 1
            v = v + term() if op == "+" else v - term()
        return v

    return expr()


# ----------------------------------------------------------------- class-based (fixed tests)

def _class_task(name, level, tags, prompt, signature, src, tests):
    TASKS.append(Task(name, level, tuple(tags), prompt, ref_src=src, tests=tests, signature=signature))


_class_task("LRUCache", 10, ["ds"],
            "implements a class LRUCache(capacity) with get(key) (returns the value, or -1 if absent, and marks the key as most recently used) and put(key, value) (inserts or updates; when over capacity it evicts the least recently used key)",
            "class LRUCache(capacity)", '''
class LRUCache:
    def __init__(self, capacity):
        from collections import OrderedDict
        self.cap, self.d = capacity, OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
''', ["c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1; c.put(3, 3); assert c.get(2) == -1; c.put(4, 4); assert c.get(1) == -1; assert c.get(3) == 3; assert c.get(4) == 4",
      "c = LRUCache(1); c.put(1, 10); c.put(1, 20); assert c.get(1) == 20; c.put(2, 5); assert c.get(1) == -1",
      "c = LRUCache(3); [c.put(i, i * i) for i in range(5)]; assert [c.get(i) for i in range(5)] == [-1, -1, 4, 9, 16]"])

_class_task("Trie", 10, ["ds", "string"],
            "implements a class Trie with insert(word), search(word) (True only if exactly that word was inserted) and starts_with(prefix) (True if any inserted word begins with prefix)",
            "class Trie()", '''
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True

    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node

    def search(self, word):
        node = self._walk(word)
        return bool(node and "$" in node)

    def starts_with(self, prefix):
        return self._walk(prefix) is not None
''', ["t = Trie(); t.insert('apple'); assert t.search('apple') is True; assert t.search('app') is False; assert t.starts_with('app') is True",
      "t = Trie(); t.insert('app'); t.insert('apple'); assert t.search('app'); assert not t.search('apples'); assert not t.starts_with('b')",
      "t = Trie(); assert not t.search('a'); assert not t.starts_with('a')"])

_class_task("UnionFind", 10.5, ["ds", "graph"],
            "implements a class UnionFind(n) for elements 0..n-1 with union(a, b), connected(a, b) (True if a and b are in the same set) and count() (the current number of separate sets)",
            "class UnionFind(n)", '''
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.sets = n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
            self.sets -= 1

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def count(self):
        return self.sets
''', ["u = UnionFind(5); assert u.count() == 5; u.union(0, 1); u.union(1, 2); assert u.connected(0, 2); assert not u.connected(0, 3); assert u.count() == 3",
      "u = UnionFind(3); u.union(0, 1); u.union(0, 1); assert u.count() == 2; u.union(1, 2); assert u.count() == 1 and u.connected(0, 2)",
      "u = UnionFind(1); assert u.count() == 1 and u.connected(0, 0)"])

_class_task("MedianFinder", 11, ["ds", "alg"],
            "implements a class MedianFinder with add(x) and median() returning the median of all numbers added so far (the average of the two middle values when the count is even; median() is only called after at least one add)",
            "class MedianFinder()", '''
class MedianFinder:
    def __init__(self):
        import heapq
        self.hq = heapq
        self.lo, self.hi = [], []

    def add(self, x):
        self.hq.heappush(self.lo, -x)
        self.hq.heappush(self.hi, -self.hq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            self.hq.heappush(self.lo, -self.hq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
''', ["m = MedianFinder(); m.add(1); assert m.median() == 1; m.add(2); assert m.median() == 1.5; m.add(3); assert m.median() == 2",
      "m = MedianFinder(); [m.add(x) for x in (5, 1, 9, 3)]; assert m.median() == 4.0",
      "m = MedianFinder(); [m.add(x) for x in (-1, -2, -3)]; assert m.median() == -2"])


# ----------------------------------------------------------------- families

def _make_code(rng, t, lo, hi, tag=None):
    pool = [k for k in TASKS if lo <= k.level <= hi and (tag is None or tag in k.tags)]
    pool.sort(key=lambda k: k.level)
    if not pool:
        pool = sorted(TASKS, key=lambda k: abs(k.level - (lo + hi) / 2))[:6]
    # lean towards harder tasks as t grows
    idx = min(len(pool) - 1, int(rng.random() ** (1.6 - t) * len(pool)))
    k = pool[idx]
    tests = k.build_tests(rng)
    what = "class" if k.signature.startswith("class ") else "function"
    sig = k.sig().removeprefix("class ")
    return Q("code", f"Write a Python {what} `{sig}` that {k.prompt}. Reply with the complete code in one ```python block.", k.source(),
             f"Reference solution:\n```python\n{k.source()}\n```", function_name=k.name, tests=tests)


CARD_CODE = ("Read the task carefully, decide the inputs and the output, and handle the edge cases (empty input, one element, negative numbers). "
             "Write a small, clear function, then trace it by hand on one of the examples before trusting it. Use loops, dictionaries and sets for most tasks; "
             "use a stack for nesting, a queue for breadth-first search, a heap for 'smallest next', and dynamic programming (a table of sub-answers) when sub-problems overlap.")


@family("code.write.basic", "Writing small functions", {"code": (1, 3)}, CARD_CODE, "Printing instead of returning the value.")
def write_basic(rng, t):
    return _make_code(rng, t, 1, 3)


@family("code.write.core", "Lists, strings and recursion", {"code": (4, 6), "data_structures": (1, 3), "algorithms": (1, 3), "software_engineering": (1, 4)}, CARD_CODE,
        "Modifying a list while looping over it.")
def write_core(rng, t):
    return _make_code(rng, t, 3.5, 6)


@family("code.write.intermediate", "Algorithms and data structures", {"code": (7, 9), "data_structures": (4, 7), "algorithms": (4, 8), "software_engineering": (5, 8)}, CARD_CODE,
        "Not handling the empty input or the not-found case.")
def write_intermediate(rng, t):
    return _make_code(rng, t, 5, 9)


@family("code.write.advanced", "Advanced algorithms and classes", {"code": (10, 12), "data_structures": (8, 12), "algorithms": (9, 12), "software_engineering": (9, 12)}, CARD_CODE,
        "Exponential recursion without memoisation.")
def write_advanced(rng, t):
    return _make_code(rng, t, 8, 12)


@family("code.ds", "Data-structure tasks", {"data_structures": (1, 12), "algorithms": (1, 12)}, CARD_CODE, "Off-by-one errors in indices.")
def write_ds(rng, t):
    lo, hi = 3.5 + 7 * t - 2, 3.5 + 7 * t + 2
    return _make_code(rng, t, lo, hi, tag="ds")


@family("code.alg", "Algorithm tasks", {"algorithms": (1, 12)}, CARD_CODE, "Choosing a quadratic algorithm when a linear one exists.")
def write_alg(rng, t):
    lo, hi = 3.5 + 7 * t - 2, 3.5 + 7 * t + 2
    return _make_code(rng, t, lo, hi, tag="alg")


# ----------------------------------------------------------------- reading code

def _snippets(rng, t):
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 9)
    n, m = rng.randint(4, 9), rng.randint(2, 4)
    w = rng.choice(["python", "banana", "garden", "rocket"])
    xs = [rng.randint(1, 9) for _ in range(rng.randint(4, 6))]
    i, j = rng.randint(0, 2), rng.randint(3, 5)
    pool = [
        (0, f"x = {a}\ny = {b}\nprint(x * y + {c})"),
        (0, f"s = '{w}'\nprint(len(s) + {a})"),
        (1, f"total = 0\nfor i in range({n}):\n    total += i * {m}\nprint(total)"),
        (1, f"s = '{w}'\nprint(s[::-1].upper())"),
        (1, f"nums = {xs}\nprint(max(nums) - min(nums))"),
        (2, f"nums = {xs}\nprint(sum([x * x for x in nums if x % 2 == 1]))"),
        (2, f"n = {n * 3}\nsteps = 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\nprint(steps)"),
        (2, f"a = list(range(12))\nprint(a[{i}:{j}:{m}])"),
        (3, f"def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nprint(f({rng.randint(3, 7)}))"),
        (3, f"d = {{}}\nfor ch in '{w}':\n    d[ch] = d.get(ch, 0) + 1\nprint(max(d.values()))"),
        (3, f"def g(x, acc=[]):\n    acc.append(x)\n    return len(acc)\ng({a}); g({b})\nprint(g({c}))"),
        (4, f"xs = {xs}\nout = []\nfor i, x in enumerate(xs):\n    if i % 2 == 0:\n        out.append(x)\n    else:\n        out.insert(0, x)\nprint(out)"),
        (4, f"try:\n    v = [1, 2, 3][{rng.randint(0, 4)}]\nexcept IndexError:\n    v = -1\nfinally:\n    v = v * 2\nprint(v)"),
    ]
    return pool


@family("code.predict", "Reading and tracing code", {"code": (1, 7), "cs": (3, 6)},
        "To predict what a program prints, run it by hand: write down every variable, update the values line by line, and follow loops one pass at a time. "
        "Remember: indices start at 0; slices a[i:j] stop before j; // is whole-number division; a mutable default argument keeps its changes between calls.",
        "Skipping loop iterations in your head.")
def predict(rng, t):
    import contextlib
    import io
    level_cap = int(t * 4) + 1
    pool = [p for lvl, p in _snippets(rng, t) if lvl <= level_cap] or [_snippets(rng, t)[0][1]]
    code = rng.choice(pool)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    out = buf.getvalue().strip()
    kind = "numeric" if out.lstrip("-").isdigit() else "cloze"
    return Q(kind, f"What does this Python program print?\n```python\n{code}\n```" + ("" if kind == "numeric" else "\n(Give the output exactly as printed.)"), out, f"Output: {out}", **({} if kind == "numeric" else {"exact": True}))
