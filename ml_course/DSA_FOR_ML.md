================================================================================
DSA FOR ML  (ml_course/DSA_FOR_ML.md)
================================================================================
Companion to:  COURSE.txt  PRACTICE.txt  PROJECT.txt  EXAMPLE.py
Track     :  ml_course  (Machine Learning) - the master-prompt "LeetCode + ML
              Coding" requirement, taught through the ML lens.
Audience  :  2nd-year B.Tech AI & ML student heading for AI/ML engineering.

WHY DSA MATTERS FOR AN ML ENGINEER (read this first)
================================================================================
An ML interview does not ask you to invert a binary tree because trees matter
per se - it asks because the SAME algorithmic muscle runs your ML:

  * Nearest-neighbor search (kNN, retrieval) is a DATA-STRUCTURE problem:
    brute force is O(n*d); a KD-tree/ball-tree is ~O(log n) on good data.
    "Why is kNN slow at inference?" is a DSA interview question.

  * Beam search decoding, top-k sampling, median-of-stream monitoring - all
    priority-queue (heap) problems.

  * Vocabulary building, TF counting, feature hashing, co-occurrence matrices -
    hash-map problems.

  * Tokenization (BPE), prefix matching - trie problems.

  * Viterbi decoding for HMM/CRF, dynamic time warping, edit distance for
    fuzzy matching - dynamic-programming problems.

  * Knowledge graphs, entity resolution, connected components of a similarity
    graph - graph + union-find problems.

  * argsort for feature importance, stable sorts for leaderboards, binary
    search for threshold/learning-rate search - sorting/searching problems.

So this module teaches DSA THROUGH ML: every pattern below names the data
structure, the ML operation it powers, canonical problems with complexity,
and the failure modes ML engineers actually hit.

Legend:  [E] easy  [M] medium  [H] hard   (relative to a first-year DSA bar)
Approaches are given so you can self-check; full code lives in your own
solutions or in EXAMPLE.py-style reference files.

================================================================================
TOPIC 1 - ARRAYS  (prefix sums, sliding window, two pointers)
================================================================================
ML CONNECTIONS
  * Rolling mean / expanding statistics for time-series features
    (Module 25).  A rolling window IS a sliding window.
  * Preprocessing in O(n) instead of O(n*k): prefix sums give any window sum
    in O(1) - the same trick as integral images in CV (Module 10 of the DL
    course: summed-area tables for box filters).
  * Two pointers: merging sorted model outputs, deduplicating sorted ids.

CANONICAL PROBLEMS
  1. [E] Prefix sums: range-sum queries. Approach: pref[i+1]=pref[i]+a[i];
     sum(l,r)=pref[r+1]-pref[l]. O(n) build, O(1) query.
  2. [E] Maximum subarray (Kadane). ML use: largest contiguous gain run of a
     strategy. Approach: cur=max(x,cur+x); best=max(best,cur). O(n).
  3. [M] Sliding-window maximum. ML use: rolling max of loss over a window.
     Approach: monotonic deque, O(n). Plain approach: heap per window O(n log k).
  4. [M] Two-sum / 3-sum on a sorted similarity matrix row. Approach:
     hash-map (2-sum O(n)); two pointers on sorted array.
  5. [M] Product of array except self (normalization without division).
     Approach: prefix products left->right and right->left, O(n) time O(1) extra.

FAILURE MODES
  * Off-by-one in window bounds (forecasting leaks the current row - the
    exact bug EXAMPLE.py Section 21 warns about).
  * Using min/max rescale where a *rolling* statistic is required.

================================================================================
TOPIC 2 - HASH MAPS
================================================================================
ML CONNECTIONS
  * Token -> id vocabulary (tokenizers), label -> index encoding.
  * Counting: term frequencies, class frequencies for priors (Naive Bayes),
    co-occurrence counting for PMI/PPMI embeddings (embedding_rag_lab.py!).
  * Feature hashing (hashing trick): O(1) memory-bounded feature maps.
  * Deduplication of training data (exact dupes) - Module 28 data engineering.

CANONICAL PROBLEMS
  1. [E] Majority element (Boyer-Moore). ML use: majority vote in Random
     Forest ensembles is literally this pattern. O(n) time O(1) space.
  2. [E] First non-repeating character. Approach: two-pass count then scan.
  3. [M] Group anagrams / group by hashable signature. ML use: bucketing
     near-duplicate rows. Approach: canonical key (sorted tuple or char
     counts) -> list.
  4. [M] Top-K frequent elements. See Topic 6 (heap) - counts first.
  5. [M] Subarray sum equals k. ML use: counting intervals with target metric.
     Approach: prefix-sum hash map of counts. O(n).
  6. [M] LRU cache (ordered dict). ML use: feature caches, model-response
     caches in serving (see rag_agent_server_prod.py). Approach: dict +
     doubly-linked list or OrderedDict.move_to_end.

FAILURE MODES
  * Unhashable keys (lists) - convert to tuples.
  * Python dict insertion order is guaranteed (3.7+), but don't rely on it
    across languages when order semantics matter.
  * Counting with default dicts vs Counter: fine; watch memory for huge
    cardinality - feature hashing bounds it.

================================================================================
TOPIC 3 - SORTING
================================================================================
ML CONNECTIONS
  * argsort -> feature-importance ranking, top-k model selection, nearest
    neighbors by distance (sort distances).
  * Stable sorting: leaderboards that must keep original order on ties
    (model comparison tables).
  * Sorting by key: ordering checkpoints by step, samples by difficulty for
    curriculum learning.

CANONICAL PROBLEMS
  1. [E] Merge two sorted arrays (merging sorted model outputs). O(n+m).
  2. [E] Kth largest element. Approach: quickselect O(n) average, or
     heap O(n log k).
  3. [M] Sort colors / 3-way partition. Approach: Dutch national flag. ML
     use: partitioning data into train/val/test by a label column in one pass.
  4. [M] Merge intervals. ML use: merging overlapping prediction windows /
     time ranges. Approach: sort by start, then merge overlapping. O(n log n).
  5. [M] Count of smaller numbers after self / inversion count. Approach:
     merge sort counting. ML use: rank correlation (Kendall tau) computation.

COMPLEXITY MEMO
  * Timsort (Python default): O(n log n) worst, O(n) on nearly-sorted data -
     exploit with nearly-sorted inputs.
  * np.argsort is C-speed; never sort in pure Python loops over big arrays.

================================================================================
TOPIC 4 - BINARY SEARCH
================================================================================
ML CONNECTIONS
  * Threshold tuning (Module 27): find the decision threshold that hits a
    target recall - binary search over thresholds.
  * Hyperparameter search on monotone curves (learning rate on loss).
  * Finding the split point in a sorted score list for top-k cutoffs.

CANONICAL PROBLEMS
  1. [E] Binary search (classic). Approach: lo/hi loop; watch mid overflow
     in C-like languages (lo+(hi-lo)//2).
  2. [M] Search in rotated sorted array. Approach: find which half is sorted,
     decide by target.
  3. [M] Find peak element. Approach: compare mid vs mid+1, move toward the
     larger side - O(log n). ML use: finding the max of a unimodal
     validation curve cheaply (ternary/unimodal search).
  4. [M] Kth smallest in sorted matrix / split-array-largest-sum. ML use:
     quantile computation on massive sorted runs, workload balancing.

FAILURE MODES
  * Infinite loops on lo<hi vs lo<=hi - test on length-1 arrays.
  * Floating thresholds: binary search on floats needs an iteration budget or
    epsilon, not integer termination.

================================================================================
TOPIC 5 - MATRIX OPERATIONS
================================================================================
ML CONNECTIONS
  * Images are matrices; convolutions are strided window ops (DL Module 10).
  * Sparse matrices: one-hot encoded data is ~all zeros; store sparsely.
  * Matrix transpose/view/reshape bugs are the #1 shape bug class in PyTorch
    (the labs in llm_course keep catching exactly these!).

CANONICAL PROBLEMS
  1. [E] Transpose (non-square). Approach: out[j][i]=a[i][j]. O(n*m).
  2. [M] Rotate image in place (90 deg). Approach: transpose + reverse rows.
  3. [M] Set matrix zeroes in O(1) extra space. Approach: use first row/col
     as flags.
  4. [M] Search a 2D matrix (row+col sorted) - staircase walk O(n+m). ML use:
     sorted score tables.
  5. [H] Maximal rectangle / largest rectangle in histogram. ML use: layout /
     bin packing analogies, convolutional feature-map analysis.
  6. [M] Sparse matrix multiplication. ML use: sparse embeddings, GNN
     adjacency products. Approach: iterate over nonzeros, skip zeros.

FAILURE MODES
  * Row-major vs column-major indexing (C vs Fortran layouts; np.asfortranarray).
  * In PyTorch: .view needs contiguous memory; use .reshape or .contiguous().
  * Broadcasting surprises (numpy) - always check shapes meet at the end.

================================================================================
TOPIC 6 - HEAPS (PRIORITY QUEUES)
================================================================================
ML CONNECTIONS
  * Top-k: nearest neighbors, top features, top misclassified samples.
  * Beam search: keep the k best partial hypotheses - a bounded heap.
  * Median of a stream: two heaps - for monitoring latency percentiles.
  * Merge k sorted lists: merging k model outputs / k shards.

CANONICAL PROBLEMS
  1. [E] Kth largest element in a stream. Approach: min-heap of size k.
  2. [M] Top-K frequent elements. Approach: Counter + heap of size k
     O(n log k), or quickselect O(n).
  3. [H] Merge k sorted lists. Approach: heap of (value, list-idx); pop-push.
     O(n log k). ML use: merging k sorted shards of scored candidates.
  4. [H] Find median from data stream. Approach: max-heap for lower half +
     min-heap for upper half; rebalance. O(log n) per insert.
  5. [H] Sliding window median (rolling percentile monitoring). Approach:
     two heaps + lazy deletion.

FAILURE MODES
  * Python heapq is a MIN-heap; negate keys for max behavior.
  * Heap of tuples compares the whole tuple - include a unique tiebreaker to
    avoid comparing unorderable objects.

================================================================================
TOPIC 7 - TREES (BST, TRIES)
================================================================================
ML CONNECTIONS
  * Decision trees ARE the module's Module 17 - tree traversal is literally
    walking a model's prediction path.
  * Tries: vocabulary prefix matching, autocomplete-style tokenization,
    BPE merges, package/import graph resolution.
  * BST-balanced ideas -> why tree ensembles limit depth (overfitting).

CANONICAL PROBLEMS
  1. [E] Binary tree traversals (pre/in/post/level). Approach: recursion or
     explicit stack; level-order with a queue. ML use: walking a decision
     tree for a feature-importance/leaf assignment.
  2. [M] Validate BST. Approach: carry (lo, hi) bounds in recursion.
  3. [M] Lowest common ancestor. ML use: finding the common prefix of two
     taxonomy nodes (hierarchical classification).
  4. [M] Implement Trie (insert/search/prefix). Approach: dict-of-dicts
     nodes + terminal flag. ML use: tokenizer vocabulary prefix checks.
  5. [H] Serialize/deserialize binary tree. ML use: model tree export.
  6. [E] Max depth of binary tree = decision-tree depth limit intuition.

DECISION-TREE SPECIFIC (ties to Module 17)
  * Splitting = recursive partition: each node stores feature index +
    threshold; inference is O(depth), not O(n).
  * Pruning = post-order deletion when a subtree doesn't improve validation.

================================================================================
TOPIC 8 - GRAPHS + UNION-FIND
================================================================================
ML CONNECTIONS
  * Knowledge graphs (entity-relation graphs) - traversal and shortest paths
    for Graph RAG (the AI_ENGINEERING course).
  * Connected components of a similarity graph = clustering; union-find is
    the workhorse (also for deduplicating records via transitive links).
  * BFS/DFS: label propagation in semi-supervised learning (Module 32).
  * Topological sort: task/dependency ordering for pipelines (DAGs in
    ML pipelines; also LangGraph graphs later).

CANONICAL PROBLEMS
  1. [M] Number of islands. Approach: DFS/BFS flood fill, or union-find.
     ML use: connected components in a segmentation mask.
  2. [M] Clone graph. Approach: BFS/DFS + node map. ML use: duplicating a
     computation graph with shared nodes (autograd graphs!).
  3. [M] Course schedule (cycle detection / topological sort). ML use:
     prerequisite ordering for the course dependency graph itself.
  4. [M] Number of connected components (union-find). Approach: path
     compression + union by rank ~O(alpha(n)).
  5. [H] Dijkstra shortest path. ML use: cost-based routing, concept drift
     path analysis. O((V+E) log V) with a heap.
  6. [H] Word ladder / minimum steps between strings. ML use: edit-distance
     neighborhood search; each step is a transformation.

================================================================================
TOPIC 9 - DYNAMIC PROGRAMMING
================================================================================
ML CONNECTIONS
  * Viterbi: optimal tag/state sequence in HMM/CRF - sequence labeling.
  * Dynamic time warping: aligning two time series of different lengths
    (Module 25 anomaly/alignment).
  * Edit/Levenshtein distance: fuzzy string matching for entity resolution.
  * Knapsack-style: resource allocation (which experiments fit a GPU budget).

CANONICAL PROBLEMS
  1. [E] Climbing stairs. Approach: dp[i]=dp[i-1]+dp[i-2]; O(n) time O(1) space.
  2. [M] Longest common subsequence. Approach: 2D dp table; O(n*m).
     ML use: sequence similarity, diffing.
  3. [M] Edit distance. Approach: dp over (i,j) with insert/delete/substitute
     costs. O(n*m). Base of fuzzy joins and string matching.
  4. [M] Coin change (min coins). Approach: dp over amounts. ML use: budgeted
     ensemble member selection.
  5. [H] Longest increasing subsequence. Approach: patience sorting O(n log n).
     ML use: detecting monotone improvement / early-stopping plateaus.
  6. [H] Word break. Approach: dp + trie/set of dict words. ML use: sentence
     tokenization with a fixed vocabulary.

DP CHECKLIST FOR ML PROBLEMS
  1. Subproblems: what's the state (i, j, k)?
  2. Recurrence: how does dp[i] relate to smaller states?
  3. Base cases: dp[0], empty-sequence behavior.
  4. Order: bottom-up table or top-down memo.
  5. Complexity: states * transitions, then optimize (space collapse,
     binary search, divide-and-conquer).

================================================================================
PATTERN -> ML OPERATION QUICK TABLE
================================================================================
  Data structure / pattern        | ML operation it powers
  --------------------------------|-----------------------------------------------
  Prefix sums / integral images   | rolling stats, box filters, window sums
  Hash map counters               | TF counts, vocabulary, co-occurrence (PPMI)
  Trie                            | tokenizer vocab prefix match, BPE merges
  Heap (top-k)                    | kNN, beam search, top features, percentile
  Binary search                   | threshold tuning, LR search, cutoffs
  Sorting + argsort               | feature ranking, nearest by distance
  Union-find                      | clustering, dedup, connected components
  Graph traversal                 | knowledge-graph retrieval, label propagation
  DP (Viterbi/DTW/edit distance)  | sequence tagging, time-series alignment, fuzzy join
  Sliding window                  | rolling metrics, time windows, convolutions

================================================================================
MONTHLY DSA PRACTICE PLAN (30-45 min/day)
================================================================================
WEEK 1-2   Arrays + hashing:    Q1-6 of Topics 1-2 (write clean O(n) code)
WEEK 3     Sorting + search:    Topic 3-4 (implement merge sort + quickselect
                                from memory)
WEEK 4     Matrix:              Topic 5 (transpose, rotation, spiral)
WEEK 5     Heaps:               Topic 6 (top-k family - the single most
                                interview-relevant pattern)
WEEK 6     Trees + tries:       Topic 7 (traversals from memory; trie)
WEEK 7     Graphs + union-find: Topic 8 (islands, components, topo sort)
WEEK 8     DP:                  Topic 9 (start LCS/edit distance/Viterbi)
ONGOING    One "ML-application" problem per week: re-implement the pattern
           against a real numpy/torch tensor problem (e.g. top-k cosine
           neighbors with a heap vs argsort; rolling quantiles with two heaps).

INTERVIEW NOTES
  * For each solution be ready to state: time complexity, space complexity,
    and the ML system that needs it.
  * Common follow-ups: "what if the data doesn't fit in memory?" (streaming,
    external sort, sharded top-k) - DSA answers with engineering answers.
  * Coding-round etiquette: clarify input size -> brute force -> optimize ->
    test edge cases (empty, single element, duplicates, negative numbers).

================================================================================
SELF-CHECK (tie to PRACTICE.txt LEVEL 0 and the FINAL EXAM coding section)
================================================================================
  1. Implement Kadane's max subarray and explain which time-series feature
     it computes.            (rolling max gain)
  2. Implement top-k via heap and via argsort; state when each is better.
                             (heap: streaming/unknown n; argsort: small n)
  3. Implement a trie insert + prefix count; use it to segment a sentence
     greedily.               (tokenizer-style longest-prefix matching)
  4. Implement union-find with path compression; count connected components
     of a kNN graph given an edge list.
  5. Implement edit distance; give the DP table for "cat" -> "cut".
                             (3x3 table; answer 1 substitution)
  6. Binary search the smallest threshold t such that recall >= 0.9 on a
     scored validation set.
  7. Two-heap median on a stream; explain the rebalancing invariant.
  8. Implement Dijkstra on a small graph by hand (5 nodes) and with a heap.
  9. Viterbi on a 2-state HMM given emissions and transitions (5 steps).
  10. Given a 100M-row log, design top-10 frequent error types with bounded
      memory.               (hash map + heap of size 10, or Count-Min sketch)
================================================================================
END DSA FOR ML
================================================================================
