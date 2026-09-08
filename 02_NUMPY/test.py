import numpy as np
import time

n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

    # Loop
start = time.time()
result_loop = np.empty(n)
for i in range(n):
    result_loop[i] = a[i] * b[i] + 1
loop_time = time.time() - start

start = time.time()
result_vec = a * b + 1
vec_time = time.time() - start

print(f"Loop: {loop_time:.4f}s")
print(f"Vectorized: {vec_time:.4f}s")
print(f"Speedup: {loop_time/vec_time:.1f}x")