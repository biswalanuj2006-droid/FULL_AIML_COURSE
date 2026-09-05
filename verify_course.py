"""
COURSE HEALTH CHECK - compile everything, re-run every fast runnable
example, and print a health summary.

What it does:
  1. py_compile over EVERY .py in the course (syntax gate).
  2. Executes the fast, dependency-light examples (numpy/sklearn-only,
     no servers, no downloads, no long training) with per-file timeouts.
  3. Explicitly SKIPS examples that need heavy frameworks, network
     downloads, or a running server - each with a reason.
  4. Reports empty directories, module-prefix uniqueness, and totals.

Run:  python verify_course.py
"""
import os
import py_compile
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# 1. COMPILE EVERYTHING
# ----------------------------------------------------------------------
def compile_all():
    bad = []
    total = 0
    for dirpath, _, files in os.walk(ROOT):
        if "__pycache__" in dirpath or ".git" in dirpath:
            continue
        for fn in files:
            if fn.endswith(".py"):
                total += 1
                p = os.path.join(dirpath, fn)
                try:
                    py_compile.compile(p, doraise=True)
                except py_compile.PyCompileError as e:
                    bad.append((p, str(e)))
    return total, bad


# ----------------------------------------------------------------------
# 2. RUNNABLE EXAMPLES (fast, dependency-light)
# ----------------------------------------------------------------------
RUN = [
    # from-scratch algorithms at code/ root
    "code/01_linear_regression_from_scratch.py",
    "code/02_logistic_regression_from_scratch.py",
    "code/03_knn_from_scratch.py",
    "code/04_kmeans_from_scratch.py",
    "code/05_decision_tree_from_scratch.py",
    "code/06_neural_network_from_scratch.py",
    # from-scratch algorithms in code/ml/
    "code/ml/gradient_descent_from_scratch.py",
    "code/ml/softmax_cross_entropy_from_scratch.py",
    "code/ml/pca_from_scratch.py",
    "code/ml/random_forest_from_scratch.py",
    "code/ml/gradient_boosting_from_scratch.py",
    # foundation examples
    "code/python/01_stdlib_for_ai.py",
    "code/numpy/01_numpy_fundamentals.py",
    "code/pandas/01_pandas_fundamentals.py",
    # NLP + RAG demos
    "code/nlp/01_nlp_classic_pipeline.py",
    "code/rag/01_rag_minimal.py",
    "code/rag/02_rag_fail_fix_demo.py",
    # new-module examples (module 59-61)
    "code/recommenders/01_matrix_factorization_sgd.py",
    "code/recommenders/02_als_vs_sgd.py",
    "code/graph/01_gcn_numpy.py",
    "code/graph/02_link_prediction.py",
    "code/rl/01_qlearning_gridworld.py",
    # ---- sub-course labs (llm_course + genai_agents_course) ----
    # entries may be a plain path (120s timeout) or
    # (path, timeout, [extra argv]) for slower labs.
    ("llm_course/kv_cache_lab.py", 120, []),
    ("llm_course/speculative_decoding_lab.py", 300, []),
    # model-size sweep: S/M/L x 300 steps each (full 450-step default is
    # ~4-8 min; the quick arg keeps CI bounded while checks still hold)
    ("llm_course/scale_sweep_lab.py", 600, ["--steps", "300"]),
    ("llm_course/lora_finetune_lab.py", 300, []),
    ("llm_course/monitoring_lab.py", 120, []),
    # quick-mode mini-GPT run (default 1000 steps is too slow for CI)
    ("llm_course/mini_gpt_lab.py", 300, ["--max-steps", "300", "--eval-every", "150"]),
    ("genai_agents_course/agent_lab.py", 120, []),
    ("genai_agents_course/multi_agent_lab.py", 120, []),
    ("genai_agents_course/rag_agent_server.py", 120, []),
    ("genai_agents_course/rag_agent_server_prod.py", 120, []),
    ("genai_agents_course/embedding_rag_lab.py", 120, []),
    # offline-only: real HF embeddings only when a model is cached locally
    ("genai_agents_course/embedding_hf_bench.py", 240, []),
    ("genai_agents_course/EXAMPLE.py", 120, []),
    ("llm_course/EXAMPLE.py", 120, []),
    # ---- ml_course (ultra-deep ML sub-course) ----
    ("ml_course/EXAMPLE.py", 180, []),
]

# (path, reason) - verified infrastructure/integration examples
SKIP = [
    ("code/visualization/01_matplotlib_basics.py",
     "writes PNG artifacts; run manually (see diagrams/graphs/)"),
    ("code/dl/01_keras_mlp_mnist.py", "trains MNIST (30 epochs by default); "
                                      "verified earlier with EPOCHS=3"),
    ("code/cnn/01_pytorch_cnn.py", "trains MNIST 5 epochs + downloads"),
    ("code/rnn/01_keras_lstm_sine.py", "trains 30 epochs (MSE 0.003, verified)"),
    ("code/transformers/01_hf_pipeline.py", "downloads HF models"),
    ("code/fastapi/01_fastapi_ml_api.py", "runs a server (uvicorn)"),
    ("code/flask/01_flask_ml_api.py", "runs a server"),
    ("code/docker/app.py", "runs a Flask server inside a container"),
    ("code/mlops/01_mlflow_example.py", "writes an mlruns store; needs mlflow"),
]


def run_examples():
    results = []
    tmpdir = tempfile.mkdtemp(prefix="verify_run_")
    for entry in RUN:
        if isinstance(entry, str):
            rel, timeout, extra = entry, 120, []
        else:
            rel, timeout, extra = entry[0], entry[1], (entry[2] if len(entry) > 2 else [])
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            results.append((rel, "MISSING", 0.0, "file not found"))
            continue
        # Run every example in a temp cwd so demo-generated artifacts
        # (PNGs, CSVs, plots) never pollute the course tree. All runnable
        # examples are self-contained (synthetic/sklearn data).
        cwd = tmpdir
        # Windows console is cp1252 by default, so code examples that print
        # emoji (unicode) crash when stdout is a pipe. Force UTF-8 mode, and
        # force matplotlib to the Agg (headless) backend so plt.show() in
        # examples never blocks inside a captured subprocess.
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["MPLBACKEND"] = "Agg"
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-X", "utf8", "-u", p] + extra,
                               cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=timeout)
            dt = time.time() - t0
            if r.returncode == 0:
                results.append((rel, "PASS", dt, ""))
            else:
                tail = (r.stderr or r.stdout).strip().splitlines()[-4:]
                results.append((rel, "FAIL", dt, " | ".join(tail)))
        except subprocess.TimeoutExpired:
            results.append((rel, "TIMEOUT", timeout, f"killed after {timeout}s"))
    return results, tmpdir


def report(compile_total, compile_bad, results, skipped, tmpdir):
    print("=" * 78)
    print("COURSE HEALTH SUMMARY")
    print("=" * 78)
    print(f"\n1. SYNTAX GATE: py_compile over {compile_total} .py files -> "
          f"{'ALL PASS' if not compile_bad else str(len(compile_bad)) + ' FAILED'}")
    for p, e in compile_bad:
        print(f"   FAIL {p}: {e}")

    print(f"\n2. RUNNABLE EXAMPLES ({len(results)} files, per-file timeouts)")
    print(f"   {'example':<48} {'result':<8} {'time':>7}")
    n_pass = n_fail = 0
    for rel, res, dt, tail in sorted(results, key=lambda x: x[1]):
        mark = {"PASS": "PASS", "FAIL": "FAIL", "TIMEOUT": "TIMEOUT",
                "MISSING": "MISSING"}[res]
        if res == "PASS":
            n_pass += 1
        else:
            n_fail += 1
        print(f"   {rel:<48} {mark:<8} {dt:6.1f}s")
        if tail and res != "PASS":
            print(f"        -> {tail}")

    print(f"\n3. SKIPPED (infra/heavy/network) - {len(skipped)} files")
    for rel, reason in skipped:
        print(f"   {rel:<48} {reason}")

    # structural checks
    empty = []
    n_mods = set()
    for d in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, d)
        if os.path.isdir(full) and d[0:2].isdigit():
            n_mods.add(d.split("_")[0])
            if not os.listdir(full):
                empty.append(d)
    print(f"\n4. STRUCTURE: {len(n_mods)} module prefixes "
          f"({'UNIQUE' if len(n_mods) == len([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and d[0:2].isdigit()]) else 'COLLISION!'})")
    print(f"   empty directories: {len(empty)}")

    print("\n" + "=" * 78)
    verdict = "HEALTHY" if not compile_bad and n_fail == 0 else "ISSUES FOUND"
    print(f"VERDICT: {verdict}  | compiled {compile_total} | ran {len(results)} "
          f"({n_pass} pass, {n_fail} fail) | skipped {len(skipped)}")
    print("=" * 78)


if __name__ == "__main__":
    compile_total, compile_bad = compile_all()
    results, tmpdir = run_examples()
    report(compile_total, compile_bad, results, SKIP, tmpdir)
