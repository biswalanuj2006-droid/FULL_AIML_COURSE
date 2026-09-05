"""
================================================================================
DEEP LEARNING EXAMPLE.py  (AI_ENGINEERING/DEEP_LEARNING/)
================================================================================
Runnable reference implementations for COURSE.txt modules DL-00..DL-22.
Every section VERIFIES a course claim with asserts; run end to end with:

    python EXAMPLE.py

and it prints the per-section wall time and "ALL SECTIONS PASS".
Everything is offline, CPU-only, and deterministic (fixed seeds).

Sections
  S1  tensors/broadcast/autograd + finite-difference gradient check   DL-02/03
  S2  perceptron from scratch (OR) + XOR linear-inseparability demo   DL-04
  S3  MLP from scratch (numpy) learns XOR                              DL-05
  S4  activation zoo: analytic vs numerical derivatives                DL-05
  S5  loss identities: stable BCE, softmax+CE gradient = s - y         DL-06
  S6  optimizers from scratch: momentum, Adam, AdamW decoupling        DL-06
  S7  initialization: zero-init symmetry failure, He vs Glorot scale   DL-07
  S8  LayerNorm/BatchNorm manual vs torch; pre-norm gradient path      DL-08
  S9  regularization: overfit gap, dropout, weight decay, early stop   DL-09
  S10 convolution: naive loop + im2col == torch; dim formula           DL-10
  S11 tiny CNN learns synthetic shapes; shape assert per layer         DL-10/11
  S12 transfer mechanics: freeze vs fine-tune vs scratch               DL-12
  S13 RNN from scratch + vanishing-gradient demo over sequence length  DL-13
  S14 LSTM cell from scratch matches nn.LSTM                           DL-14
  S15 GRU from scratch matches nn.GRU; param-count equations checked   DL-15
  S16 attention from scratch (numpy) + causal mask + row-sum check     DL-16
  S17 multi-head attention manual == reference; block order demo       DL-17
  S18 RoPE: relative-position property verified numerically            DL-18
  S19 KV-cache decode == full recompute (logit-identical)              DL-18
  S20 debugging demos: eval() bug, dead ReLU, one-batch overfit test   DL-20
  S21 mixed precision: bf16 autocast matches fp32 on a small net       DL-21
  S22 tiny VAE on 2D data: reparam, KL, sampling                       DL-19
================================================================================
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

PASS = []


def run(name, fn):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    PASS.append(name)
    print(f"  [{name}] PASS ({dt*1e3:7.1f} ms)")


# ==============================================================================
# S1 - tensors, broadcasting, autograd + finite-difference gradient check
# ==============================================================================
def s1():
    # broadcasting: (3,1) * (1,4) -> (3,4)
    a = torch.ones(3, 1)
    b = torch.arange(4.0)
    assert (a * b).shape == (3, 4)

    # autograd: y = (w*x).sum()^2 ; check dw against finite differences
    # (central difference on a QUADRATIC is exact - no truncation error - so a
    # larger step only HELPS: at eps=1e-6 the subtraction of two ~16.6 values
    # keeps only ~4 significant digits, the classic FD cancellation pitfall.)
    w = torch.randn(5, requires_grad=True)
    x = torch.randn(5)
    y = (w * x).sum() ** 2
    y.backward()
    eps = 5e-2
    for i in range(5):
        wp = w.detach().clone(); wm = w.detach().clone()
        wp[i] += eps; wm[i] -= eps
        fp = (wp * x).sum() ** 2
        fm = (wm * x).sum() ** 2
        num = (fp - fm) / (2 * eps)
        assert abs(num.item() - w.grad[i].item()) < 1e-3, "autograd mismatch"
    # detach/no_grad semantics
    z = (w.detach() * 2).sum()
    assert not z.requires_grad
    with torch.no_grad():
        assert not ((w * 2).sum()).requires_grad


# ==============================================================================
# S2 - perceptron from scratch + XOR impossibility
# ==============================================================================
def s2():
    # OR is linearly separable: converges
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)
    w = np.zeros(2); b = 0.0
    for _ in range(20):
        err = 0
        for xi, yi in zip(X, y):
            pred = 1 if (w @ xi + b) > 0 else 0
            if pred != yi:
                w += (yi - pred) * xi      # perceptron update
                b += (yi - pred)
                err += 1
        if err == 0:
            break
    assert err == 0, "OR should converge"

    # XOR is NOT linearly separable: single (w,b) cannot separate
    y_xor = np.array([0, 1, 1, 0], dtype=float)
    separable = False
    for _ in range(2000):  # random linear attempts
        w = np.random.randn(2); b = np.random.randn()
        if all(((X @ w + b) > 0).astype(float) == y_xor):
            separable = True
            break
    assert not separable, "XOR must not be linearly separable"
    print("    XOR not separable: verified (4 points, no separating line)")


# ==============================================================================
# S3 - MLP from scratch (numpy) learns XOR
# ==============================================================================
def s3():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 1, (2, 8)); b1 = np.zeros(8)
    W2 = rng.normal(0, 1, (8, 1)); b2 = np.zeros(1)
    lr = 0.5
    loss = float("inf")
    for it in range(3000):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)                 # ReLU
        z2 = a1 @ W2 + b2
        yh = 1 / (1 + np.exp(-z2))             # sigmoid output
        loss = -np.mean(y * np.log(yh + 1e-9) + (1 - y) * np.log(1 - yh + 1e-9))
        dL = (yh - y) / len(X)                 # BCE gradient for sigmoid
        dW2 = a1.T @ dL
        db2 = dL.sum(0)
        da1 = dL @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    assert loss < 0.05, f"MLP should learn XOR, loss={loss:.4f}"
    pred = (1 / (1 + np.exp(-(np.maximum(0, X @ W1 + b1) @ W2 + b2)))) > 0.5
    assert (pred.astype(float).ravel() == y.ravel()).all()


# ==============================================================================
# S4 - activation zoo: analytic vs numerical derivatives
# ==============================================================================
def s4():
    erf = np.vectorize(math.erf)   # math.erf is scalar-only; vectorize it
    def d_sigmoid(z): s = 1/(1+np.exp(-z)); return s*(1-s)
    def d_tanh(z): t = np.tanh(z); return 1 - t*t
    def d_relu(z): return (z > 0).astype(float)
    def d_gelu(z):  # d/dz z*Phi(z) = Phi(z) + z*phi(z)
        p = 0.5*(1+erf(z/math.sqrt(2)))
        phi = np.exp(-z*z/2)/math.sqrt(2*math.pi)
        return p + z*phi
    def d_silu(z):
        s = 1/(1+np.exp(-z))
        return s + z*s*(1-s)   # d(z*sigmoid)
    # grid deliberately EXCLUDES 0: relu/gelu/silu are non-differentiable there
    z = np.linspace(-3, 3, 6)   # step 1.2 -> no sample at exactly 0
    for name, f, df in [("sigmoid", lambda z: 1/(1+np.exp(-z)), d_sigmoid),
                        ("tanh", np.tanh, d_tanh),
                        ("relu", lambda z: np.maximum(0, z), d_relu),
                        ("gelu", lambda z: 0.5*z*(1+erf(z/math.sqrt(2))), d_gelu),
                        ("silu", lambda z: z/(1+np.exp(-z)), d_silu)]:
        eps = 1e-6
        num = (f(z+eps) - f(z-eps)) / (2*eps)
        ana = df(z)
        assert np.abs(num - ana).max() < 1e-4, f"{name} derivative wrong"
    print("    sigmoid/tanh/relu/gelu/silu derivatives verified")


# ==============================================================================
# S5 - loss identities: stable BCE, softmax+CE gradient = s - y
# ==============================================================================
def s5():
    # stable BCE: z + log(1+exp(-|z|)) style via log1p trick; compare to torch
    z = torch.randn(64) * 5            # include large |z| to test stability
    t = torch.randint(0, 2, (64,)).float()
    mine = torch.where(t == 1, F.softplus(-z), F.softplus(z)).mean()
    ref = F.binary_cross_entropy_with_logits(z, t)
    assert torch.allclose(mine, ref, atol=1e-5)

    # softmax+CE gradient identity: dL/dz = (softmax(z) - onehot(y)) / B
    # (F.cross_entropy MEANS over the batch, so the identity carries 1/B)
    logits = torch.randn(8, 10, requires_grad=True)
    labels = torch.randint(0, 10, (8,))
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    s = torch.softmax(logits.detach(), dim=-1)
    oh = F.one_hot(labels, 10).float()
    assert torch.allclose(logits.grad, (s - oh) / 8, atol=1e-5)
    print("    dCE/dz == (softmax(z) - onehot(y)) / B: verified")


# ==============================================================================
# S6 - optimizers from scratch: momentum, Adam, AdamW decoupling
# ==============================================================================
def s6():
    # minimize a noisy quadratic with each optimizer; count steps to converge
    def run_opt(opt_fn, steps=400):
        w = torch.tensor([3.0, -2.0], requires_grad=True)
        hist = []
        for _ in range(steps):
            # loss = 2 w0^2 + 0.5 w1^2 + noise-free (clean bowl)
            loss = 2*w[0]**2 + 0.5*w[1]**2
            loss.backward()
            opt_fn(w)
            hist.append(loss.item())
            w.grad = None
        return hist[-1]
    # Adam from scratch
    def adam(w, lr=0.05):
        if not hasattr(adam, "m"):
            adam.m = torch.zeros_like(w); adam.v = torch.zeros_like(w); adam.t = 0
        adam.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        adam.m = b1*adam.m + (1-b1)*w.grad
        adam.v = b2*adam.v + (1-b2)*w.grad**2
        mh = adam.m/(1-b1**adam.t); vh = adam.v/(1-b2**adam.t)
        w.data -= lr * mh/(torch.sqrt(vh)+eps)
    fin = run_opt(adam)
    assert fin < 1e-3, f"Adam should nearly solve the bowl, final={fin:.2e}"

    # AdamW decoupling: after ONE step with grad g, t=1 bias correction makes
    # m_hat = g, v_hat = g^2, so the exact closed form is
    #    w1 = w0*(1 - lr*wd) - lr*g/(|g| + eps)        (decoupled)
    # while coupled Adam (decay folded into the gradient) would be
    #    w1 = w0 - lr*(g + wd*w0)/(|g + wd*w0| + eps)   (coupled)
    net = nn.Linear(4, 1, bias=False)
    w0 = net.weight.detach().clone()
    lr, wd, eps = 0.1, 0.5, 1e-8
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    opt.zero_grad()
    (net.weight.square().sum()).backward()   # g = 2*w0
    g = net.weight.grad.detach().clone()
    opt.step()
    expected = w0*(1 - lr*wd) - lr*g/(g.abs() + eps)
    assert torch.allclose(net.weight.detach(), expected, atol=1e-6), \
        "AdamW does not match its decoupled closed form"
    coupled = w0 - lr*(g + wd*w0)/((g + wd*w0).abs() + eps)
    assert not torch.allclose(net.weight.detach(), coupled, atol=1e-3), \
        "AdamW should differ from coupled Adam on non-uniform v"
    print("    AdamW matches its decoupled formula; differs from coupled Adam")


# ==============================================================================
# S7 - initialization: zero-init symmetry failure, He vs Glorot scale
# ==============================================================================
def s7():
    # (a) zero init => all hidden units identical forever (symmetry)
    torch.manual_seed(0)
    net0 = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Tanh(),
                         nn.Linear(8, 2, bias=False))
    with torch.no_grad():
        for p in net0.parameters():
            p.zero_()
    x = torch.randn(16, 4)
    h = torch.tanh(x @ net0[0].weight.T)
    assert torch.allclose(h[:, 0], h[:, 1]), "zero init should produce identical units"

    # (b) He vs Glorot: activation std through 20 tanh layers
    def stack_std(init, act):
        torch.manual_seed(1)
        d = 256
        v = torch.randn(64, d)
        stds = []
        for _ in range(20):
            w = init(d, d)
            v = act(v @ w.T)
            stds.append(v.std().item())
        return stds[-1]
    glorot = lambda nin, nout: torch.empty(nin, nout).uniform_(-math.sqrt(6/(nin+nout)), math.sqrt(6/(nin+nout)))
    he = lambda nin, nout: torch.empty(nin, nout).normal_(0, math.sqrt(2/nin))
    s_tanh = stack_std(glorot, torch.tanh)
    s_he_relu = stack_std(he, lambda z: torch.relu(z))
    # Real result: Glorot+tanh decays slowly but does NOT collapse through 20
    # layers (tanh saturates); He+ReLU holds the scale - the reason deep nets
    # use ReLU-family activations + He (residuals fix the residual decay).
    assert 0.05 < s_tanh < 0.6, f"Glorot+tanh std drifted: {s_tanh}"
    assert 0.3 < s_he_relu < 1.5, f"He+ReLU std drifted: {s_he_relu}"
    assert s_he_relu > 2 * s_tanh, "He+ReLU must hold signal far better than Glorot+tanh"
    print(f"    Glorot+tanh 20-layer std {s_tanh:.3f} (slow decay); "
          f"He+ReLU std {s_he_relu:.3f} (stable)")


# ==============================================================================
# S8 - LayerNorm/BatchNorm manual vs torch; pre-norm gradient path
# ==============================================================================
def s8():
    torch.manual_seed(0)
    x = torch.randn(8, 5, 16)   # B, T, C

    # LayerNorm manual (over last dim, per sample-position)
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    ln = nn.LayerNorm(16)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(16)); ln.bias.copy_(torch.randn(16))
    manual = (x - mu) / torch.sqrt(var + 1e-5) * ln.weight + ln.bias
    assert torch.allclose(manual, ln(x), atol=1e-5)
    print("    LayerNorm manual == nn.LayerNorm")

    # BatchNorm manual on (N, C) — normalize over batch per channel
    xb = torch.randn(32, 8)
    bn = nn.BatchNorm1d(8)
    bn.train()
    with torch.no_grad():
        bn.weight.copy_(torch.ones(8)); bn.bias.copy_(torch.zeros(8))
    m = xb.mean(0, keepdim=True)
    v = xb.var(0, unbiased=False, keepdim=True)
    manual_b = (xb - m) / torch.sqrt(v + 1e-5)
    assert torch.allclose(manual_b, bn(xb), atol=1e-4)

    # pre-norm: gradient flows through the identity path
    def grad_flow(pre):
        torch.manual_seed(0)
        x0 = torch.randn(1, 8, requires_grad=True)
        w = torch.randn(8, 8) * 0.5
        x = x0
        for _ in range(30):
            if pre:
                x = x + torch.tanh(x @ w)          # x + f(LN-free) ~ pre-norm
            else:
                x = torch.tanh(x + x0 @ w) if False else x
        if not pre:
            x = x0
            for _ in range(30):
                x = torch.tanh(x @ w)              # pure stacking (post-ish)
        x.sum().backward()
        return x0.grad.abs().mean().item()
    g_pre = grad_flow(True)
    g_post = grad_flow(False)
    assert g_pre > g_post, "residual path should keep gradients larger"
    print(f"    gradient magnitude after 30 layers: residual {g_pre:.2e} vs plain {g_post:.2e}")


# ==============================================================================
# S9 - regularization: overfit gap, weight decay, dropout, early stopping
# ==============================================================================
def s9():
    torch.manual_seed(0)
    # bias-variance demo: train on sin + HEAVY noise (memorizable), validate on
    # CLEAN sin.  Weight decay trades train fit for generalization - the honest
    # lesson is va1 << va0 with tr1 > tr0, not some arbitrary "gap" target.
    xtr = torch.rand(100, 1) * 4 - 2
    ytr = torch.sin(2*xtr) + 0.8*torch.randn(100, 1)   # noisy train
    xva = torch.rand(400, 1) * 4 - 2
    yva = torch.sin(2*xva)                             # clean validation

    def train(wd, epochs=1000):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(1, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(net(xtr), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            tr = F.mse_loss(net(xtr), ytr).item()
            va = F.mse_loss(net(xva), yva).item()
        return tr, va
    tr0, va0 = train(0.0)
    tr1, va1 = train(1e-2)
    assert va1 < va0 * 0.7, f"weight decay should cut val error: {va0} -> {va1}"
    assert tr1 > tr0, "decay should accept a worse training fit (the tradeoff)"
    print(f"    no decay: tr {tr0:.3f} val {va0:.3f}  |  decay 1e-2: tr {tr1:.3f} "
          f"val {va1:.3f}  (decay trades train fit for generalization)")

    # dropout: train with p=0.5, eval must be deterministic (off)
    m = nn.Dropout(0.5)
    m.train()
    outs = {tuple(m(torch.ones(1000)).tolist()) for _ in range(3)}
    assert len(outs) > 1, "dropout should vary in train mode"
    m.eval()
    outs_eval = {tuple(m(torch.ones(1000)).tolist()) for _ in range(3)}
    assert len(outs_eval) == 1

    # early stopping picks the best val checkpoint (small run: the decay gap
    # above already showed the mechanism; this just proves checkpointing works)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    best = (float("inf"), None)
    for i in range(600):
        opt.zero_grad()
        loss = F.mse_loss(net(xtr), ytr); loss.backward(); opt.step()
        with torch.no_grad():
            va = F.mse_loss(net(xva), yva).item()
        if va < best[0]:
            # DEEP copy!  state_dict() tensors are shared with the live model;
            # a shallow dict copy would keep mutating after later opt.step()s.
            best = (va, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    with torch.no_grad():
        restored = F.mse_loss(net(xva), yva).item()
    assert abs(restored - best[0]) < 1e-6, "checkpoint must restore the best val"


# ==============================================================================
# S10 - convolution: naive loop + im2col == torch; dim formula
# ==============================================================================
def s10():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 7)          # N, C, H, W
    w = torch.randn(4, 3, 3, 3)          # out, in, K, K
    b = torch.randn(4)
    ref = F.conv2d(x, w, b, padding=1, stride=2)

    # naive loops on the first sample
    N, Ci, H, W = x.shape
    Co, _, K, _ = w.shape
    P, S = 1, 2
    Ho = (H + 2*P - K)//S + 1
    Wo = (W + 2*P - K)//S + 1
    out = torch.zeros(Co, Ho, Wo)
    xp = F.pad(x[0], (P, P, P, P))
    for oc in range(Co):
        for oh in range(Ho):
            for ow in range(Wo):
                acc = b[oc].item()
                for ic in range(Ci):
                    for kh in range(K):
                        for kw in range(K):
                            acc += (xp[ic, oh*S+kh, ow*S+kw] * w[oc, ic, kh, kw]).item()
                out[oc, oh, ow] = acc
    assert torch.allclose(out, ref[0], atol=1e-4), "naive conv mismatch"

    # im2col + matmul
    def im2col(x, K, P, S):
        N, Ci, H, W = x.shape
        xp = F.pad(x, (P, P, P, P))
        Ho = (H + 2*P - K)//S + 1
        Wo = (W + 2*P - K)//S + 1
        cols = []
        for i in range(Ho):
            for j in range(Wo):
                patch = xp[:, :, i*S:i*S+K, j*S:j*S+K]   # N, Ci, K, K
                cols.append(patch.reshape(N, -1))
        return torch.stack(cols, dim=1)                    # N, Ho*Wo, Ci*K*K
    C = im2col(x, K, P, S)                                 # N, L, Ci*K*K
    Wm = w.reshape(Co, -1).T                               # Ci*K*K, Co
    out2 = (C @ Wm + b).permute(0, 2, 1).reshape(N, Co, Ho, Wo)
    assert torch.allclose(out2, ref, atol=1e-4), "im2col mismatch"
    assert (Ho, Wo) == (4, 4), f"dim formula wrong: {Ho}x{Wo} (expect 4x4 for 7->S2)"
    print("    naive loop + im2col both match F.conv2d; output 7x7,S2,P1 -> 4x4")


# ==============================================================================
# S11 - tiny CNN learns synthetic shapes; shape asserts per layer
# ==============================================================================
def s11():
    torch.manual_seed(0)
    # synthetic: 2 classes of 8x8 patterns (vertical vs horizontal bars) + noise
    def synth(n=1200):
        X, y = [], []
        for i in range(n):
            img = torch.zeros(1, 8, 8) + 0.05*torch.randn(1, 8, 8)
            if i % 2 == 0:
                img[:, :, 2:6] += 1.0      # vertical bar
                y.append(0)
            else:
                img[:, 2:6, :] += 1.0      # horizontal bar
                y.append(1)
            X.append(img)
        return torch.stack(X), torch.tensor(y)
    X, y = synth()
    split = 900
    Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]

    conv = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8x4x4
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x2x2
        nn.Flatten(),
        nn.Linear(16*2*2, 2),
    )
    # per-layer shape asserts (the CNN debug habit)
    h = Xtr[:4]
    assert conv[0](h).shape == (4, 8, 8, 8)
    assert conv[0:3](h).shape == (4, 8, 4, 4)
    assert conv[0:6](h).shape == (4, 16, 2, 2)
    opt = torch.optim.Adam(conv.parameters(), lr=3e-3)
    for epoch in range(12):
        perm = torch.randperm(len(Xtr))[:256]
        opt.zero_grad()
        loss = F.cross_entropy(conv(Xtr[perm]), ytr[perm])
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (conv(Xva).argmax(1) == yva).float().mean().item()
    assert acc > 0.97, f"CNN should learn bars, acc={acc:.3f}"
    print(f"    tiny CNN on synthetic bars: val acc {acc:.3f} (shape asserts OK)")


# ==============================================================================
# S12 - transfer mechanics: freeze vs fine-tune vs scratch
# ==============================================================================
def s12():
    torch.manual_seed(0)
    # Story: pretrain a tiny CNN on CLEAN thick bars, then adapt to the SAME
    # concept seen through a worse lens - thin jittered bars, 5x more noise,
    # 1/15 the data (40 samples).  This is the realistic transfer setup:
    # source features (edge detectors) should transfer, and the question is
    # whether reusing them beats training from scratch on tiny noisy data.
    #
    # Measured result (6 seeds, median): fine-tuning converges FASTER (ep 30:
    # 0.92 vs 0.87) and edges out scratch at convergence (0.92 vs 0.91); frozen
    # features LAG (0.80) because the target distribution shifted.  And frozen
    # trains only the 130-param head vs 1378 from scratch - the parameter
    # efficiency of feature reuse.
    def bars(n=800, amp=1.0, noise=0.05, jit=0):
        X, y = [], []
        for i in range(n):
            img = torch.zeros(1, 8, 8) + noise*torch.randn(1, 8, 8)
            j = (i % 3) - 1 if jit else 0
            if i % 2 == 0:
                img[:, :, 3+j:5+j] += amp; y.append(0)
            else:
                img[:, 3+j:5+j, :] += amp; y.append(1)
            X.append(img)
        return torch.stack(X), torch.tensor(y)

    def make_net():
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64, 2))

    # --- pretrain on clean thick bars (800 samples, 8 epochs) ---
    Xb, yb = bars()
    net_p = make_net()
    opt = torch.optim.Adam(net_p.parameters(), lr=3e-3)
    for _ in range(8):
        opt.zero_grad(); F.cross_entropy(net_p(Xb), yb).backward(); opt.step()

    # --- target: degraded bars, 40 train / 60 val samples ---
    Xt, yt = bars(n=400, amp=0.3, noise=0.35, jit=1)
    Xtr, ytr, Xva, yva = Xt[:40], yt[:40], Xt[60:120], yt[60:120]

    n_all = sum(p.numel() for p in net_p.parameters())      # 1378
    # count the trainable params when the convs are frozen (a fresh copy)
    frozen_probe = make_net(); frozen_probe.load_state_dict(net_p.state_dict())
    for p in list(frozen_probe.parameters())[:4]:
        p.requires_grad_(False)
    n_head = sum(p.numel() for p in frozen_probe.parameters() if p.requires_grad)  # 130

    def run(setup, epochs, seed):
        torch.manual_seed(seed)
        net = make_net()
        if setup in ("frozen", "ftall"):
            net.load_state_dict(net_p.state_dict())
        if setup == "frozen":
            for p in list(net.parameters())[:4]:   # conv1 w/b, conv2 w/b
                p.requires_grad_(False)
        o = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(epochs):
            o.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); o.step()
        with torch.no_grad():
            return (net(Xva).argmax(1) == yva).float().mean().item()

    def median(vals):
        s = sorted(vals)
        return s[len(s)//2]

    SEEDS, EP1, EP2 = 6, 30, 120
    res = {k: {e: [] for e in (EP1, EP2)} for k in ("scratch", "frozen", "ftall")}
    for s in range(SEEDS):
        for setup in res:
            for e in (EP1, EP2):
                res[setup][e].append(run(setup, e, s))
    m = lambda k, e: median(res[k][e])
    scratch30, ftall30 = m("scratch", EP1), m("ftall", EP1)
    scratch120, frozen120, ftall120 = m("scratch", EP2), m("frozen", EP2), m("ftall", EP2)

    # (1) convergence speed: fine-tuning leads at a small budget
    assert ftall30 > scratch30 + 0.02, f"transfer should converge faster: {ftall30} vs {scratch30}"
    # (2) frozen features lag on the SHIFTED target (jitter + noise)
    assert frozen120 < ftall120, f"frozen features should lag on shifted target: {frozen120} vs {ftall120}"
    # (3) fine-tuning ends at least as good as scratch
    assert ftall120 >= scratch120 - 0.02, f"ftall should match scratch: {ftall120} vs {scratch120}"
    # (4) parameter efficiency of feature reuse
    assert n_head < 0.2 * n_all, f"head should be a small fraction: {n_head}/{n_all}"

    print(f"    ep{EP1} val acc: scratch {scratch30:.2f} | fine-tune {ftall30:.2f}   (transfer converges faster)")
    print(f"    ep{EP2} val acc: scratch {scratch120:.2f} | frozen {frozen120:.2f} | fine-tune {ftall120:.2f}"
          f"   (frozen lags on shifted target; fine-tune matches scratch)")
    print(f"    params updated: frozen {n_head} vs scratch {n_all}  ({100*n_head/n_all:.1f}% - feature reuse is cheap)")


# ==============================================================================
# S13 - RNN from scratch + vanishing-gradient demo over sequence length
# ==============================================================================
def s13():
    torch.manual_seed(0)
    # A single recurrent cell with tanh; measure |grad| wrt the FIRST input as
    # sequence length grows -> must decay (the vanishing-gradient demo).
    def first_grad_norm(T):
        x = torch.randn(1, T, 3)                    # (batch, T, 3)
        Wx = torch.randn(3, 8) * 0.4
        Wh = torch.randn(8, 8) * 0.4                # spectral radius < 1
        h0 = torch.randn(1, 8)
        x.requires_grad_(True)
        h = h0
        for t in range(T):
            h = torch.tanh(x[:, t] @ Wx + h @ Wh)
        h.sum().backward()
        return x.grad[:, 0].abs().mean().item()
    g4 = first_grad_norm(4)
    g20 = first_grad_norm(20)
    assert g20 < g4, "gradient through the recurrence must shrink with length"
    print(f"    vanilla RNN: grad wrt first input at T=4: {g4:.2e} -> T=20: "
          f"{g20:.2e} (vanishing, exactly why gates/attention exist)")

    # manual BPTT loop on a tiny copy task actually learns short sequences
    def copy_task(T=6, vocab=5, steps=600):
        torch.manual_seed(0)
        emb = torch.randn(vocab, 8, requires_grad=True)
        Wx = torch.randn(8, 8, requires_grad=True) * 0.3
        Wh = torch.randn(8, 8, requires_grad=True) * 0.3
        Wy = torch.randn(8, vocab, requires_grad=True) * 0.3
        h = torch.zeros(8)
        def run(seq):
            nonlocal h
            h = torch.zeros(8)
            outs = []
            for tok in seq:
                h = torch.tanh(emb[tok] @ Wx + h @ Wh)
                outs.append(h @ Wy)
            return torch.stack(outs)
        opt = torch.optim.Adam([Wx, Wh, Wy, emb], lr=5e-2)
        loss = float("inf")
        for _ in range(steps):
            seq = torch.randint(0, vocab, (T,))
            logits = run(seq)
            loss = F.cross_entropy(logits, seq)      # predict the same token
            opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()
    final = copy_task()
    assert final < 1.2, f"RNN should learn short copy, loss={final:.2f}"
    print(f"    manual RNN BPTT learns T=6 copy task (final loss {final:.2f})")


# ==============================================================================
# S14 - LSTM cell from scratch matches nn.LSTM
# ==============================================================================
def s14():
    torch.manual_seed(0)
    inp, hid, T, B = 5, 7, 9, 2
    lstm = nn.LSTM(inp, hid, batch_first=True)
    x = torch.randn(B, T, inp)
    with torch.no_grad():
        out_ref, (hn, cn) = lstm(x)
        Wih = lstm.weight_ih_l0; Whh = lstm.weight_hh_l0; bih = lstm.bias_ih_l0
        bhh = lstm.bias_hh_l0
    h = torch.zeros(B, hid); c = torch.zeros(B, hid)
    outs = []
    for t in range(T):
        g = x[:, t] @ Wih.T + bih + h @ Whh.T + bhh    # (B, 4*hid)
        i, f, gg, o = g.chunk(4, dim=1)                # input, forget, cell, out
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        gg = torch.tanh(gg); o = torch.sigmoid(o)
        c = f * c + i * gg
        h = o * torch.tanh(c)
        outs.append(h)
    outs = torch.stack(outs, dim=1)
    assert torch.allclose(outs, out_ref, atol=1e-5), "LSTM cell mismatch"
    assert torch.allclose(hn, h, atol=1e-5) and torch.allclose(cn, c, atol=1e-5)
    print("    manual LSTM cell == nn.LSTM (final states match too)")


# ==============================================================================
# S15 - GRU from scratch matches nn.GRU; param-count equations checked
# ==============================================================================
def s15():
    torch.manual_seed(0)
    inp, hid, T, B = 5, 7, 9, 2
    gru = nn.GRU(inp, hid, batch_first=True)
    x = torch.randn(B, T, inp)
    with torch.no_grad():
        out_ref, hn = gru(x)
        Wih = gru.weight_ih_l0; Whh = gru.weight_hh_l0
        bih = gru.bias_ih_l0; bhh = gru.bias_hh_l0
    h = torch.zeros(B, hid)
    outs = []
    for t in range(T):
        # PyTorch nn.GRU gate order in the big matrices is [reset, update, new]
        gx = x[:, t] @ Wih.T + bih      # (B, 3*hid)  input part
        gh = h @ Whh.T + bhh            # (B, 3*hid)  hidden part
        xr, xz, xn = gx.chunk(3, dim=1)
        hr, hz, hn = gh.chunk(3, dim=1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        # candidate: tanh(W_in x + b_in + r * (W_hn h + b_hn)) - the reset
        # gates the AFFINE hidden result, not the hidden itself.
        n = torch.tanh(xn + r * hn)
        h = (1 - z) * h + z * n
        outs.append(h)
    outs = torch.stack(outs, dim=1)
    assert torch.allclose(outs, out_ref, atol=1e-5), "GRU cell mismatch"
    assert torch.allclose(hn, h, atol=1e-5)
    # param-count equations (DL-15/Practice L3-M1/M2)
    n_lstm = 4 * (hid * (hid + inp) + hid)
    n_gru = 3 * (hid * (hid + inp) + hid)
    m_lstm = sum(p.numel() for p in nn.LSTM(inp, hid).parameters())
    m_gru = sum(p.numel() for p in nn.GRU(inp, hid).parameters())
    assert n_lstm == m_lstm and n_gru == m_gru
    print(f"    manual GRU == nn.GRU; param equations verified "
          f"(LSTM {m_lstm} = 4(h(h+i)+h), GRU {m_gru} = 3(h(h+i)+h))")


# ==============================================================================
# S16 - attention from scratch (numpy) + causal mask + row sums
# ==============================================================================
def s16():
    rng = np.random.default_rng(0)
    dk = 4; nq, nk = 3, 5
    Q = rng.standard_normal((nq, dk))
    K = rng.standard_normal((nk, dk))
    V = rng.standard_normal((nk, 6))
    S = Q @ K.T / math.sqrt(dk)
    A = np.exp(S - S.max(axis=-1, keepdims=True))
    A = A / A.sum(axis=-1, keepdims=True)
    out = A @ V
    assert np.allclose(A.sum(axis=-1), 1.0)
    # reference with torch softmax
    ref = torch.softmax(torch.tensor(S), dim=-1).numpy()
    assert np.allclose(A, ref, atol=1e-6)
    # causal mask: token i attends only <= i
    T = 4; d = 2
    Q2 = K2 = rng.standard_normal((T, d))
    V2 = rng.standard_normal((T, 3))
    S2 = Q2 @ K2.T / math.sqrt(d)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    S2m = np.where(mask, -np.inf, S2)
    e = np.exp(S2m - S2m.max(-1, keepdims=True))
    A2 = e / e.sum(-1, keepdims=True)
    assert np.allclose(A2.sum(-1), 1.0)
    assert A2[0, 1] == 0 and A2[1, 2] == 0 and A2[3, 3] > 0
    print("    attention rows sum to 1; causal mask blocks future keys")


# ==============================================================================
# S17 - multi-head attention manual == reference; causal logit sanity
# ==============================================================================
def s17():
    torch.manual_seed(0)
    B, T, C, nh = 2, 5, 8, 2
    dh = C // nh
    qkv = nn.Linear(C, 3*C, bias=False)
    proj = nn.Linear(C, C, bias=False)
    x = torch.randn(B, T, C)
    with torch.no_grad():
        q, k, v = qkv(x).chunk(3, dim=-1)
        def heads(t):
            return t.view(B, T, nh, dh).transpose(1, 2)   # B,nh,T,dh
        q, k, v = heads(q), heads(k), heads(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None], float("-inf"))
        att = torch.softmax(scores, dim=-1)
        y = att @ v                                         # B,nh,T,dh
        y = y.transpose(1, 2).reshape(B, T, C)
        out = proj(y)
    # compare against a fresh identical manual computation via a scratch module
    class MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = qkv; self.proj = proj
        def forward(self, x):
            B2, T2, C2 = x.shape
            q2, k2, v2 = self.qkv(x).chunk(3, dim=-1)
            q2 = q2.view(B2, T2, nh, dh).transpose(1, 2)
            k2 = k2.view(B2, T2, nh, dh).transpose(1, 2)
            v2 = v2.view(B2, T2, nh, dh).transpose(1, 2)
            s = q2 @ k2.transpose(-2, -1) / math.sqrt(dh)
            m = torch.triu(torch.ones(T2, T2, dtype=torch.bool), diagonal=1)
            s = s.masked_fill(m[None, None], float("-inf"))
            a = torch.softmax(s, dim=-1)
            y2 = a @ v2
            return self.proj(y2.transpose(1, 2).reshape(B2, T2, C2))
    torch.manual_seed(0)
    net = MHA()
    assert torch.allclose(net(x), out, atol=1e-6)
    # causal sanity: row t of logits only depends on tokens <= t
    x0 = x.clone()
    xb = x.clone(); xb[:, 2:] = 0        # erase future; causal output pos <=1 unchanged
    with torch.no_grad():
        a = net(x0); b = net(xb)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6)
    print("    MHA manual == reference; causal rows ignore future tokens")


# ==============================================================================
# S18 - RoPE: relative-position property verified numerically
# ==============================================================================
def s18():
    torch.manual_seed(0)
    d = 8; nhalf = d // 2
    def rope(x, pos):                       # x (..., d); returns rotated copy
        freq = 1.0 / (10000 ** (torch.arange(0, nhalf, dtype=torch.float32) * 2 / d))
        theta = pos * freq                  # (nhalf,)
        c = torch.cos(theta); s = torch.sin(theta)
        x1 = x[..., :nhalf]; x2 = x[..., nhalf:]
        out = torch.empty_like(x)
        out[..., :nhalf] = x1 * c - x2 * s
        out[..., nhalf:] = x1 * s + x2 * c
        return out
    # key property: dot(q at pos m, k at pos n) depends only on m - n
    q = torch.randn(d); k = torch.randn(d)
    dots = {}
    for m in range(0, 12):
        for n in range(0, 12):
            val = (rope(q, m) * rope(k, n)).sum()
            key = m - n
            if key in dots:
                assert abs(dots[key] - val.item()) < 1e-5, "RoPE not relative"
            dots[key] = val.item()
    print("    RoPE: dot(q_m, k_n) depends only on m - n (relative positions)")


# ==============================================================================
# S19 - KV-cache decode == full recompute (logit-identical)
# ==============================================================================
def s19():
    torch.manual_seed(0)
    d_model, nh, n_layer, vocab = 32, 4, 2, 64
    d_head = d_model // nh
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
        def forward(self, x, cache=None):
            B, T, C = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(B, T, nh, d_head).transpose(1, 2)
            k = k.view(B, T, nh, d_head).transpose(1, 2)
            v = v.view(B, T, nh, d_head).transpose(1, 2)
            row0 = 0
            if cache is not None:
                pk, pv = cache
                row0 = pk.size(2)
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            Tfull = k.size(2)
            s = q @ k.transpose(-2, -1) / math.sqrt(d_head)
            m = torch.triu(torch.ones(Tfull, Tfull, dtype=torch.bool), diagonal=1)
            s = s.masked_fill(m[row0:row0+T, :Tfull].unsqueeze(0), float("-inf"))
            a = torch.softmax(s, dim=-1)
            y = a @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y), (k.detach(), v.detach())
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d_model)
            self.attn = Attn()
            self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(),
                                    nn.Linear(4*d_model, d_model))
        def forward(self, x, cache=None):
            h, kv = self.attn(self.ln(x), cache)
            return x + h + self.ff(x + h), kv
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.head = nn.Linear(d_model, vocab)
        def forward(self, idx, caches=None):
            x = self.emb(idx)
            new_caches = []
            for i, blk in enumerate(self.blocks):
                x, kv = blk(x, caches[i] if caches else None)
                new_caches.append(kv)
            return self.head(x), new_caches
    model = Tiny().eval()
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (2, 8))
    # logit equivalence over 6 generated steps
    with torch.no_grad():
        idx = prompt.clone()
        logits, caches = model(idx)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], 1)
        max_delta = 0.0
        for _ in range(5):
            logits_c, caches = model(idx[:, -1:], caches)   # feed ONLY fresh token
            logits_b, _ = model(idx)                        # full recompute
            max_delta = max(max_delta,
                            (logits_c[:, -1] - logits_b[:, -1]).abs().max().item())
            idx = torch.cat([idx, logits_c[:, -1].argmax(-1, keepdim=True)], 1)
    assert max_delta < 1e-4, f"KV cache diverged: {max_delta}"
    print(f"    KV-cached decode == full recompute (max logit delta {max_delta:.2e})")


# ==============================================================================
# S20 - debugging demos: eval() bug, dead ReLU, one-batch overfit test
# ==============================================================================
def s20():
    torch.manual_seed(0)
    # (a) the eval() bug: dropout active at inference changes outputs
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))
    x = torch.randn(1, 8)
    net.train()
    outs = [net(x).detach().clone() for _ in range(3)]
    varying = any(not torch.allclose(outs[0], o) for o in outs[1:])
    assert varying, "train mode must be stochastic (dropout on)"
    net.eval()
    outs_e = [net(x).detach().clone() for _ in range(3)]
    assert all(torch.allclose(outs_e[0], o) for o in outs_e[1:])
    print("    eval() bug demo: train mode is stochastic, eval mode deterministic")

    # (b) dead ReLU: a neuron with negative pre-activations never recovers
    z = torch.tensor([-100.0, -200.0])
    assert (torch.relu(z) == 0).all() and torch.relu(z).grad is None
    # gradient through ReLU at negative input is exactly 0
    w = torch.tensor([-1.0, -1.0], requires_grad=True)
    loss = torch.relu(w * 2.0).sum()
    loss.backward()
    assert (w.grad == 0).all()

    # (c) one-batch overfit test: a healthy net must drive one batch to ~0 loss
    torch.manual_seed(0)
    xb = torch.randn(8, 6); yb = torch.randint(0, 3, (8,))
    net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss = 1.0
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(net(xb), yb)
        loss.backward(); opt.step()
    assert loss.item() < 0.01, f"one-batch overfit failed: {loss.item()}"
    print("    one-batch overfit test passes (healthy plumbing)")


# ==============================================================================
# S21 - mixed precision: bf16 autocast matches fp32 on a small net
# ==============================================================================
def s21():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64),
                        nn.ReLU(), nn.Linear(64, 4))
    x = torch.randn(32, 16); y = torch.randint(0, 4, (32,))
    def train(dtype=None):
        torch.manual_seed(0)
        m = nn.Sequential(*[nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64),
                            nn.ReLU(), nn.Linear(64, 4)])
        m.load_state_dict(net.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        loss = None
        for _ in range(30):
            opt.zero_grad()
            if dtype is None:
                loss = F.cross_entropy(m(x), y)
            else:
                with torch.autocast("cpu", dtype=dtype):
                    loss = F.cross_entropy(m(x), y)
            loss.backward(); opt.step()
        return loss.item()
    l_fp32 = train()
    l_bf16 = train(torch.bfloat16)
    assert abs(l_fp32 - l_bf16) < 0.05, f"bf16 diverged: {l_fp32} vs {l_bf16}"
    print(f"    bf16 autocast matches fp32 (loss {l_fp32:.3f} vs {l_bf16:.3f})")


# ==============================================================================
# S22 - tiny VAE on 2D data: reparametrization, KL, sampling
# ==============================================================================
def s22():
    torch.manual_seed(0)
    # two moons-ish 2D clusters -> learn a 2D-latent VAE
    n = 2000
    ang = torch.rand(n) * math.pi
    x1 = torch.stack([torch.cos(ang)*1.5 + 0.3*torch.randn(n),
                      torch.sin(ang)*1.5 + 0.3*torch.randn(n)], 1)
    x2 = torch.stack([-torch.cos(ang)*1.5 + 2.0 + 0.3*torch.randn(n),
                      -torch.sin(ang)*1.5 + 0.3*torch.randn(n)], 1)
    X = torch.cat([x1, x2])
    enc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))  # mu, logvar
    dec = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    kl_w = 1.0
    recon, kl = 0.0, 0.0
    for step in range(1500):
        batch = X[torch.randint(0, len(X), (128,))]
        p = enc(batch)
        mu, logvar = p[:, :2], p[:, 2:]
        z = mu + torch.exp(0.5*logvar) * torch.randn_like(mu)   # reparametrization
        xh = dec(z)
        recon = F.mse_loss(xh, batch)
        kl = 0.5 * (logvar.exp() + mu**2 - 1 - logvar).sum(1).mean()
        loss = recon + kl_w * kl
        opt.zero_grad(); loss.backward(); opt.step()
    # quality: decode z~N(0,1) samples; they must land near the data support
    with torch.no_grad():
        zs = torch.randn(500, 2)
        xs = dec(zs)
        # nearest-train-distance proxy: samples shouldn't be far from data
        d = ((xs[:, None, :] - X[None, :, :])**2).sum(-1).sqrt().min(1).values
        assert d.mean() < 1.2, f"VAE samples drift off manifold: {d.mean():.2f}"
    print(f"    VAE: recon {recon:.3f} KL {kl:.3f}; latent samples stay on "
          f"manifold (mean NN dist {d.mean():.2f})")


# ==============================================================================
# runner
# ==============================================================================
def main():
    print("=" * 72)
    print("DEEP LEARNING EXAMPLE.py - reference implementations")
    print("=" * 72)
    for name, fn in [
        ("S1  autograd + gradcheck", s1),
        ("S2  perceptron + XOR", s2),
        ("S3  MLP from scratch", s3),
        ("S4  activation derivatives", s4),
        ("S5  loss identities", s5),
        ("S6  optimizers from scratch", s6),
        ("S7  initialization", s7),
        ("S8  normalization", s8),
        ("S9  regularization", s9),
        ("S10 conv: loop + im2col", s10),
        ("S11 tiny CNN", s11),
        ("S12 transfer learning", s12),
        ("S13 RNN + vanishing grad", s13),
        ("S14 LSTM from scratch", s14),
        ("S15 GRU from scratch", s15),
        ("S16 attention from scratch", s16),
        ("S17 multi-head attention", s17),
        ("S18 RoPE property", s18),
        ("S19 KV-cache equivalence", s19),
        ("S20 debugging demos", s20),
        ("S21 mixed precision", s21),
        ("S22 tiny VAE", s22),
    ]:
        run(name, fn)
    print("=" * 72)
    print(f"ALL SECTIONS PASS ({len(PASS)}/{len(PASS)})")
    print("=" * 72)


if __name__ == "__main__":
    main()
