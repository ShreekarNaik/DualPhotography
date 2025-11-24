Below I convert the heuristic into a rigorous optimal-control / sequential experimental design formulation, derive a practical control law, show how to implement it, and explain why (and when) it outperforms Gray-code scanning. I keep assumptions explicit so the derivation is reproducible.

# 1. Vectorized linear measurement model (exact)

Vectorize the unknown transport matrix (T\in\mathbb{R}^{M\times N}) as a column vector
[
t = \operatorname{vec}(T)\in\mathbb{R}^{MN}.
]
A projector pattern (u_k\in\mathbb{R}^{N}) yields the camera measurement
[
y_k = T u_k + \eta_k \in\mathbb{R}^M.
]
This is linear in (t). Define the measurement matrix
[
H(u_k) = I_M \otimes u_k^\top \in\mathbb{R}^{M\times MN},
]
so that
[
y_k = H(u_k), t + \eta_k.
]
Assume additive Gaussian sensor noise (\eta_k\sim\mathcal N(0,R)) with (R=\sigma^2 I_M) (for tractability; non-Gaussian noise can be handled later).

# 2. Belief (state) and posterior update

Treat the belief over (t) as the state (x*k). Start with prior
[
t\sim\mathcal N(\mu_0,\Sigma_0).
]
After collecting (k) measurements ({(u_i,y_i)}*{i=1}^k), the posterior is Gaussian (\mathcal N(\mu*k,\Sigma_k)) with the standard linear Bayesian update (Kalman / Bayesian linear regression):
[
S_k \equiv H(u_k)\Sigma*{k-1}H(u_k)^\top + R,
]
[
K_k = \Sigma*{k-1}H(u*k)^\top S_k^{-1},
]
[
\mu_k = \mu*{k-1} + K*k\big(y_k - H(u_k)\mu*{k-1}\big),
]
[
\Sigma_k = \Sigma*{k-1} - K*k H(u_k)\Sigma*{k-1}.
]

The belief ((\mu_k,\Sigma_k)) is the state for a belief-space optimal control problem.

# 3. Objective: cost + reconstruction error (D-optimality / information gain)

Use a concrete, computable cost:
[
J = \mathbb{E}\Big[\sum_{k=1}^K \mathcal C(u_k) + \lambda ,\mathcal E(\hat T)\Big],
]
where (\mathcal C(u*k)) is measurement cost (e.g., time per projection), and (\mathcal E(\hat T)) is reconstruction error. For Gaussian posterior a convenient (\mathcal E) is posterior covariance volume (D-optimality):
[
\mathcal E = \log\det(\Sigma_K).
]
So minimizing (\mathcal E) is equivalent to maximizing information (minimizing uncertainty). With that choice,
[
J = \sum*{k=1}^K \mathcal C(u_k) + \lambda \log\det(\Sigma_K).
]

# 4. The exact dynamic program (in belief space)

Define the value function over beliefs:
[
V*k(\mu*{k-1},\Sigma*{k-1}) = \min*{u*k}\ \mathcal C(u_k) + \mathbb{E}*{y_k|u_k}\big[V*{k+1}(\mu_k,\Sigma_k)\big],
]
with terminal cost (V\*{K+1}(\mu_K,\Sigma_K) = \lambda \log\det(\Sigma_K)). This is the Bellman recursion in belief space. It is exact but intractable due to continuous matrix state (\Sigma) of dimension (MN\times MN).

# 5. A tractable surrogate: greedy one-step information-gain per cost

Practical approach: use one-step lookahead that maximizes expected immediate reduction in posterior uncertainty (information gain) normalized by cost.

Mutual information between (t) and future observation (y*k) under Gaussian linear model is
[
\mathcal{I}(t;y_k,|,\Sigma*{k-1},u*k) ;=; \tfrac{1}{2}\log\det\Big(I_M + R^{-1} H(u_k)\Sigma*{k-1} H(u*k)^\top\Big).
]
Equivalently the expected reduction in (\log\det(\Sigma)) is
[
\Delta*{\text{IG}}(u*k) = \tfrac{1}{2}\log\det\Big(I_M + R^{-1} H(u_k)\Sigma*{k-1} H(u*k)^\top\Big).
]
Greedy selection:
[
u_k^\* = \arg\max*{u\in\mathcal U} ; \frac{\Delta\_{\text{IG}}(u)}{\mathcal C(u)}.
]
Here (\mathcal U) is the feasible set of patterns (e.g., block indicators, binary patterns, Gray code patterns, or continuous illumination profiles), and (\mathcal C(u)) is the cost of presenting (u) (time per pattern). This is computationally feasible if we restrict (\mathcal U) to a candidate pool (coarse blocks, shifted local Gray codes, single-pixel indicators, etc.).

After choosing (u_k^\*) we apply the linear Gaussian update to get (\Sigma_k) and repeat.

# 6. Link to hierarchical / adaptive subdivision

Two simple facts follow from the information metric:

- If a coarse block pattern (u) projects to an area with no signal (so (H(u)\Sigma*{k-1}H(u)^\top) is near zero), then (\Delta*{\text{IG}}(u)\approx 0) and its normalized gain is small → prune.
- If a coarse block yields substantial (\Delta\_{\text{IG}}), subdividing that block into children patterns typically yields higher total information per unit cost when the posterior variance is concentrated inside that block.

Thus the greedy IG maximization naturally implements the proposed adaptive subdivision: coarse probes that return negligible IG are discarded; probes with high IG are refined. This is not an ad-hoc rule: it is the greedy optimization of expected uncertainty reduction.

# 7. Practical approximations for computation

Direct evaluation of (\Delta\_{\text{IG}}(u)) requires computing the (M\times M) determinant for each candidate (u). Simplifications:

1. **Low-rank structure:** (H(u)\Sigma H(u)^\top) often has low rank (rank ≤ rank((\Sigma))). Use matrix determinant lemma and low-rank updates to compute (\Delta\_{\text{IG}}) fast.

2. **Pixel-independence approximation:** If rows of (T) (camera pixels) are assumed independent a priori, (\Sigma) is block diagonal and (H\Sigma H^\top) reduces to sum over projector pixels in support of (u). This is conservative and cheap.

3. **Candidate pattern set:** restrict (\mathcal U) to multiscale block indicators (quadtree blocks) plus a small set of refined shifted Gray patterns for fine decoding. This limits the search while keeping adaptivity.

4. **Sparsity priors:** If (t) is sparse, use a Bernoulli-Gaussian prior (spike-and-slab) and approximate expected information gain by a Gaussian approximation around the MAP (Laplace). The update and IG formulas become approximate but capture sparsity and produce aggressive pruning.

# 8. Why Gray codes are not optimal (and where they are good)

Gray code scanning is an open-loop set of orthogonal (bit-plane) patterns. Strengths: deterministic, simple decode, guarantees full coverage in the worst case, cost (O(\log N)). Weaknesses relative to IG policy:

- Gray codes do not use feedback: they waste measurements on empty regions.
- They are designed for unique decoding under ideal noise, not for minimizing posterior uncertainty under sparsity and noise.
- Gray codes are near-optimal for uniform priors (worst-case dense scenes) where every projector pixel is equally likely active; they are suboptimal for sparse priors where adaptivity yields large gains.

Thus Gray codes are correct for deterministic decoding but suboptimal for minimizing expected posterior uncertainty when the prior is structured (sparse, localized) and noise exists.

# 9. The full proposed algorithm (pseudo-code)

```
Given: prior (mu0,Sigma0), candidate pattern set U (multiscale blocks + refinements), cost function C(u), noise cov R, budget or stopping rule.
For k = 1..K:
  For each candidate u in U:
    Compute DeltaIG(u) = 0.5 * log det(I + R^{-1} H(u) Sigma_{k-1} H(u)^T)
    Compute score(u) = DeltaIG(u) / C(u)
  Select u_k = argmax score(u)
  Project u_k, measure y_k
  Update mu_k, Sigma_k via linear Gaussian update
  Optionally: prune U by removing patterns that cover regions with very low posterior mass
  If stopping criterion met (budget exhausted or Sigma_k small enough) break
Return posterior mean reshaped as T_est = reshape(mu_k)
```

# 10. Theoretical justification and approximation results

_The exact DP is intractable._ The greedy IG policy is a standard surrogate in Bayesian experimental design. Under common assumptions (Gaussian model, diminishing returns of information), the mutual information objective is submodular in the set selection setting; when submodularity holds, greedy selection yields a ((1-1/e))-approximation to the optimal subset selection objective. In sequential settings the result is less clean but the greedy policy remains a widely used near-optimal heuristic. (If you require a formal proof for your report, I can include a focused derivation of submodularity for the Gaussian linear observation case under diagonal-covariance priors.)

# 11. Extensions to get more “optimal control” content

1. **Finite-horizon DP for small problems:** implement exact DP on a reduced state (e.g., beliefs over coarse blocks only). This gives a provable baseline and a controlled Bellman solution you can use to compare greedy performance and produce value-function visualizations.

2. **Pontryagin relaxation:** treat the continuous illumination (u(t)) as control and ( \Sigma(t)) as state with differential Riccati-like dynamics (continuous time limit). Apply Pontryagin Minimum Principle (PMP) on the continuous surrogate to derive structural properties of (u(t)) (e.g., bang-bang or hierarchical probes). This provides rigorous optimality conditions.

3. **Performance bounds:** derive upper and lower bounds on number of measurements (K) as function of sparsity (s) (number of active projector pixels). For example, show (K = O(s \log(N/s))) under compressed sensing style sensing designs (if you allow random projections rather than block indicators).

4. **Rigorous robustness analysis:** quantify how threshold errors (false positives/negatives) affect posterior covariance and how the IG policy trades off exploration vs exploitation under model mismatch.

# 12. Numerical experiment plan (to demonstrate superiority)

Simulate a sparse (T) (e.g., 1–5% active projector pixels mapped to some camera responses) and compare:

- Gray code open-loop (standard decoding),
- Adaptive hierarchical (heuristic subdivide),
- IG-greedy design (above).

Measure: total measurements K to reach (\log\det(\Sigma)\le \tau), reconstruction RMSE, and robustness to SNR. Expectation: IG-greedy ≪ Gray codes when sparsity is present; similar performance in the dense worst case.

# 13. Summary / recommended additions for the report

1. Replace informal “information gain” with the Gaussian mutual information formula and use (\log\det(\Sigma)) as the reconstruction cost.
2. Present the belief-space Bellman equation (exact DP) and explain intractability.
3. Derive the greedy one-step IG selection and give the closed-form (\Delta\_{\text{IG}}(u)) formula.
4. Show the posterior (Kalman) update explicitly.
5. Implement the IG greedy algorithm on a candidate pattern set and include simulation results comparing to Gray codes.
6. Optionally add a small exact DP on coarse blocks and/or a Pontryagin relaxation to add formal optimal-control content.

---

If you want, I can now (choose one)

1. derive the IG expression step-by-step from the Gaussian likelihood and show algebra for (H(u)), determinants and matrix-lemma simplifications;
2. produce the pseudo-code and numerical simulation (synthetic (T)) and plot measurement count vs RMSE for Gray vs IG-greedy; or
3. write the Bellman equation for a reduced quadtree belief state and solve it exactly for small instances so you can include a provable DP result in the report.

Pick one and I’ll do it now.
