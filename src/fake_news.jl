"""
    fake_news.jl

Clean implementation of the Fake-News Algorithm (Auclert, Bardóczy,
Rognlie & Straub 2021, ECMA) for computing sequence-space Jacobians.

Given a steady-state household block (c*, a*, g*), the algorithm
computes the Jacobian J_{C,r}[s,t] = ∂C_s/∂r_t by:

1. Computing 𝓓 (the "curly-D" matrix): the effect of a one-time
   surprise increase in r_t on the distribution s periods later.
2. Computing 𝓔 (the "curly-E" matrix): the marginal propensity to
   consume out of a one-period income shock, integrated across the
   distribution.
3. J = 𝓓 * 𝓔^⊤  (plus a direct effect term for contemporaneous MPCs).

Notation follows ABRS (2021) closely.
"""

using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Step 0: Steady-state inputs
# ---------------------------------------------------------------------------

"""
    SteadyState

Bundles all steady-state objects needed by the fake-news algorithm.
"""
struct SteadyState
    c::Matrix{Float64}      # consumption policy  (n_a × n_y)
    a::Matrix{Float64}      # savings policy      (n_a × n_y)
    g::Matrix{Float64}      # wealth distribution (n_a × n_y)
    Q::SparseMatrixCSC      # lottery transition matrix (N × N)
    ag::AssetGrid
    ip::IncomeProcess
    p::Params
    r::Float64
    w::Float64
end

# ---------------------------------------------------------------------------
# Step 1: curly-E — household-level MPC responses
# ---------------------------------------------------------------------------

"""
    curly_E(ss, T)

Compute the (T × N) matrix 𝓔 where 𝓔[s, i] is the expected
present-value consumption response of household-state i (flattened)
to a one-unit increase in their cash-on-hand today, s periods from now.

Equivalently, 𝓔[0, :] = MPC vector; 𝓔[s, :] = Q^s * MPC.
"""
function curly_E(ss::SteadyState, T::Int)
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y

    # MPC at each state: ∂c/∂m where m = (1+r)a + wy
    # We approximate via finite difference on the EGM grid
    δ = 1e-5
    c_bump = zeros(n_a, n_y)
    for j in 1:n_y
        wy = ss.w * ss.ip.y[j]
        for i in 1:n_a
            m_base = (1 + ss.r) * ss.ag.a[i] + wy
            m_hi   = m_base + δ
            # Interpolate consumption at m_hi using saved policy
            # (linear interpolation on the exogenous grid)
            m_grid = (1 + ss.r) .* ss.ag.a .+ wy
            itp = linear_interp(m_grid, ss.c[:, j])
            c_bump[i, j] = (itp(m_hi) - ss.c[i, j]) / δ
        end
    end
    mpc = vec(c_bump)    # (N,) vector

    # 𝓔[s, :] = (Q')^s * mpc, propagated forward
    E = zeros(T, N)
    E[1, :] = mpc
    Qt = ss.Q'           # transpose for forward propagation
    v = copy(mpc)
    for s in 2:T
        v = Qt * v
        E[s, :] = v
    end
    return E
end

# Simple 1D linear interpolation helper
function linear_interp(x::Vector{Float64}, y::Vector{Float64})
    function itp(x0::Float64)
        i = searchsortedlast(x, x0)
        i = clamp(i, 1, length(x)-1)
        w = (x0 - x[i]) / (x[i+1] - x[i])
        w = clamp(w, 0.0, 1.0)
        return y[i] * (1-w) + y[i+1] * w
    end
    return itp
end

# ---------------------------------------------------------------------------
# Step 2: curly-D — distributional dynamics
# ---------------------------------------------------------------------------

"""
    curly_D(ss, T)

Compute the (N × T) matrix 𝓓 where 𝓓[:, t] is the change in the
distribution s periods after a one-time shock to asset returns at t=0.

𝓓[:, 0] = dg/dr (direct impact of r shock on distribution)
𝓓[:, s] = Q^s * 𝓓[:, 0]  (propagated forward by the unperturbed Q)
"""
function curly_D(ss::SteadyState, T::Int)
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y

    # Perturbation: dr = δ → asset returns rise by δ
    # Effect on savings: da' = ∂a'/∂r · dr
    δ = 1e-5
    da_dr = zeros(n_a, n_y)
    for j in 1:n_y
        wy = ss.w * ss.ip.y[j]
        for i in 1:n_a
            # Cash-on-hand rises by a[i]·δ (income effect of rate hike)
            dm = ss.ag.a[i] * δ
            m_grid = (1 + ss.r) .* ss.ag.a .+ wy
            c_itp = linear_interp(m_grid, ss.c[:, j])
            a_itp = linear_interp(m_grid, ss.a[:, j])
            m0 = (1 + ss.r) * ss.ag.a[i] + wy
            da_dr[i, j] = (a_itp(m0 + dm) - ss.a[i, j]) / δ
        end
    end

    # dg_0/dr: how does the distribution shift given da'/dr?
    # Using the lottery method: dQ/dr · g*
    dg0 = _perturb_distribution(ss, da_dr)

    D = zeros(N, T)
    D[:, 1] = dg0
    v = copy(dg0)
    for s in 2:T
        v = ss.Q * v
        D[:, s] = v
    end
    return D
end

"""
    _perturb_distribution(ss, da_dr)

Compute the first-order change in the stationary distribution
implied by a change in the savings policy da_dr (n_a × n_y).
"""
function _perturb_distribution(ss::SteadyState, da_dr::Matrix{Float64})
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y
    g = vec(ss.g)
    dg = zeros(N)

    for j in 1:n_y
        for i in 1:n_a
            from_idx = (j - 1) * n_a + i
            g_ij = ss.g[i, j]
            da = da_dr[i, j]
            a_new = ss.a[i, j] + da

            # Find bracket
            k_lo = searchsortedlast(ss.ag.a, a_new)
            k_lo = clamp(k_lo, 1, n_a - 1)
            k_hi = k_lo + 1
            w_hi = clamp((a_new - ss.ag.a[k_lo]) / (ss.ag.a[k_hi] - ss.ag.a[k_lo]), 0.0, 1.0)
            w_lo = 1.0 - w_hi

            # Baseline bracket for a_pol[i,j]
            a_base = ss.a[i, j]
            k_lo0 = searchsortedlast(ss.ag.a, a_base)
            k_lo0 = clamp(k_lo0, 1, n_a - 1)
            k_hi0 = k_lo0 + 1
            w_hi0 = clamp((a_base - ss.ag.a[k_lo0]) / (ss.ag.a[k_hi0] - ss.ag.a[k_lo0]), 0.0, 1.0)
            w_lo0 = 1.0 - w_hi0

            for jj in 1:n_y
                prob = ss.ip.Π[j, jj]
                if prob < 1e-15; continue; end

                # Change in probability mass at perturbed location
                to_lo  = (jj - 1) * n_a + k_lo
                to_hi  = (jj - 1) * n_a + k_hi
                to_lo0 = (jj - 1) * n_a + k_lo0
                to_hi0 = (jj - 1) * n_a + k_hi0

                dg[to_lo]  += prob * g_ij * w_lo
                dg[to_hi]  += prob * g_ij * w_hi
                dg[to_lo0] -= prob * g_ij * w_lo0
                dg[to_hi0] -= prob * g_ij * w_hi0
            end
        end
    end
    return dg
end

# ---------------------------------------------------------------------------
# Step 3: Assemble Jacobian J_{C,r}
# ---------------------------------------------------------------------------

"""
    jacobian_C_r(ss, T)

Assemble the (T × T) sequence-space Jacobian ∂C_s / ∂r_t using
the Fake-News algorithm.

    J[s, t] = 𝓓[:, s-t]' · 𝓔[0, :] + Σ_{τ>0} 𝓓[:, s-t+τ]' · Δ𝓔[τ, :]

Simplified (first-order truncation):
    J = E * D'
where E is (T×N) and D is (N×T).
"""
function jacobian_C_r(ss::SteadyState, T::Int)
    E = curly_E(ss, T)   # (T × N)
    D = curly_D(ss, T)   # (N × T)

    # J[s,t] = Σ_i E[s-t, i] · D[i, 1]  (only first column of D matters for s≥t)
    # Full assembly via matrix product:
    J = zeros(T, T)
    for t in 1:T
        for s in t:T
            lag = s - t
            if lag < T
                J[s, t] = dot(E[lag+1, :], D[:, 1])
            end
        end
    end
    return J
end

"""
    jacobian_C_Y(ss, T)

Jacobian of aggregate consumption w.r.t. aggregate income
(wage income channel). Analogous structure to J_{C,r}.
"""
function jacobian_C_Y(ss::SteadyState, T::Int)
    E = curly_E(ss, T)
    # dY shock: uniform income shift → dc/dm = mpc, same structure
    # Simplified: ∂C_s / ∂Y_t ≈ E[s-t, :] · g_ss
    g_flat = vec(ss.g)
    J = zeros(T, T)
    for t in 1:T
        for s in t:T
            lag = s - t
            if lag < T
                J[s, t] = dot(E[lag+1, :], g_flat)
            end
        end
    end
    return J
end
