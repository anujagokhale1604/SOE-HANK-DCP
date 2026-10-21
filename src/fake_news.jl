"""
    fake_news.jl

Fake-News Algorithm (Auclert, Bardóczy, Rognlie & Straub 2021, ECMA).
"""

using LinearAlgebra
using SparseArrays

struct SteadyState
    c::Matrix{Float64}
    a::Matrix{Float64}
    g::Matrix{Float64}
    Q::SparseMatrixCSC
    ag::AssetGrid
    ip::IncomeProcess
    p::Params
    r::Float64
    w::Float64
end

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

# curly-E: MPC propagated forward by Q
function curly_E(ss::SteadyState, T::Int)
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y
    δ = 1e-4

    mpc = zeros(N)
    for j in 1:n_y
        wy = ss.w * ss.ip.y[j]
        m_grid = (1 + ss.r) .* ss.ag.a .+ wy
        itp = linear_interp(m_grid, ss.c[:, j])
        for i in 1:n_a
            m0 = m_grid[i]
            mpc[(j-1)*n_a + i] = (itp(m0 + δ) - ss.c[i,j]) / δ
        end
    end

    E = zeros(T, N)
    E[1, :] = mpc
    v = copy(mpc)
    for s in 2:T
        v = ss.Q * v   # forward: Q (not Q')
        E[s, :] = v
    end
    return E
end

# curly-D: distributional response propagated forward
function curly_D(ss::SteadyState, T::Int)
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y
    δ = 1e-4

    da_dr = zeros(n_a, n_y)
    for j in 1:n_y
        wy = ss.w * ss.ip.y[j]
        m_grid = (1 + ss.r) .* ss.ag.a .+ wy
        a_itp = linear_interp(m_grid, ss.a[:, j])
        for i in 1:n_a
            m0 = m_grid[i]
            # dr raises cash-on-hand by a[i]*dr; perturbation = a[i]*δ
            dm = ss.ag.a[i] * δ
            if abs(dm) < 1e-12
                da_dr[i, j] = 0.0
            else
                da_dr[i, j] = (a_itp(m0 + dm) - ss.a[i, j]) / dm
            end
        end
    end

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

function _perturb_distribution(ss::SteadyState, da_dr::Matrix{Float64})
    n_a, n_y = ss.ag.n, ss.ip.n
    N = n_a * n_y
    dg = zeros(N)

    for j in 1:n_y
        for i in 1:n_a
            g_ij = ss.g[i, j]
            g_ij == 0.0 && continue
            da = da_dr[i, j]
            a_base = ss.a[i, j]
            a_new  = a_base + da

            k0 = searchsortedlast(ss.ag.a, a_base)
            k0 = clamp(k0, 1, n_a-1)
            w0_hi = clamp((a_base - ss.ag.a[k0]) / (ss.ag.a[k0+1] - ss.ag.a[k0]), 0.0, 1.0)

            k1 = searchsortedlast(ss.ag.a, a_new)
            k1 = clamp(k1, 1, n_a-1)
            w1_hi = clamp((a_new - ss.ag.a[k1]) / (ss.ag.a[k1+1] - ss.ag.a[k1]), 0.0, 1.0)

            for jj in 1:n_y
                prob = ss.ip.Π[j, jj]
                prob < 1e-15 && continue
                pg = prob * g_ij

                dg[(jj-1)*n_a + k0]   -= pg * (1 - w0_hi)
                dg[(jj-1)*n_a + k0+1] -= pg * w0_hi
                dg[(jj-1)*n_a + k1]   += pg * (1 - w1_hi)
                dg[(jj-1)*n_a + k1+1] += pg * w1_hi
            end
        end
    end
    return dg
end

# Jacobian assembly — correct fake-news formula: J[s,t] = E[s-t+1,:] · D[:,1]
# but D[:,s-t+1] for the full formula. Using the Toeplitz structure:
function jacobian_C_r(ss::SteadyState, T::Int)
    E = curly_E(ss, T)
    D = curly_D(ss, T)
    # J[s,t] = dot(E[s-t+1, :], D[:, 1])  — fake-news first-order
    # equivalently J = lower-triangular Toeplitz built from F[lag] = E[lag+1,:] · D[:,1]
    F = [dot(E[lag, :], D[:, 1]) for lag in 1:T]
    J = zeros(T, T)
    for t in 1:T, s in t:T
        J[s, t] = F[s - t + 1]
    end
    return J
end

function jacobian_C_Y(ss::SteadyState, T::Int)
    E = curly_E(ss, T)
    g_flat = vec(ss.g)
    F = [dot(E[lag, :], g_flat) for lag in 1:T]
    J = zeros(T, T)
    for t in 1:T, s in t:T
        J[s, t] = F[s - t + 1]
    end
    return J
end
