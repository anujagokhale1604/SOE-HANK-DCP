"""
    steady_state.jl

Computes the steady-state wealth distribution g*(a, y) and checks
market clearing.

The stationary distribution satisfies:
    g*(a', y') = Σ_{a,y} g*(a, y) · Π(y, y') · 1[a_pol(a,y) = a']

We represent g* as a (n_a × n_y) matrix and iterate the
Kolmogorov Forward Equation (Young 2010 lottery method) to convergence.
"""

using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Transition operator (lottery / histogram method)
# ---------------------------------------------------------------------------

"""
    build_lottery_matrix(a_pol, ag, ip)

Build the (n_a × n_y) → (n_a × n_y) transition matrix in sparse form.
Savings a_pol[i,j] ∈ [ag.a[1], ag.a[end]] is split between the two
nearest grid points (linear lottery).

Returns a sparse matrix Q of size (n_a·n_y × n_a·n_y) such that
vec(g_new) = Q' * vec(g_old).
"""
function build_lottery_matrix(
    a_pol::Matrix{Float64},
    ag::AssetGrid,
    ip::IncomeProcess
)
    n_a, n_y = ag.n, ip.n
    N = n_a * n_y

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:n_y          # current income
        for i in 1:n_a      # current assets
            # Saved asset level
            a_save = a_pol[i, j]

            # Find bracket in asset grid
            k_lo = searchsortedlast(ag.a, a_save)
            k_lo = clamp(k_lo, 1, n_a - 1)
            k_hi = k_lo + 1

            # Linear weights
            w_hi = (a_save - ag.a[k_lo]) / (ag.a[k_hi] - ag.a[k_lo])
            w_hi = clamp(w_hi, 0.0, 1.0)
            w_lo = 1.0 - w_hi

            from_idx = (j - 1) * n_a + i   # current state index

            for jj in 1:n_y   # next income state
                prob = ip.Π[j, jj]
                if prob < 1e-15
                    continue
                end
                to_lo = (jj - 1) * n_a + k_lo
                to_hi = (jj - 1) * n_a + k_hi

                push!(rows, to_lo); push!(cols, from_idx); push!(vals, w_lo * prob)
                push!(rows, to_hi); push!(cols, from_idx); push!(vals, w_hi * prob)
            end
        end
    end

    return sparse(rows, cols, vals, N, N)
end

# ---------------------------------------------------------------------------
# Iterate distribution forward
# ---------------------------------------------------------------------------

"""
    stationary_distribution(a_pol, ag, ip; tol, max_iter)

Returns the stationary distribution g* as (n_a × n_y) matrix
(columns sum to ip.π_stat; whole matrix sums to 1).
"""
function stationary_distribution(
    a_pol::Matrix{Float64},
    ag::AssetGrid,
    ip::IncomeProcess;
    tol::Float64 = 1e-11,
    max_iter::Int = 10_000
)
    n_a, n_y = ag.n, ip.n
    N = n_a * n_y

    Q = build_lottery_matrix(a_pol, ag, ip)

    # Initialise: uniform over asset grid, weighted by stationary income dist
    g = zeros(N)
    for j in 1:n_y
        g[(j-1)*n_a+1:j*n_a] .= ip.π_stat[j] / n_a
    end

    for _ in 1:max_iter
        g_new = Q * g
        err = maximum(abs.(g_new .- g))
        g = g_new
        if err < tol
            break
        end
    end

    g ./= sum(g)  # normalise to 1
    return reshape(g, n_a, n_y)
end

# ---------------------------------------------------------------------------
# Market clearing and aggregates
# ---------------------------------------------------------------------------

"""
    aggregates(g, c_pol, a_pol, ag, ip, w)

Compute aggregate consumption C, asset supply A, and labour income Y.
"""
function aggregates(
    g::Matrix{Float64},
    c_pol::Matrix{Float64},
    a_pol::Matrix{Float64},
    ag::AssetGrid,
    ip::IncomeProcess,
    w::Float64
)
    C = sum(g .* c_pol)
    A = sum(g .* a_pol)
    L = sum(g .* (ip.y' .* ones(ag.n)))  # aggregate labour
    Y = w * L
    return (; C, A, L, Y)
end

"""
    check_market_clearing(g, c_pol, a_pol, ag, ip, w, r; tol)

Asserts Walras' law: A ≈ 0 (closed-economy bond market) or
returns the net asset position for an open economy.
"""
function check_market_clearing(
    g::Matrix{Float64},
    c_pol::Matrix{Float64},
    a_pol::Matrix{Float64},
    ag::AssetGrid,
    ip::IncomeProcess,
    w::Float64,
    r::Float64;
    tol::Float64 = 1e-4
)
    agg = aggregates(g, c_pol, a_pol, ag, ip, w)
    excess_bonds = agg.A   # in SOE: net foreign assets; in closed: should be ≈ 0
    return agg, excess_bonds
end
