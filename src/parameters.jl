"""
    parameters.jl

Model primitives for the SOE-HANK-DCP model.
All structural parameters are collected here using Parameters.jl @with_kw.
"""

using Parameters

# ---------------------------------------------------------------------------
# Asset grid
# ---------------------------------------------------------------------------

"""
    AssetGrid

Log-spaced asset grid from ā (borrowing limit) to a_max.
"""
struct AssetGrid
    a::Vector{Float64}   # grid points
    n::Int               # number of points
end

function AssetGrid(a_min::Float64, a_max::Float64, n::Int)
    # log-spaced: denser near borrowing constraint
    a = exp.(range(log(a_min - a_min + 1e-6), log(a_max - a_min + 1e-6), length=n)) .+ a_min .- 1e-6
    a[1] = a_min  # enforce exact lower bound
    AssetGrid(a, n)
end

# ---------------------------------------------------------------------------
# Income process
# ---------------------------------------------------------------------------

"""
    IncomeProcess

Discretised AR(1) log-income: log y_t = ρ log y_{t-1} + σ ε_t,  ε ~ N(0,1).
Uses Rouwenhorst (1995) method for persistence ρ close to 1.
"""
struct IncomeProcess
    y::Vector{Float64}       # income grid (levels)
    Π::Matrix{Float64}       # transition matrix (n_y × n_y), rows sum to 1
    π_stat::Vector{Float64}  # stationary distribution
    n::Int
end

function rouwenhorst(n::Int, ρ::Float64, σ::Float64)
    # Rouwenhorst (1995) discretisation
    p = (1 + ρ) / 2
    q = p
    # state space
    σ_y = σ / sqrt(1 - ρ^2)
    y_max = sqrt(n - 1) * σ_y
    y_grid = range(-y_max, y_max, length=n) |> collect

    # base 2×2 matrix
    Θ = [p 1-p; 1-q q]
    if n == 2
        Π = Θ
    else
        for k in 3:n
            z = zeros(k, k)
            Π_prev = Θ
            Θ = p * [Π_prev zeros(k-1,1); zeros(1,k)] +
                (1-p) * [zeros(k-1,1) Π_prev; zeros(1,k)] +
                (1-q) * [zeros(1,k); Π_prev zeros(k-1,1)] +
                q * [zeros(1,k); zeros(k-1,1) Π_prev]
            # normalise interior rows
            Θ[2:end-1, :] ./= 2
        end
        Π = Θ
    end
    # row-normalise
    Π = Π ./ sum(Π, dims=2)
    # stationary distribution: left eigenvector
    π_stat = stationary_dist(Π)
    # convert grid to levels (mean-normalised)
    y_lev = exp.(y_grid)
    y_lev ./= dot(π_stat, y_lev)
    return y_lev, Π, π_stat
end

function stationary_dist(Π::Matrix{Float64})
    # Power iteration for stationary distribution
    n = size(Π, 1)
    π = fill(1/n, n)
    for _ in 1:10_000
        π_new = Π' * π
        if maximum(abs.(π_new .- π)) < 1e-12
            return π_new
        end
        π = π_new
    end
    return π
end

function IncomeProcess(n::Int, ρ::Float64, σ::Float64)
    y, Π, π_stat = rouwenhorst(n, ρ, σ)
    IncomeProcess(y, Π, π_stat, n)
end

# ---------------------------------------------------------------------------
# Main parameter struct
# ---------------------------------------------------------------------------

@with_kw struct Params
    # Household preferences
    β::Float64  = 0.98       # discount factor (quarterly)
    γ::Float64  = 2.0        # CRRA coefficient
    χ::Float64  = 1.0        # Frisch elasticity denominator weight
    ν::Float64  = 1.0        # inverse Frisch elasticity

    # Asset grid
    a_min::Float64 = -2.0    # borrowing limit (< 0 → indebted households)
    a_max::Float64 = 40.0    # upper bound
    n_a::Int       = 200     # number of asset grid points

    # Income process (AR(1) in logs)
    ρ_y::Float64 = 0.966     # quarterly persistence (Flodén & Lindé 2001)
    σ_y::Float64 = 0.170     # conditional std of log income innovations
    n_y::Int     = 7         # income grid points

    # Steady-state interest / wage
    r_ss::Float64 = 0.005    # quarterly net real interest rate (~2% annual)
    w_ss::Float64 = 1.0      # normalised real wage

    # Open-economy / DCP block
    α_M::Float64  = 0.30     # import share of consumption
    η::Float64    = 1.5      # elasticity of substitution (home vs. imports)
    φ::Float64    = 0.25     # degree of dominant-currency pricing (0 = PCP, 1 = DCP)
    ρ_e::Float64  = 0.85     # persistence of exchange rate (UIP + risk premium)

    # Mortgage / cash-flow block
    m_share::Float64 = 0.35  # share of HtM/indebted households at borrowing constraint
    θ_m::Float64     = 0.25  # mortgage debt-to-income ratio of constrained households

    # New Keynesian / Taylor rule
    ϕ_π::Float64  = 1.5      # Taylor coefficient on inflation
    ϕ_Y::Float64  = 0.5      # Taylor coefficient on output gap
    κ::Float64    = 0.1      # NKPC slope
    ρ_i::Float64  = 0.7      # interest rate smoothing

    # Monetary shock
    σ_MP::Float64 = 0.01     # 100bp shock (set to 0.01 in quarterly rate)

    # Sequence-space horizon
    T::Int = 200             # IRF horizon (quarters)
end

"""
    dot(x, y)  — simple inner product (avoids importing LinearAlgebra everywhere)
"""
dot(x, y) = sum(x .* y)
