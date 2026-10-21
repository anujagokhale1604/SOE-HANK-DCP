"""
    SOE_HANK_DCP.jl

Main module: Small Open Economy HANK with Dominant Currency Pricing.

Architecture:
  - Households: HANK with idiosyncratic income risk, solved via EGM
  - Steady-state distribution: Kolmogorov Forward (lottery method)
  - Aggregate dynamics: Sequence-Space Jacobian (Auclert et al. 2021)
  - Open-economy block: DCP import pricing, UIP exchange rate, NKPC
  - Monetary shock: 100bp Taylor-rule-based interest rate hike

Author: Anuja A. Gokhale
"""

module SOE_HANK_DCP

using LinearAlgebra
using SparseArrays
using Parameters
using Interpolations

export Params, AssetGrid, IncomeProcess
export solve_household
export stationary_distribution, aggregates, check_market_clearing
export SteadyState, jacobian_C_r, jacobian_C_Y
export GESystem, solve_ge
export run_model

include("parameters.jl")
include("household_block.jl")
include("steady_state.jl")
include("fake_news.jl")
include("open_economy_dag.jl")

# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

"""
    run_model(; p, verbose)

Full model pipeline:
  1. Build grids
  2. Solve household EGM
  3. Compute stationary distribution g*
  4. Build SteadyState and compute Jacobians (Fake-News)
  5. Assemble GE DAG
  6. Run 100bp monetary policy shock and return IRFs

Returns a NamedTuple with all IRFs and steady-state objects.
"""
function run_model(; p::Params = Params(), verbose::Bool = true)

    verbose && println("=== SOE-HANK-DCP Model ===")

    # --- Grids ---
    ag = AssetGrid(p.a_min, p.a_max, p.n_a)
    ip = IncomeProcess(p.n_y, p.ρ_y, p.σ_y)
    verbose && println("✓ Grids: a ∈ [$(round(ag.a[1],digits=2)), $(round(ag.a[end],digits=2))], n_a=$(p.n_a), n_y=$(p.n_y)")

    # --- Household EGM ---
    verbose && print("  Solving EGM... ")
    c_pol, a_pol = solve_household(p, ag, ip; r=p.r_ss, w=p.w_ss)
    verbose && println("✓")

    # --- Stationary distribution ---
    verbose && print("  Computing stationary distribution... ")
    Q  = build_lottery_matrix(a_pol, ag, ip)
    g  = stationary_distribution(a_pol, ag, ip)
    agg, nfa = check_market_clearing(g, c_pol, a_pol, ag, ip, p.w_ss, p.r_ss)
    verbose && println("✓  C=$(round(agg.C,digits=4))  NFA=$(round(nfa,digits=4))")

    # --- Steady State object ---
    ss = SteadyState(c_pol, a_pol, g, Q, ag, ip, p, p.r_ss, p.w_ss)

    # --- Fake-News Jacobians ---
    verbose && print("  Computing Fake-News Jacobians (T=$(p.T))... ")
    J_Cr = jacobian_C_r(ss, p.T)
    J_CY = jacobian_C_Y(ss, p.T)
    verbose && println("✓")

    # --- GE DAG ---
    ge = GESystem(J_Cr, J_CY, p, p.T)

    # --- 100bp monetary shock ---
    verbose && println("  Running 100bp MP shock (σ_MP=$(p.σ_MP))...")
    dε_MP = zeros(p.T)
    dε_MP[1] = p.σ_MP   # surprise 100bp hike at t=1

    irfs = solve_ge(ge, dε_MP)
    verbose && println("✓ Done.\n")

    return (;
        ss, ag, ip, p,
        J_Cr, J_CY, ge,
        irfs,
        agg
    )
end

end # module
