"""
    produce_irfs.jl

Runs the 100bp monetary policy shock experiment and produces:
  - IRF plots (output, consumption, inflation, exchange rate)
  - Stationary distribution plot (wealth distribution g*(a, y))
  - Jacobian heatmap (J_{C,r})
  - DCP vs PCP comparison

Usage:
    julia --project scripts/produce_irfs.jl

Outputs saved to plots/
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra

include("../src/parameters.jl")
include("../src/household_block.jl")
include("../src/steady_state.jl")
include("../src/fake_news.jl")
include("../src/open_economy_dag.jl")

# ---------------------------------------------------------------------------
# Run model: DCP (φ=1) and PCP (φ=0)
# ---------------------------------------------------------------------------

println("Running DCP model (φ=1.0)...")
p_dcp = Params(φ=1.0, T=40, n_a=150, n_y=7)
ag_dcp = AssetGrid(p_dcp.a_min, p_dcp.a_max, p_dcp.n_a)
ip_dcp = IncomeProcess(p_dcp.n_y, p_dcp.ρ_y, p_dcp.σ_y)

c_dcp, a_dcp = solve_household(p_dcp, ag_dcp, ip_dcp)
Q_dcp = build_lottery_matrix(a_dcp, ag_dcp, ip_dcp)
g_dcp = stationary_distribution(a_dcp, ag_dcp, ip_dcp)
ss_dcp = SteadyState(c_dcp, a_dcp, g_dcp, Q_dcp, ag_dcp, ip_dcp, p_dcp, p_dcp.r_ss, p_dcp.w_ss)

J_Cr_dcp = jacobian_C_r(ss_dcp, p_dcp.T)
J_CY_dcp = jacobian_C_Y(ss_dcp, p_dcp.T)
ge_dcp   = GESystem(J_Cr_dcp, J_CY_dcp, p_dcp, p_dcp.T)

dε_MP = zeros(p_dcp.T); dε_MP[1] = p_dcp.σ_MP
irfs_dcp = solve_ge(ge_dcp, dε_MP)

println("Running PCP model (φ=0.0)...")
p_pcp = Params(φ=0.0, T=40, n_a=150, n_y=7)
ag_pcp = AssetGrid(p_pcp.a_min, p_pcp.a_max, p_pcp.n_a)
ip_pcp = IncomeProcess(p_pcp.n_y, p_pcp.ρ_y, p_pcp.σ_y)

c_pcp, a_pcp = solve_household(p_pcp, ag_pcp, ip_pcp)
Q_pcp = build_lottery_matrix(a_pcp, ag_pcp, ip_pcp)
g_pcp = stationary_distribution(a_pcp, ag_pcp, ip_pcp)
ss_pcp = SteadyState(c_pcp, a_pcp, g_pcp, Q_pcp, ag_pcp, ip_pcp, p_pcp, p_pcp.r_ss, p_pcp.w_ss)

J_Cr_pcp = jacobian_C_r(ss_pcp, p_pcp.T)
J_CY_pcp = jacobian_C_Y(ss_pcp, p_pcp.T)
ge_pcp   = GESystem(J_Cr_pcp, J_CY_pcp, p_pcp, p_pcp.T)

irfs_pcp = solve_ge(ge_pcp, dε_MP)

# ---------------------------------------------------------------------------
# Plot 1: IRFs — DCP vs PCP
# ---------------------------------------------------------------------------

T_plot = p_dcp.T
quarters = 1:T_plot
scale = 100  # convert to percent / basis points

gr(size=(900, 700), dpi=150)

p1 = plot(quarters, scale .* irfs_dcp.dr,
    label="DCP (φ=1)", lw=2, color=:steelblue, ls=:solid,
    ylabel="pp", title="Interest Rate (r)")
plot!(p1, quarters, scale .* irfs_pcp.dr,
    label="PCP (φ=0)", lw=2, color=:crimson, ls=:dash)

p2 = plot(quarters, scale .* irfs_dcp.dC,
    label="DCP", lw=2, color=:steelblue,
    ylabel="%", title="Consumption (C)")
plot!(p2, quarters, scale .* irfs_pcp.dC,
    label="PCP", lw=2, color=:crimson, ls=:dash)

p3 = plot(quarters, scale .* irfs_dcp.dπ,
    label="DCP", lw=2, color=:steelblue,
    ylabel="pp", title="CPI Inflation (π)")
plot!(p3, quarters, scale .* irfs_pcp.dπ,
    label="PCP", lw=2, color=:crimson, ls=:dash)

p4 = plot(quarters, scale .* irfs_dcp.de,
    label="DCP", lw=2, color=:steelblue,
    ylabel="%", title="Exchange Rate (ê, appreciation +)")
plot!(p4, quarters, scale .* irfs_pcp.de,
    label="PCP", lw=2, color=:crimson, ls=:dash)

for pp in [p1, p2, p3, p4]
    hline!(pp, [0], lw=0.8, color=:gray, ls=:dot, label=false)
    xlabel!(pp, "Quarters")
end

fig_irfs = plot(p1, p2, p3, p4,
    layout=(2,2),
    plot_title="SOE-HANK-DCP: 100bp Monetary Shock — DCP vs PCP",
    plot_titlefontsize=11,
    legend=:topright,
    margin=3Plots.mm)

mkpath(joinpath(@__DIR__, "../plots"))
savefig(fig_irfs, joinpath(@__DIR__, "../plots/irf_monetary_shock.png"))
println("✓ Saved plots/irf_monetary_shock.png")

# ---------------------------------------------------------------------------
# Plot 2: Stationary wealth distribution g*(a, y)
# ---------------------------------------------------------------------------

# Marginal distribution over assets
g_marg = sum(g_dcp, dims=2)[:, 1]
a_grid = ag_dcp.a

fig_dist = plot(a_grid, g_marg,
    lw=2, color=:steelblue, fill=(0, 0.15, :steelblue),
    xlabel="Assets (a)", ylabel="Density",
    title="Stationary Wealth Distribution g*(a)",
    legend=false, dpi=150)
vline!(fig_dist, [0.0], lw=1, color=:gray, ls=:dot)

# By income quartile (bottom 2 vs top 2 income states)
for (j, col, lbl) in [(1, :crimson, "Low income"), (ip_dcp.n, :teal, "High income")]
    plot!(fig_dist, a_grid, g_dcp[:, j],
        lw=1.5, color=col, ls=:dash, label=lbl, alpha=0.7)
end
plot!(fig_dist, legend=:topright)

savefig(fig_dist, joinpath(@__DIR__, "../plots/stationary_distribution.png"))
println("✓ Saved plots/stationary_distribution.png")

# ---------------------------------------------------------------------------
# Plot 3: Consumption Jacobian heatmap J_{C,r}
# ---------------------------------------------------------------------------

T_hm = min(30, p_dcp.T)
J_plot = J_Cr_dcp[1:T_hm, 1:T_hm]

fig_jac = heatmap(1:T_hm, 1:T_hm, J_plot,
    color=:RdBu,
    xlabel="Shock period t (∂r_t)",
    ylabel="Response period s (∂C_s)",
    title="Consumption Jacobian  J_{C,r} = ∂C_s / ∂r_t",
    dpi=150, clims=(-maximum(abs.(J_plot)), maximum(abs.(J_plot))))

savefig(fig_jac, joinpath(@__DIR__, "../plots/jacobian_heatmap.png"))
println("✓ Saved plots/jacobian_heatmap.png")

println("\n=== Done. All plots in plots/ ===")
