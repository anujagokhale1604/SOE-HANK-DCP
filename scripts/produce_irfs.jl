using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using LinearAlgebra

if !isdefined(Main, :SOE_HANK_DCP)
    include("../src/SOE_HANK_DCP.jl")
end
using .SOE_HANK_DCP: Params, run_model

println("Running DCP model (φ=1.0)...")
res_dcp = run_model(p = Params(φ=1.0, ϕ_Y=0.125, T=60, n_a=150, n_y=5))

println("Running PCP model (φ=0.0)...")
res_pcp = run_model(p = Params(φ=0.0, ϕ_Y=0.125, T=60, n_a=150, n_y=5))

irfs_dcp = res_dcp.irfs
irfs_pcp = res_pcp.irfs

T_plot = 40
quarters = 1:T_plot
scale = 100

gr(size=(900, 700), dpi=150)

p1 = plot(quarters, scale .* irfs_dcp.dr[1:T_plot],
    label="DCP (φ=1)", lw=2, color=:steelblue, ls=:solid,
    ylabel="pp", title="Real Interest Rate (r)")
plot!(p1, quarters, scale .* irfs_pcp.dr[1:T_plot],
    label="PCP (φ=0)", lw=2, color=:crimson, ls=:dash)

p2 = plot(quarters, scale .* irfs_dcp.dY[1:T_plot],
    label="DCP (φ=1)", lw=2, color=:steelblue,
    ylabel="%", title="Output (Y)")
plot!(p2, quarters, scale .* irfs_pcp.dY[1:T_plot],
    label="PCP (φ=0)", lw=2, color=:crimson, ls=:dash)

p3 = plot(quarters, scale .* irfs_dcp.dπ[1:T_plot],
    label="DCP (φ=1)", lw=2, color=:steelblue,
    ylabel="pp", title="CPI Inflation (π)")
plot!(p3, quarters, scale .* irfs_pcp.dπ[1:T_plot],
    label="PCP (φ=0)", lw=2, color=:crimson, ls=:dash)

p4 = plot(quarters, scale .* irfs_dcp.de[1:T_plot],
    label="DCP (φ=1)", lw=2, color=:steelblue,
    ylabel="%", title="Exchange Rate (ê, appreciation +)")
plot!(p4, quarters, scale .* irfs_pcp.de[1:T_plot],
    label="PCP (φ=0)", lw=2, color=:crimson, ls=:dash)

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

D_ss = res_dcp.hh.D_ss
g_marg = sum(D_ss, dims=2)[:, 1]
a_grid = res_dcp.ag.a

fig_dist = plot(a_grid, g_marg,
    lw=2, color=:steelblue, fill=(0, 0.15, :steelblue),
    xlabel="Assets (a)", ylabel="Probability Mass",
    title="Stationary Wealth Distribution D*(a) (FastEGM)",
    label="Marginal D*(a)", dpi=150)
vline!(fig_dist, [0.0], lw=1, color=:gray, ls=:dot, label="Zero Net Wealth")

for (j, col, lbl) in [(1, :crimson, "Lowest income state"), (res_dcp.ip.n, :teal, "Highest income state")]
    plot!(fig_dist, a_grid, D_ss[:, j],
        lw=1.5, color=col, ls=:dash, label=lbl, alpha=0.8)
end
plot!(fig_dist, legend=:topright)

savefig(fig_dist, joinpath(@__DIR__, "../plots/stationary_distribution.png"))
println("✓ Saved plots/stationary_distribution.png")

T_hm = 30
J_plot = res_dcp.hh.J_Cr[1:T_hm, 1:T_hm]

fig_jac = heatmap(1:T_hm, 1:T_hm, J_plot,
    color=:RdBu,
    xlabel="Shock period s (∂R_s)",
    ylabel="Response period t (∂C_t)",
    title="Full Sequence-Space Consumption Jacobian J_{C,R}",
    dpi=150, clims=(-maximum(abs.(J_plot)), maximum(abs.(J_plot))))

savefig(fig_jac, joinpath(@__DIR__, "../plots/jacobian_heatmap.png"))
println("✓ Saved plots/jacobian_heatmap.png")

println("\n=== Done. All plots saved in plots/ ===")
