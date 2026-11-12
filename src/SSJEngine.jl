module SSJEngine

using FastEGM

export compute_household_jacobians

function perturbed_egm_step!(
    ws::EGMWorkspace,
    c_next::Matrix{Float64},
    p::ModelParams,
    R_t::Float64,
    w_t::Float64
)
    @inbounds for iy in 1:p.Ny
        for ia in 1:p.Na
            emu = 0.0
            for jy in 1:p.Ny
                emu += p.P[iy, jy] * (c_next[ia, jy]^(-p.γ))
            end
            ws.EMU[ia, iy] = emu
        end
    end

    inv_γ = 1.0 / p.γ
    @inbounds for iy in 1:p.Ny
        y = p.y_grid[iy] * w_t
        for ia in 1:p.Na
            c = (p.β * R_t * ws.EMU[ia, iy])^(-inv_γ)
            ws.c_endo[ia, iy] = c
            ws.a_endo[ia, iy] = (c + p.a_grid[ia] - y) / R_t
        end

        for ia in 1:p.Na
            cur_a = p.a_grid[ia]
            if cur_a <= ws.a_endo[1, iy]
                ws.a_new[ia, iy] = p.a_min
                ws.c_new[ia, iy] = cur_a * R_t + y - p.a_min
            else
                k = searchsortedlast(@view(ws.a_endo[:, iy]), cur_a)
                k = min(max(k, 1), p.Na - 1)
                x0, x1 = ws.a_endo[k, iy], ws.a_endo[k+1, iy]
                y0, y1 = p.a_grid[k], p.a_grid[k+1]
                slope = (y1 - y0) / (x1 - x0)
                a_prime = y0 + slope * (cur_a - x0)

                ws.a_new[ia, iy] = a_prime
                ws.c_new[ia, iy] = cur_a * R_t + y - a_prime
            end
        end
    end
    return nothing
end

function backward_expectation_step!(
    E_next::Matrix{Float64},
    E_curr::Matrix{Float64},
    dws::DistWorkspace,
    p::ModelParams
)
    @inbounds for iy in 1:p.Ny
        for ia in 1:p.Na
            k = dws.idx[ia, iy]
            wl = dws.w_low[ia, iy]
            wh = 1.0 - wl
            val = 0.0
            for jy in 1:p.Ny
                interp_val = wl * E_curr[k, jy] + wh * E_curr[k+1, jy]
                val += p.P[iy, jy] * interp_val
            end
            E_next[ia, iy] = val
        end
    end
    return nothing
end

function compute_household_jacobians(
    c_ss::Matrix{Float64},
    a_ss::Matrix{Float64},
    D_ss::Matrix{Float64},
    dws_ss::DistWorkspace,
    p::ModelParams;
    T::Int = 200,
    ϵ::Float64 = 1e-5
)
    E_c = [zeros(p.Na, p.Ny) for _ in 1:T]
    E_a = [zeros(p.Na, p.Ny) for _ in 1:T]
    E_c[1] .= c_ss
    E_a[1] .= a_ss

    for t in 2:T
        backward_expectation_step!(E_c[t], E_c[t-1], dws_ss, p)
        backward_expectation_step!(E_a[t], E_a[t-1], dws_ss, p)
    end

    function jacobian_wrt(shock_var::Symbol)
        ws_p = EGMWorkspace(p.Na, p.Ny)
        ws_m = EGMWorkspace(p.Na, p.Ny)
        dws_p = DistWorkspace(p.Na, p.Ny)
        dws_m = DistWorkspace(p.Na, p.Ny)

        c_next_p = copy(c_ss)
        c_next_m = copy(c_ss)
        D_next_p = zeros(p.Na, p.Ny)
        D_next_m = zeros(p.Na, p.Ny)

        F_c = zeros(T, T)
        F_a = zeros(T, T)

        for s in 1:T
            if s == 1
                R_p = shock_var == :R ? p.R + ϵ : p.R
                R_m = shock_var == :R ? p.R - ϵ : p.R
                w_p = shock_var == :w ? p.w + ϵ : p.w
                w_m = shock_var == :w ? p.w - ϵ : p.w

                perturbed_egm_step!(ws_p, c_ss, p, R_p, w_p)
                perturbed_egm_step!(ws_m, c_ss, p, R_m, w_m)
            else
                egm_step!(ws_p, c_next_p, p)
                egm_step!(ws_m, c_next_m, p)
            end

            dc = (ws_p.c_new .- ws_m.c_new) ./ (2 * ϵ)
            da = (ws_p.a_new .- ws_m.a_new) ./ (2 * ϵ)

            F_c[1, s] = sum(D_ss .* dc)
            F_a[1, s] = sum(D_ss .* da)

            lottery_weights!(dws_p, ws_p.a_new, p)
            lottery_weights!(dws_m, ws_m.a_new, p)
            forward_policy_step!(D_next_p, D_ss, dws_p, p)
            forward_policy_step!(D_next_m, D_ss, dws_m, p)
            dD = (D_next_p .- D_next_m) ./ (2 * ϵ)

            for t in 2:T
                F_c[t, s] = sum(E_c[t-1] .* dD)
                F_a[t, s] = sum(E_a[t-1] .* dD)
            end

            c_next_p .= ws_p.c_new
            c_next_m .= ws_m.c_new
        end

        J_c = copy(F_c)
        J_a = copy(F_a)
        for s in 2:T, t in 2:T
            J_c[t, s] += J_c[t-1, s-1]
            J_a[t, s] += J_a[t-1, s-1]
        end

        return J_c, J_a
    end

    J_CR, J_AR = jacobian_wrt(:R)
    J_Cw, J_Aw = jacobian_wrt(:w)

    return (J_CR = J_CR, J_AR = J_AR, J_Cw = J_Cw, J_Aw = J_Aw)
end

end # module
