# SOE-HANK-DCP: Sequence-Space Jacobians in a Small Open Economy HANK Model under Dominant Currency Pricing

[![CI](https://github.com/anujagokhale1604/SOE-HANK-DCP/actions/workflows/CI.yml/badge.svg)](https://github.com/anujagokhale1604/SOE-HANK-DCP/actions/workflows/CI.yml)
[![Interactive Dashboard](https://img.shields.io/badge/Interactive_Dashboard-GitHub_Pages-1e3a8a?style=for-the-badge)](https://anujagokhale1604.github.io/SOE-HANK-DCP/)
[![Technical Note](https://img.shields.io/badge/Technical_Note-PDF-9f1239?style=for-the-badge)](https://anujagokhale1604.github.io/SOE-HANK-DCP/technical_note.pdf)
[![Companion App](https://img.shields.io/badge/Empirical_Inflation_App-Streamlit-065f46?style=for-the-badge)](https://pink-panther-inflation.streamlit.app/)

A pure-Julia computational laboratory solving a **Small Open Economy Heterogeneous-Agent New Keynesian (SOE-HANK)** model to quantify how US dollar trade invoicing (**Dominant Currency Pricing**, $\phi = 1$; Gopinath et al., 2020) reshapes monetary transmission and external interest rate spillovers relative to **Producer Currency Pricing** ($\phi = 0$).

---

## Live Links & Research Artifacts

* **Interactive Experiment Dashboard:** [anujagokhale1604.github.io/SOE-HANK-DCP/](https://anujagokhale1604.github.io/SOE-HANK-DCP/) — Features a continuous US dollar invoicing slider ($\phi \in [0, 1]$), a **HANK vs. RANK** decomposition overlay, 40-quarter impulse response charts, and a plain-English / technical equations toggle.
* **4-Page Technical Note (PDF):** [Download `technical_note.pdf`](https://anujagokhale1604.github.io/SOE-HANK-DCP/technical_note.pdf) — Complete model equations, Fake-News Jacobian derivation, household diagnostics, and policy implications.
* **Companion Empirical Macro Tool:** [pink-panther-inflation.streamlit.app/](https://pink-panther-inflation.streamlit.app/) — Live empirical inflation and monetary policy dashboard.

---

## Computational Architecture

1. **Zero-Allocation Endogenous Grid Method (`src/FastEGM.jl`):** Solves incomplete-markets consumption-savings policies over a $150 \times 5$ asset-income grid ($a' \ge \underline{a}$) with Rouwenhorst (1995) labor productivity transitions and Young (2010) non-stochastic lottery histogram iteration, achieving machine-precision convergence with zero inner-loop heap allocations.
2. **Full Non-Toeplitz Fake-News Jacobian Engine (`src/SSJEngine.jl`):** Implements the Auclert et al. (2021) Fake-News algorithm to compute exact $T \times T$ ($T = 60$) household consumption Jacobians $\mathcal{J}_{C,r}$ and $\mathcal{J}_{C,Y}$, capturing both forward-looking Euler anticipation and backward wealth redistribution.
3. **Exact Sequence-Space General Equilibrium (`src/open_economy_dag.jl`):** Stacks endogenous paths $dU = [d\hat{Y};\, dr] \in \mathbb{R}^{2T}$ across an inertial Taylor rule, damped UIP condition, import price pass-through $\pi_t^M = -(1-\phi)\Delta\hat{e}_t$, and NKPC, solving $H_U dU = -H_Z dZ$ via direct matrix inversion to floating-point precision ($\sim 1.1 \times 10^{-16}$).

---

## Quantitative Summary: First-Quarter ($t=1$) Impact Responses

| Experiment | Model & Regime | Real Rate ($dr_1$) | Exch. Rate ($d\hat{e}_1$) | Output ($d\hat{Y}_1$) | CPI Inflation ($d\pi_1$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp I:** Domestic MP ($+100\text{ bp}$) | **HANK — DCP ($\phi = 1$)** | $+1.496\text{ pp}$ | $+1.044\%$ | $-0.373\%$ | $+0.798\text{ pp}$ |
| **Exp I:** Domestic MP ($+100\text{ bp}$) | **HANK — PCP ($\phi = 0$)** | $+2.573\text{ pp}$ | $+0.415\%$ | $-0.057\%$ | $+0.970\text{ pp}$ |
| **Exp I:** Domestic MP ($+100\text{ bp}$) | **RANK — DCP ($\phi = 1$)** | $+1.333\text{ pp}$ | $+2.201\%$ | $-1.756\%$ | $+0.249\text{ pp}$ |
| **Exp II:** US Fed Spillover ($+100\text{ bp}$) | **HANK — DCP ($\phi = 1$)** | $-0.032\text{ pp}$ | $-4.168\%$ | $+2.451\%$ | $+0.412\text{ pp}$ |
| **Exp II:** US Fed Spillover ($+100\text{ bp}$) | **HANK — PCP ($\phi = 0$)** | $-2.190\text{ pp}$ | $-5.552\%$ | $+5.938\%$ | $+1.732\text{ pp}$ |
| **Exp II:** US Fed Spillover ($+100\text{ bp}$) | **RANK — DCP ($\phi = 1$)** | $-0.186\text{ pp}$ | $-2.801\%$ | $+0.522\%$ | $-0.227\text{ pp}$ |

### Key Takeaways
* **Domestic Monetary Tightening:** Under PCP, currency appreciation lowers import prices ($\pi_1^M < 0$), cushioning the initial output drop ($-0.057\%$). Under DCP, rigid USD border pricing shuts down import price deflation, resulting in a $6.5\times$ deeper initial output contraction ($-0.373\%$).
* **US Federal Reserve Spillovers:** A persistent $+100\text{ bp}$ US Fed tightening depreciates the domestic currency ($-4.168\%$ under DCP vs. $-5.552\%$ under PCP). Sticky USD border invoicing under DCP cuts imported CPI inflation pass-through by over $75\%$ ($+0.412\text{ pp}$ vs. $+1.732\text{ pp}$) while halving the expenditure-switching export boom.

---

## Quick Start

```bash
git clone [https://github.com/anujagokhale1604/SOE-HANK-DCP.git](https://github.com/anujagokhale1604/SOE-HANK-DCP.git)
cd SOE-HANK-DCP
julia --project -e 'using Pkg; Pkg.instantiate(); include("run_experiments.jl")'
```

## References
* Auclert, A., Bardóczy, B., Rognlie, M., & Straub, L. (2021). Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models. *Econometrica*, 89(5), 2375–2408.
* Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), 312–320.
* Gopinath, G., Boz, E., Casas, C., Díez, F. J., Gourinchas, P.-O., & Plagborg-Møller, M. (2020). Dominant Currency Paradigm. *American Economic Review*, 110(3), 677–719.
* Young, E. R. (2010). Solving the Incomplete Markets Model with Aggregate Uncertainty Using the Krusell–Smith Algorithm and Non-Stochastic Simulations. *Journal of Economic Dynamics and Control*, 34(1), 36–41.