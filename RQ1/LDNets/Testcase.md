- 1a — 1D advection–diffusion–reaction PDE, constant inputs (μ1, μ2, μ3), 2 latent states; shows LDNet learns exact 2D intrinsic (Fourier) state.
- 1b — same PDE with time-dependent forcing (amplitude/phase signals); sweeps #training samples (25–400), Δt, and #latent states (1–5); latent space maps to Fourier coefficients.
- 1c — forcing with time-varying frequency (fmax = 0.5/1/2), infinite-dimensional solution manifold; sweeps #latent states (2–7) and compares vs. an (x, u)-ODE-Net with no latent states (they prove crucial).
- 2 — unsteady 2D lid-driven cavity Navier–Stokes (Re up to ~1500), time-dependent lid velocity; direction-weighted loss; 80 train (T=20) / 200 test (T=40); tests interpolation and time-extrapolation with 1/5/10 latent states.
- 3 — 1D Monodomain + Aliev–Panfilov cardiac electrophysiology with two stimulation points; benchmark vs. POD-DEIM, AE/ODE, AE/LSTM (e2e variants); LDNet best accuracy with ~10x fewer parameters.
- 4 — 2D reentrant electrophysiology (refractoriness / focal activation / reentrant drivers via circular stimulus); version in TestCase_4.py adds a stimulus-diameter input parameter; benchmark vs. AE/ODE and AE/ODE-e2e, NRMSE ~7.6e-3 vs. 4–8e-2.

