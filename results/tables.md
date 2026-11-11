
**msd** (DeePC weight λ_g = 100)

| Controller | Tracking cost | Steady-state error | Max violation | Steps violated | Solve time mean / max [ms] |
|---|---|---|---|---|---|
| Nominal MPC | 1.021 ± 0.001 | 0.0420 ± 0.0003 | 0.042 | 11.7 | 4.7 / 14 |
| Oracle MPC | 1.017 ± 0.001 | 0.0006 ± 0.0001 | 0.011 | 20.6 | 4.8 / 9 |
| GP-MPC | 1.018 ± 0.002 | 0.0024 ± 0.0013 | 0.006 | 5.3 | 9.9 / 20 |
| DeePC | 1.046 ± 0.006 | 0.0056 ± 0.0048 | 0.006 | 18.8 | 15.3 / 234 |
| Koopman MPC | 1.015 ± 0.002 | 0.0012 ± 0.0006 | 0.012 | 24.6 | 0.3 / 1 |

**pendulum** (DeePC weight λ_g = 100)

| Controller | Tracking cost | Steady-state error | Max violation | Steps violated | Solve time mean / max [ms] |
|---|---|---|---|---|---|
| Nominal MPC | 0.238 ± 0.001 | 0.0112 ± 0.0002 | 0.000 | 0.0 | 5.7 / 13 |
| Oracle MPC | 0.217 ± 0.000 | 0.0008 ± 0.0002 | 0.007 | 1.4 | 6.7 / 13 |
| GP-MPC | 0.220 ± 0.001 | 0.0011 ± 0.0003 | 0.000 | 0.0 | 10.5 / 21 |
| DeePC | 0.231 ± 0.001 | 0.0035 ± 0.0010 | 0.000 | 0.0 | 3.7 / 11 |
| Koopman MPC | 0.216 ± 0.001 | 0.0011 ± 0.0002 | 0.071 | 2.7 | 0.2 / 1 |

**quadrotor** (DeePC weight λ_g = 1)

| Controller | Tracking cost | Steady-state error | Max violation | Steps violated | Solve time mean / max [ms] |
|---|---|---|---|---|---|
| Nominal MPC | 1.292 ± 0.010 | 0.1043 ± 0.0006 | 0.021 | 10.8 | 9.9 / 26 |
| Oracle MPC | 1.108 ± 0.003 | 0.0020 ± 0.0005 | 0.005 | 6.4 | 10.0 / 31 |
| GP-MPC | 1.117 ± 0.004 | 0.0029 ± 0.0017 | 0.001 | 0.5 | 49.5 / 145 |
| DeePC | 1.321 ± 0.073 | 0.0414 ± 0.0057 | 0.013 | 2.5 | 13.7 / 87 |
| Koopman MPC | 1.194 ± 0.028 | 0.0031 ± 0.0011 | 0.060 | 7.7 | 0.7 / 8 |

**Open-loop prediction error at step 20 (normalised RMSE, held-out data)**

| Model | msd | pendulum | quadrotor |
|---|---|---|---|
| Prior model | 0.563 | 22.480 | 3.562 |
| Prior + GP | 0.018 | 1.306 | 0.127 |
| Koopman (EDMD) | 0.007 | 1.003 | 0.079 |
| Linear (DMDc) | 0.006 | 1.320 | 0.079 |
| Hankel (DeePC) | 0.004 | 0.455 | 0.109 |
| GP 2σ coverage (one-step) | 0.959 | 0.951 | 0.940 |
