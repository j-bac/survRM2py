import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy
from scipy.stats import norm


# --- 1. Python Native RMST Implementation ---
def exact_rmst1(y, d, tau):
    """
    Exact translation of survRM2::rmst1.
    Returns: RMST, RMST_Variance
    """
    # 1. Exact surv computation like R
    t_idx = np.unique(np.insert(y, 0, 0.0))
    ny = len(y)

    Y = np.zeros(len(t_idx))
    N = np.zeros(len(t_idx))
    C = np.zeros(len(t_idx))
    S = np.zeros(len(t_idx))

    Y[0] = ny
    S[0] = 1.0

    for i in range(1, len(t_idx)):
        Y[i] = Y[i - 1] - N[i - 1] - C[i - 1]
        N[i] = np.sum((y == t_idx[i]) & (d == 1))
        C[i] = np.sum((y == t_idx[i]) & (d == 0))
        if Y[i] < 0:
            Y[i] = 0
        S[i] = S[i - 1] if Y[i] == 0 else S[i - 1] * (1 - (N[i] / Y[i]))
        if S[i] < 0:
            S[i] = 0

    # Remove t=0 to match R's `ft$time`
    ft_time = t_idx[1:]
    ft_surv = S[1:]
    ft_n_risk = Y[1:]
    ft_n_event = N[1:]

    # Filter times <= tau
    idx = ft_time <= tau

    # R: wk.time = sort(c(ft$time[idx], tau))
    wk_time = np.sort(np.append(ft_time[idx], tau))
    wk_surv = ft_surv[idx]
    wk_n_risk = ft_n_risk[idx]
    wk_n_event = ft_n_event[idx]

    # R: time.diff <- diff(c(0, wk.time))
    time_diff = np.diff(np.insert(wk_time, 0, 0.0))

    # R: areas <- time.diff * c(1, wk.surv)
    areas = time_diff * np.insert(wk_surv, 0, 1.0)
    rmst = np.sum(areas)

    # R: wk.var <- ifelse(...) Greenwood Formula
    wk_var = np.zeros(len(wk_n_risk))
    valid = (wk_n_risk - wk_n_event) > 0
    wk_var[valid] = wk_n_event[valid] / (wk_n_risk[valid] * (wk_n_risk[valid] - wk_n_event[valid]))

    # R: wk.var = c(wk.var, 0)
    wk_var = np.append(wk_var, 0.0)

    # R: rmst.var = sum( cumsum(rev(areas[-1]))^2 * rev(wk.var)[-1] )
    areas_minus_1 = areas[1:]  # Drop first area
    cumsum_rev_areas = np.cumsum(areas_minus_1[::-1]) ** 2
    rev_wk_var_minus_1 = wk_var[::-1][1:]  # Reverse, then drop first element

    rmst_var = np.sum(cumsum_rev_areas * rev_wk_var_minus_1)

    return rmst, rmst_var


# =====================================================================
# 2. SURVRM2 EXACT ADJUSTED TRANSLATION
# =====================================================================
def func_surv(y, d):
    t_idx = np.unique(np.insert(y, 0, 0.0))
    ny = len(y)
    Y, N, C, S = np.zeros(len(t_idx)), np.zeros(len(t_idx)), np.zeros(len(t_idx)), np.zeros(len(t_idx))
    Y[0], S[0] = ny, 1.0
    for i in range(1, len(t_idx)):
        Y[i] = Y[i - 1] - N[i - 1] - C[i - 1]
        N[i] = np.sum((y == t_idx[i]) & (d == 1))
        C[i] = np.sum((y == t_idx[i]) & (d == 0))
        if Y[i] < 0:
            Y[i] = 0
        S[i] = S[i - 1] if Y[i] == 0 else S[i - 1] * (1 - (N[i] / Y[i]))
        if S[i] < 0:
            S[i] = 0
    return S[1:]


def rmst2reg(y, delta, X_matrix, arm, tau):
    """Exact translation of survRM2::rmst2reg (type='difference')"""
    # R: x = cbind(1, x)
    n, p = X_matrix.shape  # intercept already there
    # x = np.column_stack([np.ones(len(y)), X_matrix])
    # n = len(y)
    # p = x.shape[1]

    # R: y0=pmin(y, tau); d0=delta; d0[y0==tau]=1
    y0 = np.minimum(y, tau)
    d0 = delta.copy()
    d0[y0 == tau] = 1

    # Split by arm
    idx1 = arm == 1
    idx0 = arm == 0

    d10, y10, x1 = d0[idx1], y0[idx1], X_matrix[idx1]
    d00, y00, x0 = d0[idx0], y0[idx0], X_matrix[idx0]

    # R: id1=order(y10) (using stable sort to match R exactly)
    sort1 = np.argsort(y10, kind="stable")
    y10, d10, x1 = y10[sort1], d10[sort1], x1[sort1]

    sort0 = np.argsort(y00, kind="stable")
    y00, d00, x0 = y00[sort0], d00[sort0], x0[sort0]

    # R: fitc1=func_surv(y10, 1-d10)
    fitc1_surv = func_surv(y10, 1 - d10)
    fitc0_surv = func_surv(y00, 1 - d00)

    # R: weights1=d10/rep(fitc1$surv, table(y10))
    _, counts1 = np.unique(y10, return_counts=True)
    weights1 = d10 / np.repeat(fitc1_surv, counts1)

    _, counts0 = np.unique(y00, return_counts=True)
    weights0 = d00 / np.repeat(fitc0_surv, counts0)

    weights = np.concatenate([weights1, weights0])

    # R: fitt=lm(c(y10,y00)~rbind(x1, x0)-1, weights=weights)
    Y_reg = np.concatenate([y10, y00])
    X_reg = np.vstack([x1, x0])
    fitt = sm.WLS(Y_reg, X_reg, weights=weights).fit()
    beta0 = fitt.params

    # ---kappa.arm1---
    # R: error1=y10-as.vector(x1%*%beta0)
    error1 = y10 - X_reg[: len(y10)] @ beta0
    score1 = X_reg[: len(y10)] * (weights1 * error1)[:, None]

    n1 = len(y10)
    pos1 = np.repeat(np.cumsum(counts1) - counts1, counts1)  # Matches y10pos - 1
    loc1 = n1 - pos1  # Matches y10loc
    forward1 = np.repeat(np.cumsum(counts1) - 1, counts1)  # Matches y10forward - 1

    tab1 = np.zeros((n1, p))
    for i in range(p):
        temp = np.cumsum(score1[::-1, i])[::-1]
        tab1[:, i] = temp[pos1]

    kappa2_1 = tab1 * (1 - d10)[:, None] / loc1[:, None]
    kappa3a_1 = kappa2_1 / loc1[:, None]
    kappa3b_1 = np.cumsum(kappa3a_1, axis=0)
    kappa3c_1 = kappa3b_1[forward1, :]
    kappa_arm1 = score1 + kappa2_1 - kappa3c_1

    # ---kappa.arm0---
    error0 = y00 - X_reg[len(y10) :] @ beta0
    score0 = X_reg[len(y10) :] * (weights0 * error0)[:, None]

    n0 = len(y00)
    pos0 = np.repeat(np.cumsum(counts0) - counts0, counts0)
    loc0 = n0 - pos0
    forward0 = np.repeat(np.cumsum(counts0) - 1, counts0)

    tab0 = np.zeros((n0, p))
    for i in range(p):
        temp = np.cumsum(score0[::-1, i])[::-1]
        tab0[:, i] = temp[pos0]

    kappa2_0 = tab0 * (1 - d00)[:, None] / loc0[:, None]
    kappa3a_0 = kappa2_0 / loc0[:, None]
    kappa3b_0 = np.cumsum(kappa3a_0, axis=0)
    kappa3c_0 = kappa3b_0[forward0, :]
    kappa_arm0 = score0 + kappa2_0 - kappa3c_0

    # --- Variance Calculation ---
    gamma = kappa_arm1.T @ kappa_arm1 + kappa_arm0.T @ kappa_arm0
    A = X_matrix.T @ X_matrix
    A_inv = np.linalg.inv(A)
    varbeta = A_inv @ gamma @ A_inv

    return beta0, varbeta


def rmst(df, time_col, event_col, arm_col, tau, formula=None, method="ipcw_rmst2", alpha=0.05):
    """
    Performs unadjusted and adjusted RMST comparisons.
    Adjusted analysis is handled exclusively using patsy formulas to prevent collinearity bugs.
    """
    if method in ["pseudo", "ipcw"]:
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.utils import restricted_mean_survival_time
        except ImportError:
            raise ImportError("Please install lifelines for method 'pseudo' or 'ipcw': pip install lifelines")

    T = df[time_col].values.astype(float)
    E = df[event_col].values.astype(float)
    arm = df[arm_col].values.astype(float)
    n = len(T)
    results = {}

    # 1. Unadjusted RMST (Always use exact_rmst1)
    idx1, idx0 = (arm == 1), (arm == 0)
    rmst_arm1, var_arm1 = exact_rmst1(T[idx1], E[idx1], tau)
    rmst_arm0, var_arm0 = exact_rmst1(T[idx0], E[idx0], tau)

    diff_unadj = rmst_arm1 - rmst_arm0
    se_unadj = np.sqrt(var_arm1 + var_arm0)
    z_unadj = diff_unadj / se_unadj
    p_unadj = 2 * (1 - norm.cdf(abs(z_unadj)))
    z_alpha = norm.ppf(1 - alpha / 2)

    results["rmst_arm1"] = rmst_arm1
    results["rmst_arm0"] = rmst_arm0
    results["rmst_diff_unadjusted"] = diff_unadj
    results["se_unadjusted"] = se_unadj
    results["p_unadjusted"] = p_unadj
    results["ci_unadjusted_lower"] = diff_unadj - z_alpha * se_unadj
    results["ci_unadjusted_upper"] = diff_unadj + z_alpha * se_unadj

    # 2. Adjusted RMST (Formula Only)
    if formula is not None:
        # Patsy creates the matrix, handles interactions, and adds intercept automatically
        dmatrix = patsy.dmatrix(formula, df, return_type="dataframe")
        X_matrix = np.asarray(dmatrix)
        var_names = dmatrix.design_info.column_names

        coefs, ses, p_vals, ci_lows, ci_highs = None, None, None, None, None

        if method == "pseudo":
            kmf_all = KaplanMeierFitter().fit(T, E)
            theta_hat = restricted_mean_survival_time(kmf_all, t=tau)
            pseudo_values = np.zeros(n)
            for i in range(n):
                T_i, E_i = np.delete(T, i), np.delete(E, i)
                kmf_i = KaplanMeierFitter().fit(T_i, E_i)
                pseudo_values[i] = n * theta_hat - (n - 1) * restricted_mean_survival_time(kmf_i, t=tau)

            model = sm.OLS(pseudo_values, X_matrix).fit(cov_type="HC0")
            coefs, ses = model.params, model.bse
            p_vals = model.pvalues
            ci_lows, ci_highs = model.conf_int(alpha=alpha).T

        elif method == "ipcw":
            Y = np.minimum(T, tau)
            delta_star = np.where((T > tau) | ((T <= tau) & (E == 1)), 1, 0)
            weights = np.zeros(n)
            for arm_val in [0, 1]:
                mask = arm == arm_val
                kmf_censor = KaplanMeierFitter().fit(Y[mask], 1 - delta_star[mask])
                G_hat = kmf_censor.survival_function_at_times(Y[mask]).values
                w_arm = np.zeros(sum(mask))
                valid = G_hat > 0
                w_arm[valid] = delta_star[mask][valid] / G_hat[valid]
                weights[mask] = w_arm

            model = sm.WLS(Y, X_matrix, weights=weights).fit(cov_type="HC0")
            coefs, ses = model.params, model.bse
            p_vals = model.pvalues
            ci_lows, ci_highs = model.conf_int(alpha=alpha).T

        elif method == "ipcw_rmst2":
            beta0, varbeta = rmst2reg(T, E, X_matrix, arm, tau)
            coefs = beta0
            ses = np.sqrt(np.diag(varbeta))
            z_stats = coefs / ses
            p_vals = 2 * (1 - norm.cdf(np.abs(z_stats)))
            ci_lows = coefs - z_alpha * ses
            ci_highs = coefs + z_alpha * ses

        alpha_pct = int((1 - alpha) * 100)
        adjusted_summary = pd.DataFrame(
            {
                "covariate": var_names,
                "coef": coefs,
                "se(coef)": ses,
                "z": coefs / ses,
                "p": p_vals,
                f"lower .{alpha_pct}": ci_lows,
                f"upper .{alpha_pct}": ci_highs,
            }
        )

        adjusted_summary.index.name = "covariate"
        adjusted_summary = adjusted_summary.reset_index(drop=True)

        # Dynamically locate the treatment coefficient index
        arm_idx = var_names.index(arm_col) if arm_col in var_names else 1
        results["rmst_diff_adjusted"] = coefs[arm_idx]
        results["p_adjusted"] = p_vals[arm_idx]
        results["adjusted_summary"] = adjusted_summary

    return results
