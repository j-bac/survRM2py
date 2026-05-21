import numpy as np
import pandas as pd
import patsy
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter


def rmst_r(df, time_col, event_col, arm_col, tau, formula=None, alpha=0.05):
    """
    R Wrapper.
    """
    r_df = df.rename(columns={time_col: "time", event_col: "event", arm_col: "arm"})

    ro.r("suppressPackageStartupMessages(library(survRM2))")
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        ro.globalenv["surv_data"] = cv.py2rpy(r_df)
    ro.globalenv["tau"] = float(tau)

    # Unadjusted RMST
    ro.r("rmst_unadj <- rmst2(time=surv_data$time, status=surv_data$event, arm=surv_data$arm, tau=tau)")
    unadj = ro.r("rmst_unadj$unadjusted.result[1,]")
    se_unadj = float(ro.r("sqrt(rmst_unadj$RMST.arm1$rmst.var + rmst_unadj$RMST.arm0$rmst.var)")[0])

    results = {
        "rmst_arm1": float(ro.r("rmst_unadj$RMST.arm1$rmst['Est.']")[0]),
        "rmst_arm0": float(ro.r("rmst_unadj$RMST.arm0$rmst['Est.']")[0]),
        "rmst_diff_unadjusted": float(unadj[0]),
        "se_unadjusted": se_unadj,
        "p_unadjusted": float(unadj[3]),
        "ci_unadjusted_lower": float(unadj[1]),
        "ci_unadjusted_upper": float(unadj[2]),
    }

    # Adjusted RMST (Formula Only)
    if formula is not None:
        # Translate Python formula's arm column to R's standardized "arm" name
        dmatrix = patsy.dmatrix(formula, df, return_type="dataframe")
        var_names = dmatrix.design_info.column_names

        # Non-brittle, parsed validation check
        if arm_col not in var_names:
            raise ValueError(
                f"The treatment column '{arm_col}' must be explicitly included as a main effect in your formula "
                f"(e.g., '{arm_col} + covariates' or '{arm_col} * covariates')."
            )

        # Map Python's arm column to R's standardized "arm" name inside R's formula
        r_formula = formula.replace(arm_col, "arm")
        ro.globalenv["formula_str"] = f"~ {r_formula}"

        ro.r("""
        X <- model.matrix(as.formula(formula_str), data=surv_data)
        
        # We must strip both the "(Intercept)" column AND the "arm" column.
        # This is because survRM2 automatically prepends the treatment "arm" internally
        cols_to_keep <- !colnames(X) %in% c("(Intercept)", "arm")
        X_covs <- X[, cols_to_keep, drop=FALSE]
        
        rmst_adj <- rmst2(time=surv_data$time, status=surv_data$event, arm=surv_data$arm, tau=tau, covariates=X_covs)
        """)

        # Get column names from Patsy directly to ensure exact variable labelling matches
        dmatrix = patsy.dmatrix(formula, df, return_type="dataframe")
        var_names = dmatrix.design_info.column_names

        # Extract matrices directly
        coefs = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 1]")))
        ses = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 2]")))
        z_stats = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 3]")))
        p_vals = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 4]")))
        ci_lows = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 5]")))
        ci_highs = np.array(list(ro.r("rmst_adj$RMST.difference.adjusted[, 6]")))

        alpha_pct = int((1 - alpha) * 100)
        adj_df = pd.DataFrame(
            {
                "covariate": var_names,
                "coef": coefs,
                "se(coef)": ses,
                "z": z_stats,
                "p": p_vals,
                f"lower .{alpha_pct}": ci_lows,
                f"upper .{alpha_pct}": ci_highs,
            }
        )

        results["adjusted_summary"] = adj_df
        arm_idx = var_names.index(arm_col) if arm_col in var_names else 1
        results["rmst_diff_adjusted"] = coefs[arm_idx]
        results["p_adjusted"] = p_vals[arm_idx]

    return results
