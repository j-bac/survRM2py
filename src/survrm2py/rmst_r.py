import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

def rmst_r(df, time_col, event_col, arm_col, tau, covariates=None, alpha=0.05):
    r_df = df.rename(columns={time_col: "time", event_col: "event", arm_col: "arm"})

    ro.r("suppressPackageStartupMessages(library(survRM2))")
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        ro.globalenv["surv_data"] = cv.py2rpy(r_df)
    ro.globalenv["tau"] = float(tau)

    # Unadjusted RMST
    ro.r("""
    rmst_unadj <- rmst2(time=surv_data$time, status=surv_data$event, arm=surv_data$arm, tau=tau)
    """)
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

    # Adjusted RMST
    if covariates is not None and len(covariates) > 0:
        cov_r_str = ", ".join([f'"{c}"' for c in covariates])
        ro.r(f"""
        cov_mat <- as.matrix(surv_data[, c({cov_r_str})])
        rmst_adj <- rmst2(time=surv_data$time, status=surv_data$event, arm=surv_data$arm, tau=tau, covariates=cov_mat)
        """)

        # Manually extract flat arrays to completely bypass rpy2 DataFrame conversion bugs
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
            coefs = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 1]"))
            ses = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 2]"))
            z_stats = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 3]"))
            p_vals = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 4]"))
            ci_lows = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 5]"))
            ci_highs = np.array(ro.r("rmst_adj$RMST.difference.adjusted[, 6]"))

        var_names = ["Intercept", arm_col] + covariates

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
        results["rmst_diff_adjusted"] = float(adj_df.loc[adj_df["covariate"] == arm_col, "coef"].values[0])
        results["p_adjusted"] = float(adj_df.loc[adj_df["covariate"] == arm_col, "p"].values[0])

    return results
