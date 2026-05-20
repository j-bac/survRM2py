import pytest
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import norm
from survrm2py.rmst import exact_rmst1, rmst2reg, run_rmst, func_surv
from survrm2py.rmst_r import run_rmst_r

# =====================================================================
# 3. PYTEST FIXTURES
# =====================================================================
@pytest.fixture(scope="module")
def real_survival_data():
    ro.r("suppressPackageStartupMessages(library(survival))")
    ro.r('data(pbc, package="survival")')

    # 1. Get the R object
    pbc_r = ro.globalenv["pbc"]

    # 2. Force conversion to Pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        df = cv.rpy2py(pbc_r)
        # If it returns a recarray, convert it explicitly
        if isinstance(df, np.recarray):
            df = pd.DataFrame(df)

    df = df.dropna(subset=["time", "status", "trt", "age", "bili", "protime"]).copy()
    df["event"] = (df["status"] == 2).astype(int)
    df["arm"] = (df["trt"] == 1).astype(int)
    # ... rest of your code ...
    return df


@pytest.fixture(scope="module")
def rmst2_sample_data():
    ro.r("suppressPackageStartupMessages(library(survRM2))")
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        df = cv.rpy2py(ro.r('rmst2.sample.data(t.unit="year")'))
        # If it returns a recarray, convert it explicitly
        if isinstance(df, np.recarray):
            df = pd.DataFrame(df)

    df = df.rename(columns={"status": "event"})
    np.random.seed(42)
    df["mock_cov"] = np.random.normal(50, 10, size=len(df))
    return df


# =====================================================================
# 4. DIRECT UNIT TESTS (Testing Sub-Functions Directly)
# =====================================================================
def test_func_surv_direct(rmst2_sample_data):
    """Directly unit test func_surv vs survRM2:::func_surv"""
    df = rmst2_sample_data
    df_arm1 = df[df["arm"] == 1].copy()
    y_raw = df_arm1["time"].values.astype(float)
    d_raw = df_arm1["event"].values.astype(float)

    sort_idx = np.argsort(y_raw, kind="stable")
    y_sorted = y_raw[sort_idx]
    d_sorted = d_raw[sort_idx]

    # Python
    surv_py = func_surv(y_sorted, d_sorted)

    # R (Passed strictly as structured R dataframe)
    r_df = df_arm1.rename(columns={"time": "time", "event": "event", "arm": "arm"})
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        ro.globalenv["surv_data_arm1"] = cv.py2rpy(r_df)

    ro.r("""
    y_r <- surv_data_arm1$time
    d_r <- surv_data_arm1$event
    id <- order(y_r)
    y_sorted_r <- y_r[id]
    d_sorted_r <- d_r[id]
    fit_r <- survRM2:::func_surv(y_sorted_r, d_sorted_r)
    """)
    surv_r = np.array(ro.r("fit_r$surv"))

    np.testing.assert_almost_equal(surv_py, surv_r, decimal=12)


def test_exact_rmst1_direct(rmst2_sample_data):
    """Directly unit test exact_rmst1 vs survRM2:::rmst1"""
    df = rmst2_sample_data
    df_arm1 = df[df["arm"] == 1].copy()
    y_raw = df_arm1["time"].values.astype(float)
    d_raw = df_arm1["event"].values.astype(float)
    tau = 10.0

    # Python
    rmst_py, var_py = exact_rmst1(y_raw, d_raw, tau)

    # R (Passed strictly as structured R dataframe)
    r_df = df_arm1.rename(columns={"time": "time", "event": "event", "arm": "arm"})
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        ro.globalenv["surv_data_arm1"] = cv.py2rpy(r_df)

    ro.r("""
    y_r <- surv_data_arm1$time
    d_r <- surv_data_arm1$event
    fit_r <- survRM2:::rmst1(y_r, d_r, tau=10.0)
    """)
    rmst_r = float(ro.r("fit_r$rmst['Est.']")[0])
    var_r = float(ro.r("fit_r$rmst.var")[0])

    np.testing.assert_almost_equal(rmst_py, rmst_r, decimal=12)
    np.testing.assert_almost_equal(var_py, var_r, decimal=12)


def test_rmst2reg_direct(rmst2_sample_data):
    """Directly unit test rmst2reg vs survRM2:::rmst2reg"""
    df = rmst2_sample_data
    y = df["time"].values.astype(float)
    delta = df["event"].values.astype(float)
    arm = df["arm"].values.astype(float)
    covs = df[["mock_cov"]].values.astype(float)
    tau = 10.0

    x_in = np.column_stack((arm, covs))
    x = np.column_stack([np.ones(len(y)), x_in])

    # Python
    beta_py, varbeta_py = rmst2reg(y, delta, x, arm, tau)

    # R (Passed strictly as structured R dataframe)
    r_df = df.rename(columns={"time": "time", "event": "event", "arm": "arm"})
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter) as cv:
        ro.globalenv["surv_data_r"] = cv.py2rpy(r_df)

    ro.r("""
    y_r <- surv_data_r$time
    delta_r <- surv_data_r$event
    arm_r <- surv_data_r$arm
    cov_r <- as.matrix(surv_data_r[, "mock_cov", drop=FALSE])
    x_r <- cbind(arm_r, cov_r)
    
    fit_r <- survRM2:::rmst2reg(y_r, delta_r, x_r, arm_r, tau=10.0)
    """)

    # Extract coefficients and covariance
    beta_r = np.array(ro.r("fit_r[, 'coef']"))
    se_r = np.array(ro.r("fit_r[, 'se(coef)']"))
    p_r = np.array(ro.r("fit_r[, 'p']"))

    se_py = np.sqrt(np.diag(varbeta_py))
    z_stat = beta_py / se_py
    p_py = 2 * (1 - norm.cdf(abs(z_stat)))

    np.testing.assert_almost_equal(beta_py, beta_r, decimal=12)
    np.testing.assert_almost_equal(se_py, se_r, decimal=12)
    np.testing.assert_almost_equal(p_py, p_r, decimal=12)


# =====================================================================
# 5. GENERAL INTEGRATION TESTS (Aligned DataFrame Columns)
# =====================================================================
def test_unadjusted_rmst(real_survival_data):
    df = real_survival_data
    tau = 3000.0

    res_r = run_rmst_r(df, "time", "event", "arm", tau)
    res_py = run_rmst(df, "time", "event", "arm", tau)

    np.testing.assert_almost_equal(res_py["rmst_arm1"], res_r["rmst_arm1"], decimal=12)
    np.testing.assert_almost_equal(res_py["rmst_arm0"], res_r["rmst_arm0"], decimal=12)
    np.testing.assert_almost_equal(res_py["rmst_diff_unadjusted"], res_r["rmst_diff_unadjusted"], decimal=12)
    np.testing.assert_almost_equal(res_py["se_unadjusted"], res_r["se_unadjusted"], decimal=12)
    np.testing.assert_almost_equal(res_py["p_unadjusted"], res_r["p_unadjusted"], decimal=12)
    np.testing.assert_almost_equal(res_py["ci_unadjusted_lower"], res_r["ci_unadjusted_lower"], decimal=12)
    np.testing.assert_almost_equal(res_py["ci_unadjusted_upper"], res_r["ci_unadjusted_upper"], decimal=12)


def test_adjusted_rmst_single_covariate(real_survival_data):
    df = real_survival_data
    tau = 2500.0
    covs = ["age"]

    res_r = run_rmst_r(df, "time", "event", "arm", tau, covariates=covs)
    res_py = run_rmst(df, "time", "event", "arm", tau, covariates=covs)

    pd.testing.assert_frame_equal(
        res_py["adjusted_summary"].set_index("covariate"),
        res_r["adjusted_summary"].set_index("covariate"),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_adjusted_rmst_multiple_covariates(real_survival_data):
    df = real_survival_data
    tau = 2000.0
    covs = ["age", "bili", "protime"]

    res_r = run_rmst_r(df, "time", "event", "arm", tau, covariates=covs)
    res_py = run_rmst(df, "time", "event", "arm", tau, covariates=covs)

    pd.testing.assert_frame_equal(
        res_py["adjusted_summary"].set_index("covariate"),
        res_r["adjusted_summary"].set_index("covariate"),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_rmst2_official_sample_data(rmst2_sample_data):
    df = rmst2_sample_data
    tau = 10.0

    # --- 1. Test Unadjusted ---
    res_r_unadj = run_rmst_r(df, "time", "event", "arm", tau)
    res_py_unadj = run_rmst(df, "time", "event", "arm", tau)

    np.testing.assert_almost_equal(res_py_unadj["rmst_arm1"], res_r_unadj["rmst_arm1"], decimal=12)
    np.testing.assert_almost_equal(res_py_unadj["rmst_arm0"], res_r_unadj["rmst_arm0"], decimal=12)
    np.testing.assert_almost_equal(
        res_py_unadj["rmst_diff_unadjusted"], res_r_unadj["rmst_diff_unadjusted"], decimal=12
    )
    np.testing.assert_almost_equal(res_py_unadj["se_unadjusted"], res_r_unadj["se_unadjusted"], decimal=12)
    np.testing.assert_almost_equal(res_py_unadj["p_unadjusted"], res_r_unadj["p_unadjusted"], decimal=12)
    np.testing.assert_almost_equal(res_py_unadj["ci_unadjusted_lower"], res_r_unadj["ci_unadjusted_lower"], decimal=12)
    np.testing.assert_almost_equal(res_py_unadj["ci_unadjusted_upper"], res_r_unadj["ci_unadjusted_upper"], decimal=12)

    # --- 2. Test Adjusted ---
    covs = ["mock_cov"]
    res_r_adj = run_rmst_r(df, "time", "event", "arm", tau, covariates=covs)
    res_py_adj = run_rmst(df, "time", "event", "arm", tau, covariates=covs)

    pd.testing.assert_frame_equal(
        res_py_adj["adjusted_summary"].set_index("covariate"),
        res_r_adj["adjusted_summary"].set_index("covariate"),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
