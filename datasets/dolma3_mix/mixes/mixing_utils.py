import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.optimize import differential_evolution, fsolve, minimize
from tqdm.auto import tqdm

# ==============================================================
# =                  FUNCTION OPTIMIZATION STUFF               =
# ==============================================================


def power_exponential_function(x, p, lam):
    """
    Power-Exponential function: f(x) = x^p * exp(lambda * x)

    Parameters:
    ----------
    x : array-like
        Input values
    p : float >= 0
        Power parameter (controls behavior near zero)
        p = 0: pure exponential
        p > 0: heavier behavior near zero
    lam : float
        Exponential parameter (lam > 0 for increasing function)
    """
    x = np.asarray(x, dtype=float)

    # Handle x=0 case
    result = np.zeros_like(x, dtype=float)

    # For x > 0
    mask = x > 0
    if np.any(mask):
        result[mask] = (x[mask] ** p) * np.exp(lam * x[mask])

    # For x = 0
    zero_mask = x == 0
    if np.any(zero_mask):
        if p > 0:
            result[zero_mask] = 0  # x^p goes to 0 for p > 0
        else:  # p = 0
            result[zero_mask] = 1  # x^0 * exp(0) = 1

    return result


def compute_power_exp_integral(p, lam, W):
    """
    Compute integral of x^p * exp(lambda * x) from 0 to W.
    Uses integration by parts approach when possible.
    """

    def integrand(x):
        return power_exponential_function(x, p, lam)

    try:
        integral, _ = quad(integrand, 0, W, limit=200)
        return integral
    except:
        return np.inf


def compute_power_exp_tail_average(p, lam, C, W, a):
    """
    Compute average value of C * x^p * exp(lambda * x) over [W-a, W].
    """

    def integrand(x):
        return C * power_exponential_function(x, p, lam)

    try:
        integral, _ = quad(integrand, W - a, W, limit=200)
        return integral / a
    except:
        return np.inf


def compute_power_exp_average(p, lam, C, a, b):
    """
    Compute average value of C * x^p * exp(lambda * x) over [W-a, W].
    """

    def integrand(x):
        return C * power_exponential_function(x, p, lam)

    try:
        integral, _ = quad(integrand, a, b, limit=200)
        return integral / (b - a)
    except:
        return np.inf


def solve_power_exponential_parameters(
    W,
    M,
    a,
    Z,
    p_range=(0.0, 5.0),
    lambda_range=(0.01, 2.0),
    method="constraint_based",
    n_trials=5,
    tolerance=1e-6,
    verbose=False,
):
    """
    Solve for optimal parameters of truncated Power-Exponential function.

    Function form: f(x) = C * x^p * exp(lambda * x) for x in [0, W]

    Constraints:
    - Integral over [0, W] equals Z
    - Average value over [W-a, W] is at most M
    - Function is monotonic increasing (lambda > 0)

    Parameters:
    ----------
    W, M, a, Z : float
        Constraint parameters
    p_range, lambda_range : tuple
        Parameter search ranges
    method : str
        'constraint_based' (recommended) or 'optimization'
    n_trials : int
        Number of optimization trials
    tolerance : float
        Numerical tolerance for constraint satisfaction
    verbose : bool
        If True, print progress and diagnostic information

    Returns:
    -------
    dict: Contains 'p', 'lambda', 'C', 'tail_average', 'feasible', etc.
    """

    # Validate inputs
    if not (0 <= a <= W):
        raise ValueError("Must have 0 < a < W")
    if Z <= 0:
        raise ValueError("Integral Z must be positive")
    if M <= 0:
        raise ValueError("Maximum average M must be positive")

    def compute_normalization_constant(p, lam):
        """Compute C such that integral equals Z"""
        base_integral = compute_power_exp_integral(p, lam, W)
        if base_integral <= 0 or not np.isfinite(base_integral):
            return np.inf
        return Z / base_integral

    def is_feasible(p, lam, verbose_inner=False):
        """Check if (p, lambda) satisfies all constraints"""
        try:
            C = compute_normalization_constant(p, lam)
            if not np.isfinite(C) or C <= 0:
                if verbose_inner:
                    print(f"  C = {C} (invalid)")
                return False, np.inf, np.inf

            tail_avg = compute_power_exp_tail_average(p, lam, C, W, a)
            if not np.isfinite(tail_avg):
                if verbose_inner:
                    print(f"  tail_avg = {tail_avg} (invalid)")
                return False, C, np.inf

            actual_integral = C * compute_power_exp_integral(p, lam, W)
            integral_error = abs(actual_integral - Z)

            constraint_satisfied = (
                tail_avg <= M + tolerance and integral_error <= tolerance
            )

            if verbose_inner:
                print(
                    f"  C = {C:.6f}, tail_avg = {tail_avg:.6f}, constraint = {tail_avg <= M}"
                )

            return constraint_satisfied, C, tail_avg

        except Exception as e:
            if verbose_inner:
                print(f"  Exception: {e}")
            return False, np.inf, np.inf

    if method == "constraint_based":
        # Method 1: Systematic search with constraint verification
        if verbose:
            print("Using constraint-based approach...")

        best_result = None
        best_p = -1

        # Test different p values systematically
        p_test_values = np.linspace(p_range[0], p_range[1], 20)

        for p_test in p_test_values:
            if verbose:
                print(f"\nTesting p = {p_test:.3f}")

            # For this p, find the maximum lambda that satisfies constraints
            def constraint_violation(lam):
                """Returns positive value if constraint violated"""
                feasible, C, tail_avg = is_feasible(p_test, lam)
                if not feasible:
                    return 1000  # Large penalty
                return max(0, tail_avg - M)  # Constraint violation

            # Binary search for maximum feasible lambda
            lambda_low = lambda_range[0]
            lambda_high = lambda_range[1]

            # First check if any solution exists in this range
            if constraint_violation(lambda_low) > tolerance:
                if verbose:
                    print(f"  No feasible solution for p = {p_test:.3f}")
                continue

            # Binary search
            for _ in range(50):  # Should be enough iterations
                lambda_mid = (lambda_low + lambda_high) / 2

                if constraint_violation(lambda_mid) <= tolerance:
                    lambda_low = lambda_mid  # Feasible, try larger lambda
                else:
                    lambda_high = lambda_mid  # Not feasible, reduce lambda

                if abs(lambda_high - lambda_low) < 1e-8:
                    break

            lambda_opt = lambda_low
            feasible, C_opt, tail_avg = is_feasible(
                p_test, lambda_opt, verbose_inner=verbose
            )

            if feasible and p_test > best_p:
                best_p = p_test
                best_result = {
                    "p": p_test,
                    "lambda": lambda_opt,
                    "C": C_opt,
                    "tail_average": tail_avg,
                }
                if verbose:
                    print(f"  ✓ New best solution: p={p_test:.3f}, λ={lambda_opt:.6f}")

        if best_result is None:
            raise RuntimeError(
                "No feasible solution found with constraint-based method"
            )

    else:
        # Method 2: Global optimization (original approach)
        if verbose:
            print("Using optimization approach...")

        def objective_function(params):
            """Minimize constraint violation, then maximize p"""
            p, lam = params

            # Parameter bounds check
            if (
                p < p_range[0]
                or p > p_range[1]
                or lam < lambda_range[0]
                or lam > lambda_range[1]
            ):
                return 1e6

            feasible, C, tail_avg = is_feasible(p, lam)

            if not feasible:
                return 1e6

            # Penalize constraint violations heavily
            constraint_violation = max(0, tail_avg - M)
            if constraint_violation > tolerance:
                return 1e6 + constraint_violation * 1e6

            # Among feasible solutions, prefer higher p
            return -p

        best_score = np.inf
        best_result = None

        for trial in range(n_trials):
            try:
                bounds = [p_range, lambda_range]
                result = differential_evolution(
                    objective_function,
                    bounds,
                    seed=trial,
                    maxiter=2000,
                    atol=1e-12,
                    tol=1e-12,
                    popsize=20,
                )

                if result.fun < best_score:
                    best_score = result.fun
                    p_opt, lambda_opt = result.x
                    feasible, C_opt, tail_avg = is_feasible(p_opt, lambda_opt)

                    if feasible:
                        best_result = {
                            "p": p_opt,
                            "lambda": lambda_opt,
                            "C": C_opt,
                            "tail_average": tail_avg,
                        }

            except Exception as e:
                if verbose:
                    warnings.warn(f"Trial {trial} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("No feasible solution found with optimization method")

    # Verify final result
    p_final = best_result["p"]
    lambda_final = best_result["lambda"]
    C_final = best_result["C"]

    # Double-check all constraints
    actual_integral = C_final * compute_power_exp_integral(p_final, lambda_final, W)
    final_tail_avg = compute_power_exp_tail_average(
        p_final, lambda_final, C_final, W, a
    )

    if verbose:
        print(f"\nFinal verification:")
        print(f"  Integral: {actual_integral:.8f} (target: {Z})")
        print(f"  Tail average: {final_tail_avg:.8f} (max allowed: {M})")
        print(f"  Constraint satisfied: {final_tail_avg <= M + tolerance}")

    return {
        "p": p_final,
        "lambda": lambda_final,
        "C": C_final,
        "tail_average": final_tail_avg,
        "actual_integral": actual_integral,
        "feasible": final_tail_avg <= M + tolerance,
        "constraint_satisfied": final_tail_avg <= M + tolerance,
        "integral_error": abs(actual_integral - Z),
        "method_used": method,
    }


def verify_solution_thoroughly(W, M, a, Z, result, n_test_points=10000, verbose=True):
    """
    Thoroughly verify that a solution satisfies all constraints using high-precision integration.
    """
    p, lam, C = result["p"], result["lambda"], result["C"]

    if verbose:
        print(f"\n{'='*60}")
        print("THOROUGH VERIFICATION")
        print(f"{'='*60}")
        print(f"Parameters: p={p:.8f}, λ={lam:.8f}, C={C:.8f}")

    # High precision integration for verification
    def integrand_full(x):
        return C * power_exponential_function(x, p, lam)

    def integrand_tail(x):
        return C * power_exponential_function(x, p, lam)

    # Check full integral
    try:
        full_integral, full_error = quad(
            integrand_full, 0, W, limit=500, epsabs=1e-12, epsrel=1e-12
        )
        if verbose:
            print(f"Full integral: {full_integral:.10f} (target: {Z:.10f})")
            print(f"  Error: {abs(full_integral - Z):.2e}")
            print(f"  Integration uncertainty: ±{full_error:.2e}")
        integral_ok = abs(full_integral - Z) < 1e-6
    except Exception as e:
        if verbose:
            print(f"Full integral failed: {e}")
        integral_ok = False
        full_integral = np.inf

    # Check tail average
    try:
        tail_integral, tail_error = quad(
            integrand_tail, W - a, W, limit=500, epsabs=1e-12, epsrel=1e-12
        )
        tail_average = tail_integral / a
        if verbose:
            print(f"Tail average: {tail_average:.10f} (max allowed: {M:.10f})")
            print(f"  Constraint satisfied: {tail_average <= M}")
            print(f"  Integration uncertainty: ±{tail_error/a:.2e}")
        constraint_ok = tail_average <= M + 1e-8
    except Exception as e:
        if verbose:
            print(f"Tail integral failed: {e}")
        constraint_ok = False
        tail_average = np.inf

    # Check monotonicity by sampling derivatives
    x_test = np.linspace(1e-6, W, n_test_points)
    y_test = C * power_exponential_function(x_test, p, lam)
    dy_dx = np.diff(y_test) / np.diff(x_test)

    monotonic = np.all(dy_dx >= -1e-10)  # Allow tiny numerical errors
    if verbose:
        print(f"Monotonicity check: {'✓' if monotonic else '✗'}")
        if not monotonic:
            negative_derivatives = np.sum(dy_dx < -1e-10)
            print(f"  {negative_derivatives} negative derivatives found")

    # Function behavior analysis
    f_0 = C * power_exponential_function(0, p, lam)
    f_W = C * power_exponential_function(W, p, lam)
    if verbose:
        print(f"Function values: f(0)={f_0:.8f}, f(W)={f_W:.8f}")

    overall_ok = integral_ok and constraint_ok and monotonic
    if verbose:
        print(f"\nOVERALL RESULT: {'✓ VALID' if overall_ok else '✗ INVALID'}")
        print(f"{'='*60}")

    return {
        "integral_ok": integral_ok,
        "constraint_ok": constraint_ok,
        "monotonic": monotonic,
        "overall_valid": overall_ok,
        "verified_tail_average": tail_average,
        "verified_integral": full_integral,
    }


def find_maximum_feasible_p(W, M, a, Z, max_p=10.0, tolerance=1e-8, verbose=False):
    """
    Find the maximum value of p for which a feasible solution exists.
    This helps understand the limits of the parameter space.
    """
    if verbose:
        print(f"Finding maximum feasible p value...")

    def is_p_feasible(p_test):
        """Check if any lambda exists for this p that satisfies constraints"""
        try:
            result = solve_power_exponential_parameters(
                W,
                M,
                a,
                Z,
                p_range=(p_test - 0.001, p_test + 0.001),
                lambda_range=(0.001, 5.0),
                method="constraint_based",
                n_trials=1,
                verbose=False,
            )
            return result["feasible"]
        except:
            return False

    # Binary search for maximum feasible p
    p_low = 0.0
    p_high = max_p

    # First check if max_p is feasible
    if not is_p_feasible(p_high):
        # Find upper bound
        while p_high > 0.1 and not is_p_feasible(p_high):
            p_high /= 2

        if p_high <= 0.1:
            if verbose:
                print("Warning: Even small p values may not be feasible")
            return 0.0

    # Binary search
    for _ in range(30):
        p_mid = (p_low + p_high) / 2

        if is_p_feasible(p_mid):
            p_low = p_mid
        else:
            p_high = p_mid

        if abs(p_high - p_low) < tolerance:
            break

    max_feasible_p = p_low
    if verbose:
        print(f"Maximum feasible p ≈ {max_feasible_p:.6f}")

    return max_feasible_p


# ===========================================================
# =                    SPECIFIC MIXING STUFF                =
# ===========================================================


def process_config_file(config_file):
    config_dict = json.load(open(config_file))
    return process_config_dict(config_dict)


def process_config_dict(config_dict):
    d = copy.deepcopy(config_dict)
    topic_counts = {}
    for item in d["pool"]:
        q, t, count = item[:3]
        topic_counts[t] = topic_counts.get(t, 0) + count
    d["topic_counts"] = topic_counts
    d["target_topic_tokens"] = {
        k: round(v * config_dict["target_tokens"]) for k, v in d["pstar"].items()
    }
    return d


def solve_fit(config_dict, topic, verbose=False):
    """Outputs a dict like:
    {
      upsample_ratio: {0: 2.0, ..., quality : upsample_ratio},
      params: {p: 0.42, lambda: 9.0, C: 10.0}
      token_yields: {0: 2348, ..., quality : output_token_count}
    }
    """
    upsample_ratio = {}
    token_yields = {}

    # Get quantiles normalized from [0.0, 1.0] (cdf style)
    original_pool = sorted(
        [_ for _ in config_dict["pool"] if _[1] == topic], key=lambda p: p[0]
    )
    quality_lookup = {_[0]: _[2] for _ in original_pool}
    upsample_boundaries = [0]
    for item in original_pool:
        tok = item[2]
        upsample_boundaries.append(upsample_boundaries[-1] + tok)
    total_tokens = upsample_boundaries[-1]
    quantiles = [_ / total_tokens for _ in upsample_boundaries]
    # Compute the buckets that get fully excluded
    throwaway_strict_upper_bound = None
    for i, el in enumerate(quantiles):
        if el > config_dict.get("throwaway", 0.0):
            throwaway_strict_upper_bound = i - 1
            break
    if throwaway_strict_upper_bound > 0:
        for idx in range(throwaway_strict_upper_bound):
            bucket_idx = original_pool[idx][0]
            upsample_ratio[bucket_idx] = 0.0
            token_yields[bucket_idx] = 0

    # Fit the curve
    rshift = quantiles[throwaway_strict_upper_bound]
    final_bucket_size = quantiles[-1] - quantiles[-2]
    target_ratio = (
        config_dict["target_topic_tokens"][topic] / config_dict["topic_counts"][topic]
    )

    function_params = solve_power_exponential_parameters(
        1 - rshift,
        config_dict["max_upsample"],
        final_bucket_size,
        target_ratio,
        p_range=[0.01, 2.0],
        lambda_range=[0.0, 2.0],
        verbose=verbose,
    )
    for i in range(throwaway_strict_upper_bound, len(quantiles) - 1):
        int_range = [quantiles[i] - rshift, quantiles[i + 1] - rshift]
        quality_bucket = original_pool[i][0]
        avg_integral = compute_power_exp_average(
            function_params["p"],
            function_params["lambda"],
            function_params["C"],
            int_range[0],
            int_range[1],
        )
        upsample_ratio[quality_bucket] = avg_integral
        token_yields[quality_bucket] = round(
            avg_integral * quality_lookup[quality_bucket]
        )
    return {
        "upsample_ratio": upsample_ratio,
        "params": function_params,
        "token_yields": token_yields,
    }


def process_all(config_dict, verbose=False):
    topic_results = {}
    for topic in tqdm(config_dict["pstar"], desc="Topics"):
        print(topic)
        if verbose:
            print("Working on topic: %s" % topic)
        topic_results[topic] = solve_fit(config_dict, topic, verbose=verbose)
    config_dict["topic_results"] = topic_results
    return config_dict


def plot_curves(output_config, topic, alpha=0.5):
    """
    Create a 2D plot with vertical lines, horizontal reference lines, and data plots.

    Parameters:
    -----------
    x_data : list or array
        X coordinates for the function plot
    y_data : list or array
        Y coordinates for the function plot
    vertical_line_positions : list
        X positions where vertical dotted lines should be drawn
    alpha : float, optional
        Transparency of vertical lines (0-1), default is 0.5
    """

    # PROCESS THINGS
    original_pool = sorted(
        [_ for _ in output_config["pool"] if _[1] == topic], key=lambda p: p[0]
    )
    quality_lookup = {_[0]: _[2] for _ in output_config}
    upsample_boundaries = [0]
    for item in original_pool:
        tok = item[2]
        upsample_boundaries.append(upsample_boundaries[-1] + tok)
    total_tokens = upsample_boundaries[-1]
    quantiles = [_ / total_tokens for _ in upsample_boundaries]
    throwaway_strict_upper_bound = None
    for i, el in enumerate(quantiles):
        if el > output_config.get("throwaway", 0.0):
            throwaway_strict_upper_bound = i - 1
            break
    rshift = quantiles[throwaway_strict_upper_bound]
    fparams = output_config["topic_results"][topic]["params"]
    print(fparams)
    curve_data_x = np.arange(0, 101, 1) / 100.0 * (1 - rshift)
    curve_data_y = fparams["C"] * power_exponential_function(
        curve_data_x, fparams["p"], fparams["lambda"]
    )
    curve_data_x += rshift

    upsample_ratios = [
        _[1]
        for _ in sorted(
            output_config["topic_results"][topic]["upsample_ratio"].items(),
            key=lambda p: p[0],
        )
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Set x-axis range
    ax.set_xlim(0, 1.0)

    for q in quantiles:
        ax.axvline(x=q, color="blue", linestyle=":", alpha=alpha, linewidth=1.5)

    # Plot dotted red horizontal lines at y=1.0 and y=7.0
    ax.axhline(
        y=1.0, color="red", linestyle=":", linewidth=1.5, label="Natural distribution"
    )
    ax.axhline(
        y=output_config["max_upsample"],
        color="red",
        linestyle=":",
        linewidth=1.5,
        label="Max upsample",
    )

    # Plot the function (smooth line)
    ax.plot(curve_data_x, curve_data_y, color="blue", linewidth=2, label="Smooth curve")

    # Plot piecewise constant function (step plot)
    ax.step(
        quantiles,
        upsample_ratios + [upsample_ratios[-1]],
        color="green",
        linewidth=2,
        where="post",
        label="Upsampled",
        alpha=0.7,
    )

    # Labels and grid
    ax.set_xlabel("Data quality", fontsize=12)
    ax.set_ylabel("Upsampling", fontsize=12)
    ax.set_title("Upsampling curve: %s" % topic, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def verify_output(output_config):
    # For each topic: check that we got the right number of tokens and didn't upsample too hard
    M = output_config["max_upsample"]
    print("Checking topics...")
    for topic in output_config["pstar"].keys():
        target_tokens = output_config["target_topic_tokens"][topic]
        actual_tokens = sum(
            output_config["topic_results"][topic]["token_yields"].values()
        )
        assert (
            abs(target_tokens - actual_tokens) <= 10
        ), "Incorrect token yields for topic: %s| Wanted %s | Got %s" % (
            topic,
            target_tokens,
            actual_tokens,
        )
        max_topic_upsample = max(
            output_config["topic_results"][topic]["upsample_ratio"].values()
        )
        assert (
            max_topic_upsample <= M + 0.01
        ), "Oversampled on topic: %s | Upsampled a bucket %s times" % (
            topic,
            max_topic_upsample,
        )
        print("\tTopic %s passed checks" % topic)
    print("All topics passed checks!")


def make_good_yaml(output_config_info, output_yaml=None, total_tokens=None):
    # Set total_tokens to be the desired number of tokens (i.e., if there are other sources. else we infer from the token yield results)
    assert all(
        len(item) == 4 for item in output_config_info["pool"]
    ), "Need s3 paths for all tokens to make yaml config!"
    if total_tokens == None:
        total_tokens = 0
        for topic, result in output_config_info["topic_results"].items():
            for v in result["token_yields"].values():
                total_tokens += v

    path_lookup = {}
    to_yaml = []
    for q, t, _, p in output_config_info["pool"]:
        token_yield = output_config_info["topic_results"][t]["token_yields"][q]
        upsample = output_config_info["topic_results"][t]["upsample_ratio"][q]
        if token_yield == 0:
            continue
        item = {
            "name": "%s %s" % (t, q),
            "target_ratio": token_yield / total_tokens,
            "max_repetition": max(1.0, upsample)
            + 1e-2,  # TODO: @tylerm, what's the right field here?
            "paths": [p],
        }
        to_yaml.append(item)

    yaml_dict = {"dataset": {"sources": to_yaml}}
    if output_yaml != None:
        os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
        yaml.dump(yaml_dict, open(output_yaml, "w"))
    else:
        return yaml_dict
