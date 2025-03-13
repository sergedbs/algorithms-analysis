import time
import numpy as np

# Measure the execution time of a method for a given input
def measure_execution_time(func, n, trials=10, warmup=3):

    try:
        result = func(n)
        for _ in range(warmup):
            func(n)
    except Exception as e:
        return f"ERROR: {str(e)}", -1

    if result == -1:
        return "OVERFLOW", -1

    times = []
    for _ in range(trials):
        try:
            start = time.perf_counter_ns()
            result = func(n)
            elapsed = time.perf_counter_ns() - start

            if result == -1:
                return "OVERFLOW", -1

            times.append(elapsed)
        except Exception as e:
            return f"ERROR: {str(e)}", -1

    # Filter out outliers using modified z-scores and IQR
    if len(times) >= 5:
        # Calculate Modified Z-scores (more robust than standard z-scores)
        median_time = np.median(times)
        mad = np.median([abs(t - median_time) for t in times])
        modified_z_scores = [0.6745 * (t - median_time) / mad if mad else 0 for t in times]

        # Calculate IQR
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1

        # Combined filtering using both methods
        filtered_times = []
        for t, mz in zip(times, modified_z_scores):
            is_within_iqr = (q1 - 1.5 * iqr <= t <= q3 + 1.5 * iqr)
            is_within_z = (abs(mz) <= 3.5)  # More conservative threshold
            if is_within_iqr and is_within_z:
                filtered_times.append(t)

        # Fallback if too many values were filtered
        if len(filtered_times) < max(3, int(len(times) / 3)):
            # Use less stringent criteria
            filtered_times = [t for t in times if q1 - 2.0 * iqr <= t <= q3 + 2.0 * iqr]

        # Final fallback
        if len(filtered_times) < 3:
            filtered_times = times
    else:
        filtered_times = times

    return result, float(np.median(filtered_times))