def region_weights(frequencies):
    return {i: min(3.0, 1.0 / max(float(f), 1e-12) ** 0.5) for (i, f) in enumerate(frequencies)}
