//! The handful of statistics the feature extractor needs, matching
//! numpy / scipy defaults (`np.std` = population, `np.percentile` = linear,
//! `scipy.stats.skew(bias=False)`, `scipy.stats.linregress`).

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// `np.std` (ddof=0)
pub fn std_pop(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// `np.std(ddof=1)`
pub fn std_sample(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0)).sqrt()
}

/// `np.percentile(a, q*100)` with linear interpolation on a sorted slice.
pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor();
    let hi = h.ceil();
    let li = lo as usize;
    let hidx = hi as usize;
    if li == hidx {
        sorted[li]
    } else {
        sorted[li] + (sorted[hidx] - sorted[li]) * (h - lo)
    }
}

/// `scipy.stats.skew(xs, bias=False)`; returns 0.0 where scipy would return nan
/// (which the Python code maps to 0.0 as well).
pub fn skew_unbiased(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n <= 2 {
        return 0.0;
    }
    let nf = n as f64;
    let m = mean(xs);
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for x in xs {
        let d = x - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    m2 /= nf;
    m3 /= nf;
    if m2 <= 0.0 {
        return 0.0;
    }
    let g1 = m3 / m2.powf(1.5);
    let corr = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    let v = g1 * corr;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Pearson correlation (population formula, like `scipy.stats.pearsonr`).
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(&xs[..n]);
    let my = mean(&ys[..n]);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let den = (sxx * syy).sqrt();
    if den <= 0.0 {
        return 0.0;
    }
    let r = sxy / den;
    if r.is_finite() {
        r
    } else {
        0.0
    }
}

/// Slope of an ordinary least-squares fit; `f64::NAN` when undefined
/// (same as `scipy.stats.linregress` on degenerate input).
pub fn linregress_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return f64::NAN;
    }
    let mx = mean(&xs[..n]);
    let my = mean(&ys[..n]);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..n {
        let dx = xs[i] - mx;
        sxy += dx * (ys[i] - my);
        sxx += dx * dx;
    }
    if sxx <= 0.0 {
        return f64::NAN;
    }
    sxy / sxx
}

/// `np.round` rounds halves to even; `f64::round` rounds halves away from zero.
pub fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    if x - f == 0.5 {
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    } else {
        x.round()
    }
}

/// `round(x, digits)` from Python: correct rounding of the exact binary value,
/// ties to even - which is what Rust's float formatting does as well.
pub fn round_digits(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let s = format!("{:.*}", digits, x);
    s.parse::<f64>().unwrap_or(x)
}

/// `10**(a + i * (b - a) / (num - 1))`, i.e. `np.logspace(a, b, num)`.
pub fn logspace(a: f64, b: f64, num: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(num);
    if num == 0 {
        return out;
    }
    if num == 1 {
        out.push(10f64.powf(a));
        return out;
    }
    let step = (b - a) / (num as f64 - 1.0);
    for i in 0..num {
        out.push(10f64.powf(a + i as f64 * step));
    }
    out
}
