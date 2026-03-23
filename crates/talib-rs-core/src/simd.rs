//! SIMD 加速的批量运算函数
//!
//! 使用 `wide` crate 的 `f64x4` 类型实现 4 路并行计算。
//! 所有函数在 `simd` feature 关闭时会退化为标量实现。

#[cfg(feature = "simd")]
use wide::f64x4;

// ============================================================
// 批量求和 (用于 SMA 初始窗口、STDDEV 等)
// ============================================================

/// SIMD 加速的数组求和
#[cfg(feature = "simd")]
pub fn sum_f64(data: &[f64]) -> f64 {
    let len = data.len();
    let chunks = len / 4;
    let remainder = len % 4;

    let mut acc = f64x4::ZERO;
    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        acc += v;
    }

    let mut total = acc.reduce_add();

    // 处理剩余元素
    let tail_start = chunks * 4;
    for i in tail_start..len {
        total += data[i];
    }

    total
}

#[cfg(not(feature = "simd"))]
pub fn sum_f64(data: &[f64]) -> f64 {
    data.iter().sum()
}

// ============================================================
// 批量平方和 (用于 STDDEV, VAR)
// ============================================================

/// SIMD 加速的平方和: Σ(x - mean)²
#[cfg(feature = "simd")]
pub fn sum_sq_diff(data: &[f64], mean: f64) -> f64 {
    let len = data.len();
    let chunks = len / 4;
    let mean_v = f64x4::splat(mean);

    let mut acc = f64x4::ZERO;
    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let diff = v - mean_v;
        acc = diff.mul_add(diff, acc); // acc += diff * diff (fused)
    }

    let mut total = acc.reduce_add();

    let tail_start = chunks * 4;
    for i in tail_start..len {
        let diff = data[i] - mean;
        total += diff * diff;
    }

    total
}

#[cfg(not(feature = "simd"))]
pub fn sum_sq_diff(data: &[f64], mean: f64) -> f64 {
    data.iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum()
}

// ============================================================
// 批量逐元素运算 (用于 Math Operators)
// ============================================================

/// SIMD 加速的逐元素加法
#[cfg(feature = "simd")]
pub fn add_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let vb = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let vr = va + vb;
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = a[i] + b[i];
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn add_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// SIMD 加速的逐元素减法
#[cfg(feature = "simd")]
pub fn sub_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let vb = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let vr = va - vb;
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = a[i] - b[i];
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn sub_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// SIMD 加速的逐元素乘法
#[cfg(feature = "simd")]
pub fn mult_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let vb = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let vr = va * vb;
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = a[i] * b[i];
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn mult_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// SIMD 加速的逐元素除法
#[cfg(feature = "simd")]
pub fn div_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let vb = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let vr = va / vb;
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = a[i] / b[i];
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn div_arrays(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

// ============================================================
// 批量逐元素一元运算 (用于 Math Transform)
// ============================================================

/// SIMD 加速的逐元素 sqrt
#[cfg(feature = "simd")]
pub fn sqrt_array(input: &[f64]) -> Vec<f64> {
    let len = input.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            input[offset],
            input[offset + 1],
            input[offset + 2],
            input[offset + 3],
        ]);
        let vr = v.sqrt();
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = input[i].sqrt();
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn sqrt_array(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&v| v.sqrt()).collect()
}

/// SIMD 加速的逐元素 abs
#[cfg(feature = "simd")]
pub fn abs_array(input: &[f64]) -> Vec<f64> {
    let len = input.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            input[offset],
            input[offset + 1],
            input[offset + 2],
            input[offset + 3],
        ]);
        let vr = v.abs();
        let arr = vr.to_array();
        result[offset] = arr[0];
        result[offset + 1] = arr[1];
        result[offset + 2] = arr[2];
        result[offset + 3] = arr[3];
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        result[i] = input[i].abs();
    }

    result
}

#[cfg(not(feature = "simd"))]
pub fn abs_array(input: &[f64]) -> Vec<f64> {
    input.iter().map(|&v| v.abs()).collect()
}

/// 通用的 SIMD 逐元素标量运算宏
/// 对 sin/cos/exp/ln 等无法直接 SIMD 的函数，使用 4 路并行展开
macro_rules! simd_unary_scalar {
    ($name:ident, $op:expr) => {
        pub fn $name(input: &[f64]) -> Vec<f64> {
            let len = input.len();
            let mut result = vec![0.0; len];

            // 4 路循环展开 (即使不是真 SIMD，也利于 CPU 流水线和缓存)
            let chunks = len / 4;
            for i in 0..chunks {
                let offset = i * 4;
                result[offset] = $op(input[offset]);
                result[offset + 1] = $op(input[offset + 1]);
                result[offset + 2] = $op(input[offset + 2]);
                result[offset + 3] = $op(input[offset + 3]);
            }

            let tail_start = chunks * 4;
            for i in tail_start..len {
                result[i] = $op(input[i]);
            }

            result
        }
    };
}

simd_unary_scalar!(sin_array, f64::sin);
simd_unary_scalar!(cos_array, f64::cos);
simd_unary_scalar!(tan_array, f64::tan);
simd_unary_scalar!(asin_array, f64::asin);
simd_unary_scalar!(acos_array, f64::acos);
simd_unary_scalar!(atan_array, f64::atan);
simd_unary_scalar!(sinh_array, f64::sinh);
simd_unary_scalar!(cosh_array, f64::cosh);
simd_unary_scalar!(tanh_array, f64::tanh);
simd_unary_scalar!(exp_array, f64::exp);
simd_unary_scalar!(ln_array, f64::ln);
simd_unary_scalar!(log10_array, f64::log10);
simd_unary_scalar!(ceil_array, f64::ceil);
simd_unary_scalar!(floor_array, f64::floor);

// ============================================================
// SIMD window max/min scan (used by MAX, MIN, MINMAX, AROON, WILLR, MIDPOINT rescans)
// ============================================================

/// SIMD-accelerated max of a slice, returning (max_value, relative_index_of_max).
/// `data` must not be empty.
#[cfg(feature = "simd")]
pub fn slice_max_with_index(data: &[f64]) -> (f64, usize) {
    let len = data.len();
    debug_assert!(len > 0);
    let chunks = len / 4;

    if chunks == 0 {
        // Pure scalar for tiny slices
        let mut best = data[0];
        let mut best_idx = 0;
        for i in 1..len {
            if data[i] >= best {
                best = data[i];
                best_idx = i;
            }
        }
        return (best, best_idx);
    }

    // Phase 1: SIMD scan to find the max value
    let mut acc = f64x4::new([data[0], data[1], data[2], data[3]]);
    for i in 1..chunks {
        let offset = i * 4;
        let v = f64x4::new([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        acc = acc.max(v);
    }
    let arr = acc.to_array();
    let mut best = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);

    // Handle tail
    let tail_start = chunks * 4;
    for i in tail_start..len {
        if data[i] > best {
            best = data[i];
        }
    }

    // Phase 2: Find the LAST index with >= best (matching TA-Lib's >= semantics)
    let mut best_idx = 0;
    for i in 0..len {
        if data[i] >= best {
            best_idx = i;
        }
    }

    (best, best_idx)
}

#[cfg(not(feature = "simd"))]
pub fn slice_max_with_index(data: &[f64]) -> (f64, usize) {
    let mut best = data[0];
    let mut best_idx = 0;
    for i in 1..data.len() {
        if data[i] >= best {
            best = data[i];
            best_idx = i;
        }
    }
    (best, best_idx)
}

/// SIMD-accelerated min of a slice, returning (min_value, relative_index_of_min).
/// `data` must not be empty.
#[cfg(feature = "simd")]
pub fn slice_min_with_index(data: &[f64]) -> (f64, usize) {
    let len = data.len();
    debug_assert!(len > 0);
    let chunks = len / 4;

    if chunks == 0 {
        let mut best = data[0];
        let mut best_idx = 0;
        for i in 1..len {
            if data[i] <= best {
                best = data[i];
                best_idx = i;
            }
        }
        return (best, best_idx);
    }

    // Phase 1: SIMD scan to find the min value
    let mut acc = f64x4::new([data[0], data[1], data[2], data[3]]);
    for i in 1..chunks {
        let offset = i * 4;
        let v = f64x4::new([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        acc = acc.min(v);
    }
    let arr = acc.to_array();
    let mut best = arr[0].min(arr[1]).min(arr[2]).min(arr[3]);

    let tail_start = chunks * 4;
    for i in tail_start..len {
        if data[i] < best {
            best = data[i];
        }
    }

    // Phase 2: Find the LAST index with <= best
    let mut best_idx = 0;
    for i in 0..len {
        if data[i] <= best {
            best_idx = i;
        }
    }

    (best, best_idx)
}

#[cfg(not(feature = "simd"))]
pub fn slice_min_with_index(data: &[f64]) -> (f64, usize) {
    let mut best = data[0];
    let mut best_idx = 0;
    for i in 1..data.len() {
        if data[i] <= best {
            best = data[i];
            best_idx = i;
        }
    }
    (best, best_idx)
}

// ============================================================
// SIMD element-wise operations for TRANGE, BOP, price transforms
// ============================================================

/// SIMD True Range: output[i] = max(h[i]-l[i], |h[i]-prev_c[i-1]|, |l[i]-prev_c[i-1]|)
/// Processes elements from index `start` to `len-1`, writing to `output`.
#[cfg(feature = "simd")]
pub fn true_range_simd(high: &[f64], low: &[f64], close: &[f64], output: &mut [f64], start: usize) {
    let len = high.len();
    let count = len - start;
    let chunks = count / 4;
    let tail_start = start + chunks * 4;

    for i in 0..chunks {
        let base = start + i * 4;
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vpc = f64x4::new([close[base - 1], close[base], close[base + 1], close[base + 2]]);

        let hl = vh - vl;
        let hc = (vh - vpc).abs();
        let lc = (vl - vpc).abs();
        let result = hl.max(hc).max(lc);
        let arr = result.to_array();
        output[base] = arr[0];
        output[base + 1] = arr[1];
        output[base + 2] = arr[2];
        output[base + 3] = arr[3];
    }

    for i in tail_start..len {
        let h = high[i];
        let l = low[i];
        let pc = close[i - 1];
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        output[i] = hl.max(hc).max(lc);
    }
}

#[cfg(not(feature = "simd"))]
pub fn true_range_simd(high: &[f64], low: &[f64], close: &[f64], output: &mut [f64], start: usize) {
    let len = high.len();
    for i in start..len {
        let h = high[i];
        let l = low[i];
        let pc = close[i - 1];
        let hl = h - l;
        let hc = (h - pc).abs();
        let lc = (l - pc).abs();
        output[i] = hl.max(hc).max(lc);
    }
}

/// SIMD BOP: output[i] = (close[i]-open[i]) / (high[i]-low[i]), 0 if range==0
#[cfg(feature = "simd")]
pub fn bop_simd(open: &[f64], high: &[f64], low: &[f64], close: &[f64], output: &mut [f64]) {
    let len = open.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let base = i * 4;
        let vo = f64x4::new([open[base], open[base + 1], open[base + 2], open[base + 3]]);
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vc = f64x4::new([close[base], close[base + 1], close[base + 2], close[base + 3]]);

        let num = vc - vo;
        let den = vh - vl;
        // Do the division, then fix up zeros
        let ratio = num / den;
        let arr_den = den.to_array();
        let arr = ratio.to_array();
        output[base] = if arr_den[0] > 0.0 { arr[0] } else { 0.0 };
        output[base + 1] = if arr_den[1] > 0.0 { arr[1] } else { 0.0 };
        output[base + 2] = if arr_den[2] > 0.0 { arr[2] } else { 0.0 };
        output[base + 3] = if arr_den[3] > 0.0 { arr[3] } else { 0.0 };
    }

    let tail_start = chunks * 4;
    for i in tail_start..len {
        let range = high[i] - low[i];
        output[i] = if range > 0.0 { (close[i] - open[i]) / range } else { 0.0 };
    }
}

#[cfg(not(feature = "simd"))]
pub fn bop_simd(open: &[f64], high: &[f64], low: &[f64], close: &[f64], output: &mut [f64]) {
    for i in 0..open.len() {
        let range = high[i] - low[i];
        output[i] = if range > 0.0 { (close[i] - open[i]) / range } else { 0.0 };
    }
}

/// SIMD AVGPRICE: (O + H + L + C) / 4
#[cfg(feature = "simd")]
pub fn avgprice_simd(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = open.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;
    let quarter = f64x4::splat(0.25);

    for i in 0..chunks {
        let base = i * 4;
        let vo = f64x4::new([open[base], open[base + 1], open[base + 2], open[base + 3]]);
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vc = f64x4::new([close[base], close[base + 1], close[base + 2], close[base + 3]]);
        let vr = (vo + vh + vl + vc) * quarter;
        let arr = vr.to_array();
        result[base] = arr[0];
        result[base + 1] = arr[1];
        result[base + 2] = arr[2];
        result[base + 3] = arr[3];
    }

    let tail = chunks * 4;
    for i in tail..len {
        result[i] = (open[i] + high[i] + low[i] + close[i]) * 0.25;
    }
    result
}

#[cfg(not(feature = "simd"))]
pub fn avgprice_simd(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..open.len()).map(|i| (open[i] + high[i] + low[i] + close[i]) / 4.0).collect()
}

/// SIMD MEDPRICE: (H + L) / 2
#[cfg(feature = "simd")]
pub fn medprice_simd(high: &[f64], low: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;
    let half = f64x4::splat(0.5);

    for i in 0..chunks {
        let base = i * 4;
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vr = (vh + vl) * half;
        let arr = vr.to_array();
        result[base] = arr[0];
        result[base + 1] = arr[1];
        result[base + 2] = arr[2];
        result[base + 3] = arr[3];
    }

    let tail = chunks * 4;
    for i in tail..len {
        result[i] = (high[i] + low[i]) * 0.5;
    }
    result
}

#[cfg(not(feature = "simd"))]
pub fn medprice_simd(high: &[f64], low: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| (high[i] + low[i]) / 2.0).collect()
}

/// SIMD TYPPRICE: (H + L + C) / 3
#[cfg(feature = "simd")]
pub fn typprice_simd(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;
    let third = f64x4::splat(1.0 / 3.0);

    for i in 0..chunks {
        let base = i * 4;
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vc = f64x4::new([close[base], close[base + 1], close[base + 2], close[base + 3]]);
        let vr = (vh + vl + vc) * third;
        let arr = vr.to_array();
        result[base] = arr[0];
        result[base + 1] = arr[1];
        result[base + 2] = arr[2];
        result[base + 3] = arr[3];
    }

    let tail = chunks * 4;
    for i in tail..len {
        result[i] = (high[i] + low[i] + close[i]) / 3.0;
    }
    result
}

#[cfg(not(feature = "simd"))]
pub fn typprice_simd(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| (high[i] + low[i] + close[i]) / 3.0).collect()
}

/// SIMD WCLPRICE: (H + L + 2*C) / 4
#[cfg(feature = "simd")]
pub fn wclprice_simd(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut result = vec![0.0; len];
    let chunks = len / 4;
    let two = f64x4::splat(2.0);
    let quarter = f64x4::splat(0.25);

    for i in 0..chunks {
        let base = i * 4;
        let vh = f64x4::new([high[base], high[base + 1], high[base + 2], high[base + 3]]);
        let vl = f64x4::new([low[base], low[base + 1], low[base + 2], low[base + 3]]);
        let vc = f64x4::new([close[base], close[base + 1], close[base + 2], close[base + 3]]);
        let vr = (vh + vl + two * vc) * quarter;
        let arr = vr.to_array();
        result[base] = arr[0];
        result[base + 1] = arr[1];
        result[base + 2] = arr[2];
        result[base + 3] = arr[3];
    }

    let tail = chunks * 4;
    for i in tail..len {
        result[i] = (high[i] + low[i] + 2.0 * close[i]) * 0.25;
    }
    result
}

#[cfg(not(feature = "simd"))]
pub fn wclprice_simd(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    (0..high.len()).map(|i| (high[i] + low[i] + 2.0 * close[i]) / 4.0).collect()
}

/// SIMD offset subtraction: output[i] = a[i] - a[i-offset] for i in [offset..len)
/// output[0..offset] is untouched by caller (set to NaN).
#[cfg(feature = "simd")]
pub fn sub_offset_simd(input: &[f64], output: &mut [f64], offset: usize) {
    let len = input.len();
    let count = len - offset;
    let chunks = count / 4;

    for i in 0..chunks {
        let base = offset + i * 4;
        let va = f64x4::new([input[base], input[base + 1], input[base + 2], input[base + 3]]);
        let vb = f64x4::new([input[base - offset], input[base - offset + 1], input[base - offset + 2], input[base - offset + 3]]);
        let vr = va - vb;
        let arr = vr.to_array();
        output[base] = arr[0];
        output[base + 1] = arr[1];
        output[base + 2] = arr[2];
        output[base + 3] = arr[3];
    }

    let tail = offset + chunks * 4;
    for i in tail..len {
        output[i] = input[i] - input[i - offset];
    }
}

#[cfg(not(feature = "simd"))]
pub fn sub_offset_simd(input: &[f64], output: &mut [f64], offset: usize) {
    for i in offset..input.len() {
        output[i] = input[i] - input[i - offset];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_f64() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = sum_f64(&data);
        assert!((result - 5050.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_f64_non_aligned() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]; // 7 元素
        let result = sum_f64(&data);
        assert!((result - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_sq_diff() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let result = sum_sq_diff(&data, mean);
        // (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 4+1+0+1+4 = 10
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_arrays() {
        let a: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let b: Vec<f64> = (11..=20).map(|x| x as f64).collect();
        let result = add_arrays(&a, &b);
        for i in 0..10 {
            assert!((result[i] - (a[i] + b[i])).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sqrt_array() {
        let input = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let result = sqrt_array(&input);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
        assert!((result[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_array() {
        let input = vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
        let result = sin_array(&input);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
    }
}
