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
