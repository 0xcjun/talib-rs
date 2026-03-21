/// 数学变换函数 — 对数组每个元素应用数学函数
/// 编译器自动向量化对逐元素操作已经非常高效，无需手动 SIMD。
/// 使用 4 路循环展开帮助 CPU 流水线。

macro_rules! math_transform {
    ($name:ident, $op:expr) => {
        pub fn $name(input: &[f64]) -> Vec<f64> {
            let len = input.len();
            let mut result = vec![0.0; len];
            let chunks = len / 4;
            for i in 0..chunks {
                let o = i * 4;
                result[o] = $op(input[o]);
                result[o + 1] = $op(input[o + 1]);
                result[o + 2] = $op(input[o + 2]);
                result[o + 3] = $op(input[o + 3]);
            }
            let tail = chunks * 4;
            for i in tail..len {
                result[i] = $op(input[i]);
            }
            result
        }
    };
}

math_transform!(acos, f64::acos);
math_transform!(asin, f64::asin);
math_transform!(atan, f64::atan);
math_transform!(ceil, f64::ceil);
math_transform!(cos, f64::cos);
math_transform!(cosh, f64::cosh);
math_transform!(exp, f64::exp);
math_transform!(floor, f64::floor);
math_transform!(ln, f64::ln);
math_transform!(log10, f64::log10);
math_transform!(sin, f64::sin);
math_transform!(sinh, f64::sinh);
math_transform!(sqrt, f64::sqrt);
math_transform!(tan, f64::tan);
math_transform!(tanh, f64::tanh);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt() {
        let input = vec![1.0, 4.0, 9.0, 16.0];
        let result = sqrt(&input);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_cos_identity() {
        let input: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let s = sin(&input);
        let c = cos(&input);
        for i in 0..100 {
            let sum = s[i] * s[i] + c[i] * c[i];
            assert!((sum - 1.0).abs() < 1e-10, "sin²+cos²≠1 at {}", i);
        }
    }
}
