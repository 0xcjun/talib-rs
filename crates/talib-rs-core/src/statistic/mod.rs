use crate::error::{TaError, TaResult};

/// Standard Deviation (STDDEV) — O(n) 滑动窗口算法
///
/// 使用 Var(X) = E(X²) - E(X)² 的在线计算:
/// 维护 sum 和 sum_sq 的滑动窗口，每步 O(1)。
pub fn stddev(input: &[f64], timeperiod: usize, nbdev: f64) -> TaResult<Vec<f64>> {
    let var_result = var_internal(input, timeperiod)?;
    let mut output = vec![f64::NAN; input.len()];
    let lookback = timeperiod - 1;
    for i in lookback..input.len() {
        output[i] = var_result[i].max(0.0).sqrt() * nbdev;
    }
    Ok(output)
}

/// Variance (VAR) — O(n) 滑动窗口算法
pub fn var(input: &[f64], timeperiod: usize, _nbdev: f64) -> TaResult<Vec<f64>> {
    var_internal(input, timeperiod)
}

/// 内部: O(n) 方差计算 (滑动窗口 sum + sum_sq)
fn var_internal(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let n = timeperiod as f64;

    // 初始窗口的 sum 和 sum_sq
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for j in 0..timeperiod {
        sum += input[j];
        sum_sq += input[j] * input[j];
    }
    // Var = E(X²) - E(X)² = sum_sq/n - (sum/n)²
    output[lookback] = sum_sq / n - (sum / n) * (sum / n);

    // O(1) 滑动
    for i in timeperiod..len {
        let old = input[i - timeperiod];
        let new = input[i];
        sum += new - old;
        sum_sq += new * new - old * old;
        output[i] = sum_sq / n - (sum / n) * (sum / n);
    }

    Ok(output)
}

/// Beta — O(n) 滑动窗口算法
///
/// 使用恒等式: beta = (n*sxy - sx*sy) / (n*sxx - sx*sx)
/// 维护 sx, sy, sxx, sxy 四个滑动和，每步 O(1)。
pub fn beta(input0: &[f64], input1: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = input0.len();
    if len != input1.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: input1.len(),
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }
    let mut output = vec![f64::NAN; len];
    let n = timeperiod as f64;

    // 初始化第一个窗口 [1..=timeperiod] 的滑动求和
    let init_x = &input0[1..=timeperiod];
    let init_y = &input1[1..=timeperiod];
    let mut sx = crate::simd::sum_f64(init_x);
    let mut sy = crate::simd::sum_f64(init_y);
    let mut sxx: f64 = init_x.iter().map(|&v| v * v).sum();
    let mut sxy: f64 = init_x.iter().zip(init_y.iter()).map(|(&x, &y)| x * y).sum();

    let denom_x = n * sxx - sx * sx;
    output[timeperiod] = if denom_x > 0.0 {
        (n * sxy - sx * sy) / denom_x
    } else {
        0.0
    };

    // 滑动窗口：每次加入新元素、移除最旧元素
    for i in (timeperiod + 1)..len {
        let old_x = input0[i - timeperiod];
        let old_y = input1[i - timeperiod];
        let new_x = input0[i];
        let new_y = input1[i];

        sx += new_x - old_x;
        sy += new_y - old_y;
        sxx += new_x * new_x - old_x * old_x;
        sxy += new_x * new_y - old_x * old_y;

        let denom_x = n * sxx - sx * sx;
        output[i] = if denom_x > 0.0 {
            (n * sxy - sx * sy) / denom_x
        } else {
            0.0
        };
    }
    Ok(output)
}

/// Pearson's Correlation Coefficient (CORREL) — O(n) 滑动窗口算法
///
/// 使用恒等式: correl = (n*sxy - sx*sy) / sqrt((n*sxx - sx²) * (n*syy - sy²))
/// 维护 sx, sy, sxx, syy, sxy 五个滑动和，每步 O(1)。
pub fn correl(input0: &[f64], input1: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = input0.len();
    if len != input1.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: input1.len(),
        });
    }
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let n = timeperiod as f64;

    // 初始化第一个窗口 [0..timeperiod] 的滑动求和
    let init_x = &input0[0..timeperiod];
    let init_y = &input1[0..timeperiod];
    let mut sx = crate::simd::sum_f64(init_x);
    let mut sy = crate::simd::sum_f64(init_y);
    let mut sxx: f64 = init_x.iter().map(|&v| v * v).sum();
    let mut syy: f64 = init_y.iter().map(|&v| v * v).sum();
    let mut sxy: f64 = init_x.iter().zip(init_y.iter()).map(|(&x, &y)| x * y).sum();

    let num = n * sxy - sx * sy;
    let denom = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt();
    output[lookback] = if denom > 0.0 { num / denom } else { 0.0 };

    // 滑动窗口
    for i in timeperiod..len {
        let old_x = input0[i - timeperiod];
        let old_y = input1[i - timeperiod];
        let new_x = input0[i];
        let new_y = input1[i];

        sx += new_x - old_x;
        sy += new_y - old_y;
        sxx += new_x * new_x - old_x * old_x;
        syy += new_y * new_y - old_y * old_y;
        sxy += new_x * new_y - old_x * old_y;

        let num = n * sxy - sx * sy;
        let denom = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt();
        output[i] = if denom > 0.0 { num / denom } else { 0.0 };
    }
    Ok(output)
}

/// Linear Regression (LINEARREG)
pub fn linearreg(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let (slope, intercept) = linearreg_components(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    for i in lookback..len {
        output[i] = intercept[i] + slope[i] * lookback as f64;
    }
    Ok(output)
}

/// Linear Regression Slope
pub fn linearreg_slope(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let (slope, _) = linearreg_components(input, timeperiod)?;
    Ok(slope)
}

/// Linear Regression Intercept
pub fn linearreg_intercept(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let (_, intercept) = linearreg_components(input, timeperiod)?;
    Ok(intercept)
}

/// Linear Regression Angle (degrees)
pub fn linearreg_angle(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let (slope, _) = linearreg_components(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![f64::NAN; len];
    for i in 0..len {
        if !slope[i].is_nan() {
            output[i] = slope[i].atan().to_degrees();
        }
    }
    Ok(output)
}

/// Time Series Forecast (TSF)
pub fn tsf(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let (slope, intercept) = linearreg_components(input, timeperiod)?;
    let len = input.len();
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    for i in lookback..len {
        output[i] = intercept[i] + slope[i] * timeperiod as f64;
    }
    Ok(output)
}

/// 内部: 计算线性回归的斜率和截距 — O(n) 滑动窗口算法
///
/// 维护两个滑动和: sum_y (值的和) 和 ws (加权和, ws = Σ k*v[k] for k=0..p-1)
/// 当窗口右移一位时:
///   ws_new = ws_old - sum_y_old + period * input[new]
///   sum_y_new = sum_y_old - input[old] + input[new]
/// 其中 sum_x 和 sum_x2 为常量 (仅依赖 period)。
fn linearreg_components(input: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let mut slope = vec![f64::NAN; len];
    let mut intercept = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let n = timeperiod as f64;
    let p = timeperiod;

    // 常量: sum_x = 0+1+...+(p-1), sum_x2 = 0²+1²+...+(p-1)²
    let sum_x = n * (n - 1.0) / 2.0;
    let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
    let denom = n * sum_x2 - sum_x * sum_x;

    // 初始化第一个窗口 [0..timeperiod]
    let init_window = &input[0..p];
    let mut sum_y = crate::simd::sum_f64(init_window);
    // ws = Σ k * input[k], k = 0..p-1 (加权和，权重为窗口内位置索引)
    let mut ws: f64 = init_window
        .iter()
        .enumerate()
        .map(|(k, &v)| k as f64 * v)
        .sum();

    if denom != 0.0 {
        let m = (n * ws - sum_x * sum_y) / denom;
        let b = (sum_y - m * sum_x) / n;
        slope[lookback] = m;
        intercept[lookback] = b;
    }

    // O(1) 滑动: 窗口从 [0..p] 逐步移到 [i-p+1..i+1]
    for i in p..len {
        let old_val = input[i - p]; // 离开窗口的旧元素 (位置索引为0)
        let new_val = input[i]; // 进入窗口的新元素 (位置索引为p-1)
                                // ws 更新推导:
                                //   旧窗口各元素的位置索引各减1 => ws -= sum_y_old
                                //   旧元素 (索引0) 离开 => 需补回减掉的 old_val => ws += old_val
                                //   新元素以索引 (p-1) 进入 => ws += (p-1) * new_val
        ws = ws - sum_y + old_val + (n - 1.0) * new_val;
        // 更新 sum_y: 减去离开的旧值，加上进入的新值
        sum_y = sum_y - old_val + new_val;

        if denom != 0.0 {
            let m = (n * ws - sum_x * sum_y) / denom;
            let b = (sum_y - m * sum_x) / n;
            slope[i] = m;
            intercept[i] = b;
        }
    }

    Ok((slope, intercept))
}
