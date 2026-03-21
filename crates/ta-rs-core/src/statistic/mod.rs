use crate::error::{TaError, TaResult};

/// Standard Deviation (STDDEV)
pub fn stddev(input: &[f64], timeperiod: usize, nbdev: f64) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: timeperiod.to_string(), reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let window = &input[start..=i];
        let mean = crate::simd::sum_f64(window) / timeperiod as f64;
        let var = crate::simd::sum_sq_diff(window, mean) / timeperiod as f64;
        output[i] = var.sqrt() * nbdev;
    }
    Ok(output)
}

/// Variance (VAR)
pub fn var(input: &[f64], timeperiod: usize, nbdev: f64) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: timeperiod.to_string(), reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let window = &input[start..=i];
        let mean = crate::simd::sum_f64(window) / timeperiod as f64;
        let variance = crate::simd::sum_sq_diff(window, mean) / timeperiod as f64;
        output[i] = variance;
    }
    Ok(output)
}

/// Beta
pub fn beta(input0: &[f64], input1: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = input0.len();
    if len != input1.len() {
        return Err(TaError::LengthMismatch { expected: len, got: input1.len() });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod + 1, got: len });
    }
    let mut output = vec![f64::NAN; len];
    for i in timeperiod..len {
        let start = i + 1 - timeperiod;
        let x_mean: f64 = input0[start..=i].iter().sum::<f64>() / timeperiod as f64;
        let y_mean: f64 = input1[start..=i].iter().sum::<f64>() / timeperiod as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for j in start..=i {
            let dx = input0[j] - x_mean;
            let dy = input1[j] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
        }
        output[i] = if var_x > 0.0 { cov / var_x } else { 0.0 };
    }
    Ok(output)
}

/// Pearson's Correlation Coefficient (CORREL)
pub fn correl(input0: &[f64], input1: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = input0.len();
    if len != input1.len() {
        return Err(TaError::LengthMismatch { expected: len, got: input1.len() });
    }
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let x_mean: f64 = input0[start..=i].iter().sum::<f64>() / timeperiod as f64;
        let y_mean: f64 = input1[start..=i].iter().sum::<f64>() / timeperiod as f64;
        let (mut cov, mut var_x, mut var_y) = (0.0, 0.0, 0.0);
        for j in start..=i {
            let dx = input0[j] - x_mean;
            let dy = input1[j] - y_mean;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        output[i] = if denom > 0.0 { cov / denom } else { 0.0 };
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

/// 内部: 计算线性回归的斜率和截距
fn linearreg_components(input: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: timeperiod.to_string(), reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }

    let mut slope = vec![f64::NAN; len];
    let mut intercept = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let n = timeperiod as f64;
    let sum_x = n * (n - 1.0) / 2.0;
    let sum_x2 = n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;

    for i in lookback..len {
        let start = i + 1 - timeperiod;
        let sum_y: f64 = input[start..=i].iter().sum();
        let mut sum_xy = 0.0;
        for (x, j) in (start..=i).enumerate() {
            sum_xy += x as f64 * input[j];
        }
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom != 0.0 {
            let m = (n * sum_xy - sum_x * sum_y) / denom;
            let b = (sum_y - m * sum_x) / n;
            slope[i] = m;
            intercept[i] = b;
        }
    }

    Ok((slope, intercept))
}
