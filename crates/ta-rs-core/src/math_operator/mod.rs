use crate::error::{TaError, TaResult};

/// 逐元素加法 (编译器自动向量化，比手动 SIMD 更快)
pub fn add(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0.iter().zip(input1.iter()).map(|(a, b)| a + b).collect())
}

/// 逐元素减法
pub fn sub(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0.iter().zip(input1.iter()).map(|(a, b)| a - b).collect())
}

/// 逐元素乘法
pub fn mult(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0.iter().zip(input1.iter()).map(|(a, b)| a * b).collect())
}

/// 逐元素除法
pub fn div(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0.iter().zip(input1.iter()).map(|(a, b)| a / b).collect())
}

/// 滑动窗口最大值
pub fn max(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    sliding_extreme(input, timeperiod, true)
}

/// 滑动窗口最大值的索引
pub fn maxindex(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    sliding_extreme_index(input, timeperiod, true)
}

/// 滑动窗口最小值
pub fn min(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    sliding_extreme(input, timeperiod, false)
}

/// 滑动窗口最小值的索引
pub fn minindex(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    sliding_extreme_index(input, timeperiod, false)
}

/// 滑动窗口求和
pub fn sum(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: "0".to_string(), reason: "must be >= 1",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod, got: len });
    }
    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let mut s: f64 = input[..timeperiod].iter().sum();
    output[lookback] = s;
    for i in timeperiod..len {
        s += input[i] - input[i - timeperiod];
        output[i] = s;
    }
    Ok(output)
}

fn validate_pair(a: &[f64], b: &[f64]) -> TaResult<()> {
    if a.len() != b.len() {
        return Err(TaError::LengthMismatch { expected: a.len(), got: b.len() });
    }
    Ok(())
}

fn sliding_extreme(input: &[f64], timeperiod: usize, is_max: bool) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: "0".to_string(), reason: "must be >= 1",
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
        let val = if is_max {
            input[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            input[start..=i].iter().cloned().fold(f64::INFINITY, f64::min)
        };
        output[i] = val;
    }
    Ok(output)
}

fn sliding_extreme_index(input: &[f64], timeperiod: usize, is_max: bool) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: "0".to_string(), reason: "must be >= 1",
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
        let mut extreme_idx = start;
        for j in (start + 1)..=i {
            if is_max {
                if input[j] >= input[extreme_idx] { extreme_idx = j; }
            } else {
                if input[j] <= input[extreme_idx] { extreme_idx = j; }
            }
        }
        output[i] = extreme_idx as f64;
    }
    Ok(output)
}
