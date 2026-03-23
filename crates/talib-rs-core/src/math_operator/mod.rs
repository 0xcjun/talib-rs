use crate::error::{TaError, TaResult};

/// 逐元素加法 (编译器自动向量化，比手动 SIMD 更快)
pub fn add(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0
        .iter()
        .zip(input1.iter())
        .map(|(a, b)| a + b)
        .collect())
}

/// 逐元素减法
pub fn sub(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0
        .iter()
        .zip(input1.iter())
        .map(|(a, b)| a - b)
        .collect())
}

/// 逐元素乘法
pub fn mult(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0
        .iter()
        .zip(input1.iter())
        .map(|(a, b)| a * b)
        .collect())
}

/// 逐元素除法
pub fn div(input0: &[f64], input1: &[f64]) -> TaResult<Vec<f64>> {
    validate_pair(input0, input1)?;
    Ok(input0
        .iter()
        .zip(input1.iter())
        .map(|(a, b)| a / b)
        .collect())
}

/// 滑动窗口最大值 -- brute rescan with slice iteration (amortized O(n))
pub fn max(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let mut highest = input[0];
    let mut highest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] >= highest {
            highest = input[j];
            highest_idx = j;
        }
    }
    output[lookback] = highest;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];
        if highest_idx < trailing_idx {
            // Rescan window using slice iterator (bounds-check-free)
            highest = input[trailing_idx];
            highest_idx = trailing_idx;
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val >= highest {
                    highest = val;
                    highest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }
        output[today] = highest;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

/// 滑动窗口最大值的索引 -- brute rescan with slice iteration (amortized O(n))
pub fn maxindex(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let mut highest = input[0];
    let mut highest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] >= highest {
            highest = input[j];
            highest_idx = j;
        }
    }
    output[lookback] = highest_idx as f64;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];
        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = input[trailing_idx];
            // Rescan window using slice iterator (bounds-check-free)
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val >= highest {
                    highest = val;
                    highest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }
        output[today] = highest_idx as f64;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

/// 滑动窗口最小值 -- brute rescan with slice iteration (amortized O(n))
pub fn min(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let mut lowest = input[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] <= lowest {
            lowest = input[j];
            lowest_idx = j;
        }
    }
    output[lookback] = lowest;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];
        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            // Rescan window using slice iterator (bounds-check-free)
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val <= lowest {
                    lowest = val;
                    lowest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }
        output[today] = lowest;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

/// 滑动窗口最小值的索引 -- brute rescan with slice iteration (amortized O(n))
pub fn minindex(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    let mut lowest = input[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] <= lowest {
            lowest = input[j];
            lowest_idx = j;
        }
    }
    output[lookback] = lowest_idx as f64;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];
        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            // Rescan window using slice iterator (bounds-check-free)
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val <= lowest {
                    lowest = val;
                    lowest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }
        output[today] = lowest_idx as f64;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

/// 滑动窗口求和
pub fn sum(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);
    let mut s: f64 = input[..timeperiod].iter().sum();
    output[lookback] = s;
    for i in timeperiod..len {
        s += input[i] - input[i - timeperiod];
        output[i] = s;
    }
    Ok(output)
}

/// MINMAX -- fused single-pass: find both max and min in one scan.
///
/// Reads input once (vs two-pass reading twice). Saves ~8MB memory bandwidth at 1M bars.
pub fn minmax(input: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut out_min = vec![0.0_f64; len];
    out_min[..lookback].fill(f64::NAN);
    let mut out_max = vec![0.0_f64; len];
    out_max[..lookback].fill(f64::NAN);

    let mut highest = input[0];
    let mut highest_idx: usize = 0;
    let mut lowest = input[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] >= highest { highest = input[j]; highest_idx = j; }
        if input[j] <= lowest { lowest = input[j]; lowest_idx = j; }
    }
    out_max[lookback] = highest;
    out_min[lookback] = lowest;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];

        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = input[trailing_idx];
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val >= highest { highest = val; highest_idx = trailing_idx + 1 + j; }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }

        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val <= lowest { lowest = val; lowest_idx = trailing_idx + 1 + j; }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }

        out_max[today] = highest;
        out_min[today] = lowest;
        trailing_idx += 1;
        today += 1;
    }

    Ok((out_min, out_max))
}

/// MINMAXINDEX -- fused single-pass
pub fn minmaxindex(input: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    validate_period(input, timeperiod)?;
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut out_minidx = vec![0.0_f64; len];
    out_minidx[..lookback].fill(f64::NAN);
    let mut out_maxidx = vec![0.0_f64; len];
    out_maxidx[..lookback].fill(f64::NAN);

    let mut highest = input[0];
    let mut highest_idx: usize = 0;
    let mut lowest = input[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] >= highest { highest = input[j]; highest_idx = j; }
        if input[j] <= lowest { lowest = input[j]; lowest_idx = j; }
    }
    out_maxidx[lookback] = highest_idx as f64;
    out_minidx[lookback] = lowest_idx as f64;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];

        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = input[trailing_idx];
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val >= highest { highest = val; highest_idx = trailing_idx + 1 + j; }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }

        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val <= lowest { lowest = val; lowest_idx = trailing_idx + 1 + j; }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }

        out_maxidx[today] = highest_idx as f64;
        out_minidx[today] = lowest_idx as f64;
        trailing_idx += 1;
        today += 1;
    }

    Ok((out_minidx, out_maxidx))
}

/// MINMAXINDEX -- original fused implementation (unused, kept for reference)
#[allow(dead_code)]
fn minmaxindex_fused(input: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = input.len();
    let lookback = timeperiod - 1;
    let mut out_minidx = vec![0.0_f64; len];
    out_minidx[..lookback].fill(f64::NAN);
    let mut out_maxidx = vec![0.0_f64; len];
    out_maxidx[..lookback].fill(f64::NAN);

    let mut highest = input[0];
    let mut highest_idx: usize = 0;
    let mut lowest = input[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if input[j] >= highest {
            highest = input[j];
            highest_idx = j;
        }
        if input[j] <= lowest {
            lowest = input[j];
            lowest_idx = j;
        }
    }
    out_maxidx[lookback] = highest_idx as f64;
    out_minidx[lookback] = lowest_idx as f64;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];

        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = input[trailing_idx];
            // Rescan window using slice iterator (bounds-check-free)
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val >= highest {
                    highest = val;
                    highest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }

        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            // Rescan window using slice iterator (bounds-check-free)
            for (j, &val) in input[trailing_idx + 1..=today].iter().enumerate() {
                if val <= lowest {
                    lowest = val;
                    lowest_idx = trailing_idx + 1 + j;
                }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }

        out_maxidx[today] = highest_idx as f64;
        out_minidx[today] = lowest_idx as f64;
        trailing_idx += 1;
        today += 1;
    }

    Ok((out_minidx, out_maxidx))
}

fn validate_pair(a: &[f64], b: &[f64]) -> TaResult<()> {
    if a.len() != b.len() {
        return Err(TaError::LengthMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(())
}

fn validate_period(input: &[f64], timeperiod: usize) -> TaResult<()> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    if input.len() < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: input.len(),
        });
    }
    Ok(())
}
