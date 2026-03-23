use crate::error::{TaError, TaResult};

/// Aroon (AROON) — two-pass cache-friendly architecture
///
/// Pass 1: scan high[] → aroon_up (2-way cache: high + aroon_up)
/// Pass 2: scan low[]  → aroon_down (2-way cache: low + aroon_down)
///
/// Returns (aroon_down, aroon_up)
pub fn aroon(high: &[f64], low: &[f64], timeperiod: usize) -> TaResult<(Vec<f64>, Vec<f64>)> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch { expected: len, got: low.len() });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: timeperiod.to_string(), reason: "must be >= 2",
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod + 1, got: len });
    }

    let inv_period = 100.0 / timeperiod as f64;

    let aroon_up = aroon_max_pass(high, timeperiod, inv_period);
    let aroon_down = aroon_min_pass(low, timeperiod, inv_period);

    Ok((aroon_down, aroon_up))
}

/// Aroon Oscillator — single-pass with pre-multiplied inv_period
///
/// Tracks both highest and lowest in one pass (3-way cache: high + low + output).
/// Uses pre-computed inv_period to eliminate per-bar division.
pub fn aroon_osc(high: &[f64], low: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch { expected: len, got: low.len() });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod", value: timeperiod.to_string(), reason: "must be >= 2",
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData { need: timeperiod + 1, got: len });
    }

    let inv_period = 100.0 / timeperiod as f64;
    let window = timeperiod + 1;
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    let mut highest = high[0];
    let mut highest_idx: usize = 0;
    let mut lowest = low[0];
    let mut lowest_idx: usize = 0;
    for j in 1..window {
        if high[j] >= highest { highest = high[j]; highest_idx = j; }
        if low[j] <= lowest { lowest = low[j]; lowest_idx = j; }
    }
    {
        let up = (timeperiod - (timeperiod - highest_idx)) as f64 * inv_period;
        let down = (timeperiod - (timeperiod - lowest_idx)) as f64 * inv_period;
        output[timeperiod] = up - down;
    }

    let mut trailing_idx = 1;
    let mut today = timeperiod + 1;

    while today < len {
        let h = high[today];
        let l = low[today];

        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = high[trailing_idx];
            for (j, &val) in high[trailing_idx + 1..today + 1].iter().enumerate() {
                if val >= highest { highest = val; highest_idx = trailing_idx + 1 + j; }
            }
        } else if h >= highest {
            highest_idx = today;
            highest = h;
        }

        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = low[trailing_idx];
            for (j, &val) in low[trailing_idx + 1..today + 1].iter().enumerate() {
                if val <= lowest { lowest = val; lowest_idx = trailing_idx + 1 + j; }
            }
        } else if l <= lowest {
            lowest_idx = today;
            lowest = l;
        }

        let up = (timeperiod - (today - highest_idx)) as f64 * inv_period;
        let down = (timeperiod - (today - lowest_idx)) as f64 * inv_period;
        output[today] = up - down;
        trailing_idx += 1;
        today += 1;
    }

    Ok(output)
}

/// Single-pass max tracker for aroon_up (2-way cache).
#[inline]
fn aroon_max_pass(data: &[f64], timeperiod: usize, inv_period: f64) -> Vec<f64> {
    let len = data.len();
    let window = timeperiod + 1;
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    let mut highest = data[0];
    let mut highest_idx: usize = 0;
    for j in 1..window {
        if data[j] >= highest { highest = data[j]; highest_idx = j; }
    }
    output[timeperiod] = (timeperiod - (timeperiod - highest_idx)) as f64 * inv_period;

    let mut trailing_idx = 1;
    let mut today = timeperiod + 1;

    while today < len {
        let h = data[today];
        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = data[trailing_idx];
            for (j, &val) in data[trailing_idx + 1..today + 1].iter().enumerate() {
                if val >= highest { highest = val; highest_idx = trailing_idx + 1 + j; }
            }
        } else if h >= highest {
            highest_idx = today;
            highest = h;
        }
        output[today] = (timeperiod - (today - highest_idx)) as f64 * inv_period;
        trailing_idx += 1;
        today += 1;
    }
    output
}

/// Single-pass min tracker for aroon_down (2-way cache).
#[inline]
fn aroon_min_pass(data: &[f64], timeperiod: usize, inv_period: f64) -> Vec<f64> {
    let len = data.len();
    let window = timeperiod + 1;
    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    let mut lowest = data[0];
    let mut lowest_idx: usize = 0;
    for j in 1..window {
        if data[j] <= lowest { lowest = data[j]; lowest_idx = j; }
    }
    output[timeperiod] = (timeperiod - (timeperiod - lowest_idx)) as f64 * inv_period;

    let mut trailing_idx = 1;
    let mut today = timeperiod + 1;

    while today < len {
        let l = data[today];
        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = data[trailing_idx];
            for (j, &val) in data[trailing_idx + 1..today + 1].iter().enumerate() {
                if val <= lowest { lowest = val; lowest_idx = trailing_idx + 1 + j; }
            }
        } else if l <= lowest {
            lowest_idx = today;
            lowest = l;
        }
        output[today] = (timeperiod - (today - lowest_idx)) as f64 * inv_period;
        trailing_idx += 1;
        today += 1;
    }
    output
}
