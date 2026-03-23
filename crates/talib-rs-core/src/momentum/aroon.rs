use crate::error::{TaError, TaResult};

/// Aroon (AROON) — single-pass brute rescan with slice iterators
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

    let period_f = timeperiod as f64;
    let window = timeperiod + 1;

    let mut aroon_down = vec![0.0_f64; len];
    aroon_down[..timeperiod].fill(f64::NAN);
    let mut aroon_up = vec![0.0_f64; len];
    aroon_up[..timeperiod].fill(f64::NAN);

    let mut highest = high[0];
    let mut highest_idx: usize = 0;
    let mut lowest = low[0];
    let mut lowest_idx: usize = 0;
    for j in 1..window {
        if high[j] >= highest { highest = high[j]; highest_idx = j; }
        if low[j] <= lowest { lowest = low[j]; lowest_idx = j; }
    }
    aroon_up[timeperiod] = 100.0 * (period_f - (timeperiod - highest_idx) as f64) / period_f;
    aroon_down[timeperiod] = 100.0 * (period_f - (timeperiod - lowest_idx) as f64) / period_f;

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

        aroon_up[today] = 100.0 * (period_f - (today - highest_idx) as f64) / period_f;
        aroon_down[today] = 100.0 * (period_f - (today - lowest_idx) as f64) / period_f;
        trailing_idx += 1;
        today += 1;
    }

    Ok((aroon_down, aroon_up))
}

/// Aroon Oscillator — single-pass, AROONOSC = Aroon Up - Aroon Down
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

    let period_f = timeperiod as f64;
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
        let up = 100.0 * (period_f - (timeperiod - highest_idx) as f64) / period_f;
        let down = 100.0 * (period_f - (timeperiod - lowest_idx) as f64) / period_f;
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

        let up = 100.0 * (period_f - (today - highest_idx) as f64) / period_f;
        let down = 100.0 * (period_f - (today - lowest_idx) as f64) / period_f;
        output[today] = up - down;
        trailing_idx += 1;
        today += 1;
    }

    Ok(output)
}
