use crate::error::{TaError, TaResult};

/// Williams %R -- scalar brute rescan (amortized O(n))
///
/// WILLR = -100 * (highest_high - close) / (highest_high - lowest_low)
/// lookback = timeperiod - 1
pub fn willr(high: &[f64], low: &[f64], close: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let lookback = timeperiod - 1;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Initialize first window [0..timeperiod)
    let mut highest = high[0];
    let mut highest_idx: usize = 0;
    let mut lowest = low[0];
    let mut lowest_idx: usize = 0;
    for j in 1..timeperiod {
        if high[j] >= highest {
            highest = high[j];
            highest_idx = j;
        }
        if low[j] <= lowest {
            lowest = low[j];
            lowest_idx = j;
        }
    }
    {
        let range = highest - lowest;
        output[lookback] = if range > 0.0 {
            -100.0 * (highest - unsafe { *close.get_unchecked(lookback) }) / range
        } else {
            0.0
        };
    }

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let h = unsafe { *high.get_unchecked(today) };
        let l = unsafe { *low.get_unchecked(today) };

        // Max tracking on high[] — scalar brute rescan
        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = high[trailing_idx];
            for j in (trailing_idx + 1)..=today {
                if high[j] >= highest {
                    highest = high[j];
                    highest_idx = j;
                }
            }
        } else if h >= highest {
            highest_idx = today;
            highest = h;
        }

        // Min tracking on low[] — scalar brute rescan
        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = low[trailing_idx];
            for j in (trailing_idx + 1)..=today {
                if low[j] <= lowest {
                    lowest = low[j];
                    lowest_idx = j;
                }
            }
        } else if l <= lowest {
            lowest_idx = today;
            lowest = l;
        }

        let range = highest - lowest;
        let c = unsafe { *close.get_unchecked(today) };
        unsafe {
            *output.get_unchecked_mut(today) = if range > 0.0 {
                -100.0 * (highest - c) / range
            } else {
                0.0
            };
        }
        trailing_idx += 1;
        today += 1;
    }

    Ok(output)
}
