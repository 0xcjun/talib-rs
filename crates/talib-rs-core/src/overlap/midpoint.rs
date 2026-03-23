use crate::error::{TaError, TaResult};

/// MIDPOINT -- scalar brute rescan (amortized O(n))
///
/// MIDPOINT = (highest + lowest) / 2
/// lookback = timeperiod - 1
pub fn midpoint(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = input.len();
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

    // Initialize first window
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
    output[lookback] = (highest + lowest) / 2.0;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let v = input[today];

        if highest_idx < trailing_idx {
            highest_idx = trailing_idx;
            highest = input[trailing_idx];
            for j in (trailing_idx + 1)..=today {
                if input[j] >= highest {
                    highest = input[j];
                    highest_idx = j;
                }
            }
        } else if v >= highest {
            highest_idx = today;
            highest = v;
        }

        if lowest_idx < trailing_idx {
            lowest_idx = trailing_idx;
            lowest = input[trailing_idx];
            for j in (trailing_idx + 1)..=today {
                if input[j] <= lowest {
                    lowest = input[j];
                    lowest_idx = j;
                }
            }
        } else if v <= lowest {
            lowest_idx = today;
            lowest = v;
        }

        output[today] = (highest + lowest) / 2.0;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

/// MIDPRICE -- scalar brute rescan (amortized O(n))
///
/// MIDPRICE = (highest_high + lowest_low) / 2
/// lookback = timeperiod - 1
pub fn midprice(high: &[f64], low: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod == 0 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: "0".to_string(),
            reason: "must be >= 1",
        });
    }
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len < timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod,
            got: len,
        });
    }

    let lookback = timeperiod - 1;
    let mut output = vec![0.0_f64; len];
    output[..lookback].fill(f64::NAN);

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
    output[lookback] = (highest + lowest) / 2.0;

    let mut trailing_idx = 1;
    let mut today = timeperiod;

    while today < len {
        let h = high[today];
        let l = low[today];

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

        output[today] = (highest + lowest) / 2.0;
        trailing_idx += 1;
        today += 1;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midpoint_basic() {
        let input = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = midpoint(&input, 3).unwrap();
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 1e-10);
    }
}
