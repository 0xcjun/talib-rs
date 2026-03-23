use crate::error::{TaError, TaResult};

/// Money Flow Index (MFI) — precomputed tp/mf arrays for cache efficiency
///
/// MFI = 100 - (100 / (1 + Money Flow Ratio))
/// Money Flow Ratio = Positive Money Flow / Negative Money Flow
/// lookback = timeperiod
///
/// Two temp arrays (tp_arr, mf_arr) trade 16KB allocation at 1K bars
/// for 4× fewer FP ops per sliding window step at all sizes.
pub fn mfi(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    timeperiod: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() || len != volume.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()).min(volume.len()),
        });
    }
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    // Precompute tp and mf arrays — accessed multiple times per window step,
    // so array reads from L1 cache are faster than recomputing.
    let mut tp_arr = vec![0.0_f64; len];
    let mut mf_arr = vec![0.0_f64; len];
    for i in 0..len {
        let tp = (high[i] + low[i] + close[i]) / 3.0;
        tp_arr[i] = tp;
        mf_arr[i] = tp * volume[i];
    }

    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    // Initial window
    let mut pos_mf = 0.0_f64;
    let mut neg_mf = 0.0_f64;
    for i in 1..=timeperiod {
        let tp_cur = tp_arr[i];
        let tp_prev = tp_arr[i - 1];
        let mf = mf_arr[i];
        if tp_cur > tp_prev {
            pos_mf += mf;
        } else if tp_cur < tp_prev {
            neg_mf += mf;
        }
    }

    output[timeperiod] = if neg_mf > 0.0 {
        100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
    } else {
        100.0
    };

    // Sliding window
    for i in (timeperiod + 1)..len {
        let old_idx = i - timeperiod;
        let old_tp = tp_arr[old_idx];
        let old_tp_prev = tp_arr[old_idx - 1];
        let old_mf = mf_arr[old_idx];
        if old_tp > old_tp_prev {
            pos_mf -= old_mf;
        } else if old_tp < old_tp_prev {
            neg_mf -= old_mf;
        }

        let new_tp = tp_arr[i];
        let new_tp_prev = tp_arr[i - 1];
        let new_mf = mf_arr[i];
        if new_tp > new_tp_prev {
            pos_mf += new_mf;
        } else if new_tp < new_tp_prev {
            neg_mf += new_mf;
        }

        output[i] = if neg_mf > 0.0 {
            100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
        } else {
            100.0
        };
    }

    Ok(output)
}
