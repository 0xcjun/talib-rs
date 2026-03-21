use crate::error::{TaError, TaResult};

/// Money Flow Index (MFI)
///
/// MFI = 100 - (100 / (1 + Money Flow Ratio))
/// Money Flow Ratio = Positive Money Flow / Negative Money Flow
/// lookback = timeperiod
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

    // Typical Price
    let tp: Vec<f64> = (0..len)
        .map(|i| (high[i] + low[i] + close[i]) / 3.0)
        .collect();

    // Money Flow
    let mf: Vec<f64> = (0..len).map(|i| tp[i] * volume[i]).collect();

    let mut output = vec![f64::NAN; len];

    // 初始窗口
    let mut pos_mf = 0.0;
    let mut neg_mf = 0.0;
    for i in 1..=timeperiod {
        if tp[i] > tp[i - 1] {
            pos_mf += mf[i];
        } else if tp[i] < tp[i - 1] {
            neg_mf += mf[i];
        }
    }

    output[timeperiod] = if neg_mf > 0.0 {
        100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
    } else {
        100.0
    };

    // 滑动窗口
    for i in (timeperiod + 1)..len {
        // 移除旧值
        let old_idx = i - timeperiod;
        if tp[old_idx] > tp[old_idx - 1] {
            pos_mf -= mf[old_idx];
        } else if tp[old_idx] < tp[old_idx - 1] {
            neg_mf -= mf[old_idx];
        }
        // 添加新值
        if tp[i] > tp[i - 1] {
            pos_mf += mf[i];
        } else if tp[i] < tp[i - 1] {
            neg_mf += mf[i];
        }

        output[i] = if neg_mf > 0.0 {
            100.0 - (100.0 / (1.0 + pos_mf / neg_mf))
        } else {
            100.0
        };
    }

    Ok(output)
}
