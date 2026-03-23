use crate::error::{TaError, TaResult};

/// Ultimate Oscillator (ULTOSC)
///
/// ULTOSC = 100 * (4*avg7 + 2*avg14 + avg28) / 7
/// 其中 avg_n = sum(BP, n) / sum(TR, n)
/// BP (Buying Pressure) = close - min(low, prev_close)
/// TR (True Range) = max(high, prev_close) - min(low, prev_close)
pub fn ultosc(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() || len != close.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len().min(close.len()),
        });
    }
    let max_period = timeperiod1.max(timeperiod2).max(timeperiod3);
    if len <= max_period {
        return Err(TaError::InsufficientData {
            need: max_period + 1,
            got: len,
        });
    }

    // 计算 BP 和 TR
    let mut bp = vec![0.0; len];
    let mut tr = vec![0.0; len];
    for i in 1..len {
        let true_low = low[i].min(close[i - 1]);
        let true_high = high[i].max(close[i - 1]);
        bp[i] = close[i] - true_low;
        tr[i] = true_high - true_low;
    }

    let mut output = vec![0.0_f64; len];
    output[..max_period].fill(f64::NAN);

    // Initialize sliding sums for the first output position (i = max_period)
    // Window for period p at position i covers bp[(i+1-p)..=i]
    let i0 = max_period;
    let mut sum_bp1: f64 = bp[(i0 + 1 - timeperiod1)..=i0].iter().sum();
    let mut sum_tr1: f64 = tr[(i0 + 1 - timeperiod1)..=i0].iter().sum();
    let mut sum_bp2: f64 = bp[(i0 + 1 - timeperiod2)..=i0].iter().sum();
    let mut sum_tr2: f64 = tr[(i0 + 1 - timeperiod2)..=i0].iter().sum();
    let mut sum_bp3: f64 = bp[(i0 + 1 - timeperiod3)..=i0].iter().sum();
    let mut sum_tr3: f64 = tr[(i0 + 1 - timeperiod3)..=i0].iter().sum();

    let avg1 = if sum_tr1 > 0.0 { sum_bp1 / sum_tr1 } else { 0.0 };
    let avg2 = if sum_tr2 > 0.0 { sum_bp2 / sum_tr2 } else { 0.0 };
    let avg3 = if sum_tr3 > 0.0 { sum_bp3 / sum_tr3 } else { 0.0 };
    output[i0] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;

    // Slide sums forward: add new element, remove oldest
    for i in (max_period + 1)..len {
        sum_bp1 += bp[i] - bp[i - timeperiod1];
        sum_tr1 += tr[i] - tr[i - timeperiod1];
        sum_bp2 += bp[i] - bp[i - timeperiod2];
        sum_tr2 += tr[i] - tr[i - timeperiod2];
        sum_bp3 += bp[i] - bp[i - timeperiod3];
        sum_tr3 += tr[i] - tr[i - timeperiod3];

        let avg1 = if sum_tr1 > 0.0 { sum_bp1 / sum_tr1 } else { 0.0 };
        let avg2 = if sum_tr2 > 0.0 { sum_bp2 / sum_tr2 } else { 0.0 };
        let avg3 = if sum_tr3 > 0.0 { sum_bp3 / sum_tr3 } else { 0.0 };
        output[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
    }

    Ok(output)
}
