use crate::error::{TaError, TaResult};

/// Chande Momentum Oscillator (CMO)
///
/// CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down)
/// lookback = timeperiod
pub fn cmo(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    if len <= timeperiod {
        return Err(TaError::InsufficientData {
            need: timeperiod + 1,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];

    // 初始窗口
    let mut sum_up = 0.0;
    let mut sum_down = 0.0;
    for i in 1..=timeperiod {
        let change = input[i] - input[i - 1];
        if change > 0.0 {
            sum_up += change;
        } else {
            sum_down -= change;
        }
    }

    let total = sum_up + sum_down;
    output[timeperiod] = if total > 0.0 {
        100.0 * (sum_up - sum_down) / total
    } else {
        0.0
    };

    // 滑动窗口
    for i in (timeperiod + 1)..len {
        // 移除旧值
        let old_change = input[i - timeperiod] - input[i - timeperiod - 1];
        if old_change > 0.0 {
            sum_up -= old_change;
        } else {
            sum_down += old_change;
        }
        // 添加新值
        let new_change = input[i] - input[i - 1];
        if new_change > 0.0 {
            sum_up += new_change;
        } else {
            sum_down -= new_change;
        }

        let total = sum_up + sum_down;
        output[i] = if total > 0.0 {
            100.0 * (sum_up - sum_down) / total
        } else {
            0.0
        };
    }

    Ok(output)
}
