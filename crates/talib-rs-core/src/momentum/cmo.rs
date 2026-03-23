use crate::error::{TaError, TaResult};

/// Chande Momentum Oscillator (CMO)
///
/// Uses Wilder smoothing (matching C TA-Lib exactly).
/// Initial sum_up/sum_down = sum of first `timeperiod` changes.
/// Subsequent values use Wilder smoothing:
///   sum_up  = sum_up  - (sum_up  / period) + current_up
///   sum_down = sum_down - (sum_down / period) + current_down
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

    let mut output = vec![0.0_f64; len];
    output[..timeperiod].fill(f64::NAN);

    // 初始窗口：计算前 timeperiod 个变化值的总和
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

    // Wilder 平滑递推（与 RSI 相同的平滑方式）
    let pf = timeperiod as f64;
    for i in (timeperiod + 1)..len {
        let change = input[i] - input[i - 1];
        let (cur_up, cur_down) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        sum_up = sum_up - (sum_up / pf) + cur_up;
        sum_down = sum_down - (sum_down / pf) + cur_down;

        let total = sum_up + sum_down;
        output[i] = if total > 0.0 {
            100.0 * (sum_up - sum_down) / total
        } else {
            0.0
        };
    }

    Ok(output)
}
