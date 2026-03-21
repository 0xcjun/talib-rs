use crate::error::{TaError, TaResult};

/// Kaufman Adaptive Moving Average (KAMA)
///
/// KAMA 根据价格效率比（方向变动/总波动）动态调整平滑常数。
/// 效率比高时跟踪快（接近快速 EMA），低时跟踪慢（接近慢速 EMA）。
/// lookback = timeperiod
pub fn kama(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
    if timeperiod < 2 {
        return Err(TaError::InvalidParameter {
            name: "timeperiod",
            value: timeperiod.to_string(),
            reason: "must be >= 2",
        });
    }
    let len = input.len();
    let lookback = timeperiod;
    if len <= lookback {
        return Err(TaError::InsufficientData {
            need: lookback + 1,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];

    // TA-Lib 默认的快速/慢速常数
    let fast_sc = 2.0 / (2.0 + 1.0);   // 快速 EMA 周期 = 2
    let slow_sc = 2.0 / (30.0 + 1.0);   // 慢速 EMA 周期 = 30

    // 第一个 KAMA 值 = input[lookback]（TA-Lib 兼容）
    let mut prev_kama = input[lookback];
    output[lookback] = prev_kama;

    for i in (lookback + 1)..len {
        // 方向变动 (direction)
        let direction = (input[i] - input[i - timeperiod]).abs();

        // 总波动 (volatility)
        let mut volatility = 0.0;
        for j in (i - timeperiod + 1)..=i {
            volatility += (input[j] - input[j - 1]).abs();
        }

        // 效率比 (ER)
        let er = if volatility > 0.0 {
            direction / volatility
        } else {
            0.0
        };

        // 平滑常数 (SC)
        let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

        // KAMA
        let kama_val = prev_kama + sc * (input[i] - prev_kama);
        output[i] = kama_val;
        prev_kama = kama_val;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kama_basic() {
        let input: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = kama(&input, 10).unwrap();
        // lookback = 10
        assert!(result[9].is_nan());
        assert!(!result[10].is_nan());
    }
}
