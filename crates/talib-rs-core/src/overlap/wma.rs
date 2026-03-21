use crate::error::{TaError, TaResult};

/// Weighted Moving Average (WMA) — O(n) 递推算法
///
/// 递推关系:
///   WS[i] = WS[i-1] + period * x[i] - PS[i-1]
///   PS[i] = PS[i-1] + x[i] - x[i-period]
/// 其中 WS = 加权和, PS = 简单和（前 period 个值）
/// WMA[i] = WS[i] / divider
///
/// 复杂度: O(n) 而非 O(n*period)
/// lookback = timeperiod - 1
pub fn wma(input: &[f64], timeperiod: usize) -> TaResult<Vec<f64>> {
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

    let mut output = vec![f64::NAN; len];
    let lookback = timeperiod - 1;
    let period_f = timeperiod as f64;
    let divider = period_f * (period_f + 1.0) / 2.0;

    // 初始窗口: 计算加权和 WS 和简单和 PS
    let mut ws = 0.0; // 加权和: Σ(w_i * x_i), w_i = 1..period
    let mut ps = 0.0; // 简单和: Σ(x_i), i in window
    for j in 0..timeperiod {
        let w = (j + 1) as f64;
        ws += input[j] * w;
        ps += input[j];
    }
    output[lookback] = ws / divider;

    // O(1) 递推:
    // 新元素 x[i] 进入窗口，旧元素 x[i-period] 离开
    // ws_new = ws_old + period * x[i] - ps_old
    //   解释: 新元素获得最高权重 period，所有旧元素权重各减1，等价于减去旧的简单和
    // ps_new = ps_old + x[i] - x[i-period]
    for i in timeperiod..len {
        ws = ws + period_f * input[i] - ps;
        ps = ps + input[i] - input[i - timeperiod];
        output[i] = ws / divider;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wma_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma(&input, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        let expected = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 6.0;
        assert!((result[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_wma_longer() {
        // 验证递推结果与暴力计算一致
        let input: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let period = 10;
        let result = wma(&input, period).unwrap();
        let divider = (period * (period + 1)) as f64 / 2.0;

        for i in (period - 1)..100 {
            let start = i + 1 - period;
            let mut expected = 0.0;
            for (w, j) in (start..=i).enumerate() {
                expected += input[j] * (w + 1) as f64;
            }
            expected /= divider;
            assert!(
                (result[i] - expected).abs() < 1e-8,
                "WMA mismatch at {}: got {} expected {}",
                i,
                result[i],
                expected
            );
        }
    }
}
