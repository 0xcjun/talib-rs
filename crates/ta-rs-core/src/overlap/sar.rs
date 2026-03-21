use crate::error::{TaError, TaResult};

/// Parabolic SAR
///
/// acceleration 默认 0.02, maximum 默认 0.2
/// lookback = 1
pub fn sar(
    high: &[f64],
    low: &[f64],
    acceleration: f64,
    maximum: f64,
) -> TaResult<Vec<f64>> {
    let len = high.len();
    if len != low.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: low.len(),
        });
    }
    if len < 2 {
        return Err(TaError::InsufficientData { need: 2, got: len });
    }

    let mut output = vec![f64::NAN; len];
    let mut af = acceleration;
    let mut is_long: bool;
    let mut sar_val: f64;
    let mut ep: f64;

    // 初始化: 判断初始方向
    if high[1] > high[0] {
        is_long = true;
        ep = high[0];
        sar_val = low[0];
    } else {
        is_long = false;
        ep = low[0];
        sar_val = high[0];
    }

    output[0] = sar_val;

    for i in 1..len {
        if is_long {
            // 多头
            if low[i] < sar_val {
                // 反转为空头
                is_long = false;
                sar_val = ep;
                ep = low[i];
                af = acceleration;
            } else {
                if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(maximum);
                }
            }
        } else {
            // 空头
            if high[i] > sar_val {
                // 反转为多头
                is_long = true;
                sar_val = ep;
                ep = high[i];
                af = acceleration;
            } else {
                if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration).min(maximum);
                }
            }
        }

        output[i] = sar_val;

        // 更新 SAR
        sar_val = sar_val + af * (ep - sar_val);

        // SAR 不能超过前两根 K 线的范围
        if is_long {
            sar_val = sar_val.min(low[i]);
            if i > 0 {
                sar_val = sar_val.min(low[i - 1]);
            }
        } else {
            sar_val = sar_val.max(high[i]);
            if i > 0 {
                sar_val = sar_val.max(high[i - 1]);
            }
        }
    }

    Ok(output)
}

/// Parabolic SAR Extended — 与 sar 相同但参数更多
pub fn sar_ext(
    high: &[f64],
    low: &[f64],
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
) -> TaResult<Vec<f64>> {
    // 简化实现: 使用 sar 的逻辑，暂时使用 long 参数
    // TODO: 完整实现区分多空不同加速因子
    sar(high, low, accelerationinitlong, accelerationmaxlong)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sar_basic() {
        let high = vec![10.0, 11.0, 12.0, 11.5, 13.0, 12.0, 11.0, 10.5, 10.0, 9.0];
        let low = vec![9.0, 9.5, 10.0, 10.0, 11.0, 10.5, 9.5, 9.0, 8.5, 8.0];
        let result = sar(&high, &low, 0.02, 0.2).unwrap();
        assert_eq!(result.len(), 10);
        assert!(!result[0].is_nan());
    }
}
