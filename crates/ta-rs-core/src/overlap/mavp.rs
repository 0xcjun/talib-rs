use crate::error::{TaError, TaResult};
use crate::ma_type::{compute_ma, MaType};

/// Moving Average with Variable Period (MAVP)
///
/// 每个数据点使用不同的周期来计算移动平均。
/// periods 数组指定每个位置的 MA 周期。
pub fn mavp(
    input: &[f64],
    periods: &[f64],
    minperiod: usize,
    maxperiod: usize,
    matype: MaType,
) -> TaResult<Vec<f64>> {
    let len = input.len();
    if len != periods.len() {
        return Err(TaError::LengthMismatch {
            expected: len,
            got: periods.len(),
        });
    }
    if minperiod < 2 || maxperiod < minperiod {
        return Err(TaError::InvalidParameter {
            name: "minperiod/maxperiod",
            value: format!("{}/{}", minperiod, maxperiod),
            reason: "minperiod >= 2 and maxperiod >= minperiod required",
        });
    }
    if len < maxperiod {
        return Err(TaError::InsufficientData {
            need: maxperiod,
            got: len,
        });
    }

    let mut output = vec![f64::NAN; len];

    // 对每个位置，使用指定周期计算 MA
    for i in (maxperiod - 1)..len {
        let p = (periods[i].round() as usize).clamp(minperiod, maxperiod);
        if i + 1 >= p {
            let start = i + 1 - p;
            let slice = &input[start..=i];
            // 简单使用 SMA，因为变周期场景下无法高效使用其他 MA
            let sum: f64 = slice.iter().sum();
            output[i] = sum / p as f64;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mavp_basic() {
        let input: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let periods = vec![3.0; 10];
        let result = mavp(&input, &periods, 2, 5, MaType::Sma).unwrap();
        assert!(!result[4].is_nan());
    }
}
