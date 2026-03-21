use crate::error::{TaError, TaResult};

/// 移动平均类型枚举，与 TA-Lib 的 MA_Type 完全兼容
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum MaType {
    Sma = 0,
    Ema = 1,
    Wma = 2,
    Dema = 3,
    Tema = 4,
    Trima = 5,
    Kama = 6,
    Mama = 7,
    T3 = 8,
}

impl TryFrom<i32> for MaType {
    type Error = TaError;

    fn try_from(value: i32) -> TaResult<Self> {
        match value {
            0 => Ok(MaType::Sma),
            1 => Ok(MaType::Ema),
            2 => Ok(MaType::Wma),
            3 => Ok(MaType::Dema),
            4 => Ok(MaType::Tema),
            5 => Ok(MaType::Trima),
            6 => Ok(MaType::Kama),
            7 => Ok(MaType::Mama),
            8 => Ok(MaType::T3),
            _ => Err(TaError::InvalidParameter {
                name: "matype",
                value: value.to_string(),
                reason: "must be 0-8",
            }),
        }
    }
}

/// 根据 MaType 调度到对应的移动平均计算函数
pub fn compute_ma(input: &[f64], period: usize, ma_type: MaType) -> TaResult<Vec<f64>> {
    use crate::overlap;
    match ma_type {
        MaType::Sma => overlap::sma(input, period),
        MaType::Ema => overlap::ema(input, period),
        MaType::Wma => overlap::wma(input, period),
        MaType::Dema => overlap::dema(input, period),
        MaType::Tema => overlap::tema(input, period),
        MaType::Trima => overlap::trima(input, period),
        MaType::Kama => overlap::kama(input, period),
        // MAMA 和 T3 有额外参数，这里使用默认值
        MaType::Mama => {
            let (mama, _fama) = overlap::mama(input, 0.5, 0.05)?;
            Ok(mama)
        }
        MaType::T3 => overlap::t3(input, period, 0.7),
    }
}
