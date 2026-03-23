use crate::error::TaResult;
use crate::ma_type::{compute_ma, MaType};

/// MA - Moving Average (selectable type)
///
/// Wrapper that dispatches to SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, MAMA, or T3
/// based on the `matype` parameter.
///
/// C TA-Lib signature: MA(input, timeperiod=30, matype=0)
/// matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, 5=TRIMA, 6=KAMA, 7=MAMA, 8=T3
pub fn ma(input: &[f64], timeperiod: usize, matype: MaType) -> TaResult<Vec<f64>> {
    compute_ma(input, timeperiod, matype)
}
