/// 将输出数组的前 `lookback` 个元素填充为 NaN，匹配 TA-Lib 行为
#[inline]
pub fn fill_nan_prefix(output: &mut [f64], lookback: usize) {
    for v in output.iter_mut().take(lookback) {
        *v = f64::NAN;
    }
}

/// 验证输入数组长度是否满足 lookback 要求
#[inline]
pub fn validate_length(len: usize, lookback: usize) -> bool {
    len > lookback
}

/// 用于 OHLCV 输入的统一容器，所有切片均为借用（零拷贝）
#[derive(Debug, Clone, Copy)]
pub struct OhlcvInputs<'a> {
    pub open: Option<&'a [f64]>,
    pub high: Option<&'a [f64]>,
    pub low: Option<&'a [f64]>,
    pub close: Option<&'a [f64]>,
    pub volume: Option<&'a [f64]>,
}

impl<'a> OhlcvInputs<'a> {
    /// 创建仅包含 close 数据的输入
    pub fn close_only(close: &'a [f64]) -> Self {
        Self {
            open: None,
            high: None,
            low: None,
            close: Some(close),
            volume: None,
        }
    }

    /// 创建包含 HLC 数据的输入
    pub fn hlc(high: &'a [f64], low: &'a [f64], close: &'a [f64]) -> Self {
        Self {
            open: None,
            high: Some(high),
            low: Some(low),
            close: Some(close),
            volume: None,
        }
    }

    /// 创建包含完整 OHLCV 数据的输入
    pub fn full(
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
    ) -> Self {
        Self {
            open: Some(open),
            high: Some(high),
            low: Some(low),
            close: Some(close),
            volume: Some(volume),
        }
    }
}
