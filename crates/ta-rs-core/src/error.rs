use thiserror::Error;

/// ta-rs 核心错误类型
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TaError {
    /// 输入数据长度不足以计算指标
    #[error("input data too short: need at least {need} elements, got {got}")]
    InsufficientData { need: usize, got: usize },

    /// 参数值无效
    #[error("invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        name: &'static str,
        value: String,
        reason: &'static str,
    },

    /// 缺少必要的输入数组
    #[error("missing required input: {0}")]
    MissingInput(&'static str),

    /// 输入数组长度不一致
    #[error("input length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
}

/// 便捷结果类型
pub type TaResult<T> = Result<T, TaError>;
