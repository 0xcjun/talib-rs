/// 参数默认值类型
#[derive(Debug, Clone)]
pub enum ParamDefault {
    Integer(i32),
    Real(f64),
    MaType(i32),
}

/// 参数定义
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: &'static str,
    pub default: ParamDefault,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// 指标标志位
#[derive(Debug, Clone, Copy, Default)]
pub struct IndicatorFlags {
    /// 是否有不稳定期
    pub has_unstable_period: bool,
    /// 是否为 K 线形态识别
    pub is_candlestick: bool,
}

/// 指标元数据 trait — 用于 Abstract API 和函数自省
pub trait IndicatorInfo {
    /// 指标名称，如 "SMA"
    fn name() -> &'static str;
    /// 所属分组，如 "Overlap Studies"
    fn group() -> &'static str;
    /// 所需输入名称，如 ["close"] 或 ["high", "low", "close"]
    fn input_names() -> &'static [&'static str];
    /// 输出名称，如 ["real"] 或 ["upperband", "middleband", "lowerband"]
    fn output_names() -> &'static [&'static str];
    /// 参数定义列表
    fn parameters() -> Vec<ParamDef>;
    /// 标志位
    fn flags() -> IndicatorFlags;
}
