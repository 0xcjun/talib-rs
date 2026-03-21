pub mod error;
pub mod common;
pub mod traits;
pub mod ma_type;
pub mod simd;
pub mod sliding_window;

pub mod overlap;
pub mod momentum;
pub mod volatility;
pub mod volume;
pub mod price_transform;
pub mod cycle;
pub mod pattern;
pub mod statistic;
pub mod math_transform;
pub mod math_operator;

pub use error::{TaError, TaResult};
pub use ma_type::MaType;
