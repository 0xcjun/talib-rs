pub mod common;
pub mod error;
pub mod ma_type;
pub mod simd;
pub mod sliding_window;
pub mod traits;

pub mod cycle;
pub mod math_operator;
pub mod math_transform;
pub mod momentum;
pub mod overlap;
#[allow(unused_mut, unused_variables, unused_assignments, dead_code)]
pub mod pattern;
pub mod price_transform;
pub mod statistic;
pub mod volatility;
pub mod volume;

pub use error::{TaError, TaResult};
pub use ma_type::MaType;
