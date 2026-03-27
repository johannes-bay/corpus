pub mod axes;
pub mod composer;
pub mod concept;
pub mod explain;
pub mod matcher;

// Re-export key types for convenience.
pub use axes::{Axis, AxisRegistry, ScoringContext};
pub use explain::MatchExplanation;
pub use matcher::{ScoredMatch, WeightedAxis};
