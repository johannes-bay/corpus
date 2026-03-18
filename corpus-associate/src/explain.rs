use std::fmt;

/// Human-readable explanation of why a match was scored the way it was.
#[derive(Debug, Clone)]
pub struct MatchExplanation {
    pub filename: String,
    pub total_score: f64,
    /// (axis_name, score, detail_text)
    pub axis_details: Vec<(String, f64, String)>,
}

impl MatchExplanation {
    /// Build a new explanation from axis results.
    pub fn new(
        filename: &str,
        total_score: f64,
        axis_details: Vec<(String, f64, String)>,
    ) -> Self {
        Self {
            filename: filename.to_string(),
            total_score,
            axis_details,
        }
    }
}

impl fmt::Display for MatchExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.axis_details.len();
        for (i, (name, score, detail)) in self.axis_details.iter().enumerate() {
            let connector = if i + 1 < n { "\u{251c}\u{2500}" } else { "\u{2514}\u{2500}" };
            writeln!(f, "  {connector} {name:<12} {score:.2}  ({detail})")?;
        }
        Ok(())
    }
}
