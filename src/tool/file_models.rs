use serde::{Deserialize, Serialize};

use crate::{
    tool::error_analysis::{EvaluationError, ToolErrorCategory},
};







#[derive(Serialize, Deserialize, Clone)]
pub struct EvaluationResultEntry {
    pub id: String,
    pub valid: bool,
    pub error: Option<EvaluationError>,
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvaluationSummary {
    pub accuracy: f32,
    pub total_cases: usize,
    pub correct_cases: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CategorizedEntry {
    pub id: String,
    pub error_category: ToolErrorCategory,
    pub error: EvaluationError,
}
