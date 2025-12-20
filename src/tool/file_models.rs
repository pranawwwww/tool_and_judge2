use serde::{Deserialize, Serialize};

use crate::{
    tool::bfcl_formats::BfclOutputFunctionCall,
    tool::error_analysis::{EvaluationError, ToolErrorCategory},
};

#[derive(Serialize, Deserialize)]
pub struct InferenceRawEntry {
    pub id: String,
    pub raw_output: String,
}


#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceJsonEntry {
    pub id: String,
    pub valid: bool,
    pub result: Result<Vec<BfclOutputFunctionCall>, EvaluationError>,
}


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
