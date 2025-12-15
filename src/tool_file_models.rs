
use serde::{Deserialize, Serialize};

use crate::{tool_bfcl_formats::BfclOutputFunctionCall, tool_error_analysis::{EvaluationError, ToolErrorCategory}};

#[derive(Serialize, Deserialize)]
pub struct InferenceRawEntry{
    pub id: String,
    pub raw_output: String,
}

// impl InferenceRawEntry {
//     pub fn new(id: String, raw_output: String) -> Self {
//         Self { id, raw_output }
//     }
// }


// #[derive(Serialize, Deserialize, Clone)]
// pub enum ToolCallParsingResult{
//     Success(Vec<serde_json::Value>),
//     Failure(EvaluationError),
// }

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceJsonEntry{
    pub id: String,
    pub valid: bool,
    pub result: Result<Vec<BfclOutputFunctionCall>, EvaluationError>,
}

// impl InferenceJsonEntry {
//     pub fn new(id: String, valid: bool, result: Result<Vec<BfclOutputFunctionCall>, EvaluationError>) -> Self {
//         Self { id, valid, result }
//     }
// }

#[derive(Serialize, Deserialize, Clone)]
pub struct EvaluationResultEntry{
    pub id: String,
    pub valid: bool,
    pub error: Option<EvaluationError>,
}

// impl EvaluationResultEntry {
//     pub fn new(id: String, valid: bool, error: Option<EvaluationError>) -> Self {
//         Self { id, valid, error }
//     }
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvaluationSummary{
    pub accuracy: f32,
    pub total_cases: usize,
    pub correct_cases: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CategorizedEntry{
    pub id: String,
    pub error_category: ToolErrorCategory, 
    pub error: EvaluationError,    
}

// impl CategorizedEntry {
//     pub fn new(id: String, error: EvaluationError, error_category: ToolErrorCategory) -> Self {
//         Self { id, error, error_category }
//     }
// }