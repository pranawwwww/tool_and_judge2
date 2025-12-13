
use serde::{Deserialize, Serialize};

use crate::tool_error_analysis::{EvaluationError, ToolErrorCategory};

#[derive(Serialize, Deserialize)]
pub struct InferenceRawEntry{
    pub id: String,
    pub raw_output: String,
}

impl InferenceRawEntry {
    pub fn new(id: String, raw_output: String) -> Self {
        Self { id, raw_output }
    }
}


// #[derive(Serialize, Deserialize, Clone)]
// pub enum ToolCallParsingResult{
//     Success(Vec<serde_json::Value>),
//     Failure(EvaluationError),
// }

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceJsonEntry{
    pub id: String,
    pub valid: bool,
    pub result: Result<Vec<serde_json::Value>, EvaluationError>,
}

impl InferenceJsonEntry {
    pub fn new(id: String, valid: bool, result: Result<Vec<serde_json::Value>, EvaluationError>) -> Self {
        Self { id, valid, result }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EvaluationResultEntry{
    pub id: String,
    pub valid: bool,
    pub error: Option<EvaluationError>,
}

impl EvaluationResultEntry {
    pub fn new(id: String, valid: bool, error: Option<EvaluationError>) -> Self {
        Self { id, valid, error }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EvaluationSummary{
    pub accuracy: f32,
    pub total_cases: usize,
    pub correct_cases: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CategorizedEntry{
    pub id: String,
    pub error: EvaluationError,
    pub error_category: ToolErrorCategory, 
}

impl CategorizedEntry {
    pub fn new(id: String, error: EvaluationError) -> Self {
        Self { id, error }
    }
}