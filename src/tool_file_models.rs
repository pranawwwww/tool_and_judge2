
use serde::{Deserialize, Serialize};

use crate::tool_error_analysis::EvaluationError;

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


#[derive(Serialize, Deserialize, Clone)]
pub enum ToolCallParsingResult{
    Success(Vec<serde_json::Value>),
    Failure(EvaluationError),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceJsonEntry{
    pub id: String,
    pub valid: bool,
    pub result: ToolCallParsingResult,
}

impl InferenceJsonEntry {
    pub fn new(id: String, valid: bool, result: ToolCallParsingResult) -> Self {
        Self { id, valid, result }
    }
}
