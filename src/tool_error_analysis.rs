use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Clone, Serialize, Deserialize, Display)]
pub enum ToolErrorCategory {
    #[serde(rename = "syntax error")]
    SyntaxError,
    #[serde(rename = "misc error")]
    MiscError,
    #[serde(rename = "wrong value")]
    WrongValue,
    #[serde(rename = "relevant but incorrect")]
    RelevantButIncorrect,
    #[serde(rename = "exactly same meaning")]
    ExactlySameMeaning,
    #[serde(rename = "language mismatch wrong value")]
    LanguageMismatchWrongValue,
    #[serde(rename = "language mismatch relevant but incorrect")]
    LanguageMismatchRelevantButIncorrect,
    #[serde(rename = "language mismatch exactly same meaning")]
    LanguageMismatchExactlySameMeaning,
    #[serde(rename = "other error")]
    OtherError,
}

#[derive(Clone, Display, Serialize, Deserialize)]
pub enum EvaluationError {
    NoFunctionCallsFound{
        raw_output: String,
    },
    JsonDecodeError{
        error_message: String,
        raw_output: String,
    },
    ParsingError{
        error_message: String,
        raw_output: String,
    },
    InvalidEntryCount{
        expected_count: usize,
        actual_count: usize,
        decoded_output: String,
    },
    WrongFuncName{
        expected_name: String,
        actual_name: String,
        decoded_output: String,
    },
    MissingRequiredParam{
        missing_param: String,
        required_params: Vec<String>,
        decoded_output: String,
    },
    UnexpectedParam{
        unexpected_param: String,
        expected_params: Vec<String>,
        decoded_output: String,
    },
    InvalidParamValue{
        param: String,
        actual_value: serde_json::Value,
        expected_values: Vec<serde_json::Value>,
        decoded_output: String,
    }
}
