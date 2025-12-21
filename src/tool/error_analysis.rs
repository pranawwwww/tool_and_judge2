use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumIter};

/// The serialized names must match the gpt5's prompt
#[derive(Clone, Serialize, Deserialize, Display, EnumIter, PartialEq, Eq, Hash)]
pub enum ToolErrorCategory {
    #[serde(rename = "SYNTAX_ERROR")]
    SyntaxError,
    #[serde(rename = "MISC_ERROR")]
    MiscError,
    #[serde(rename = "WRONG_VALUE")]
    WrongValue,
    #[serde(rename = "RELEVANT_BUT_INCORRECT")]
    RelevantButIncorrect,
    #[serde(rename = "EXACTLY_SAME_MEANING")]
    ExactlySameMeaning,
    #[serde(rename = "LANGUAGE_MISMATCH_WRONG_VALUE")]
    LanguageMismatchWrongValue,
    #[serde(rename = "LANGUAGE_MISMATCH_RELEVANT_BUT_INCORRECT")]
    LanguageMismatchRelevantButIncorrect,
    #[serde(rename = "LANGUAGE_MISMATCH_EXACTLY_SAME_MEANING")]
    LanguageMismatchExactlySameMeaning,
    #[serde(rename = "OTHER_ERROR")]
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
