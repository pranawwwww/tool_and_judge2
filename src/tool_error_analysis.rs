use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Clone, EnumString, Display)]
pub enum ToolErrorCategory {
    #[strum(serialize = "syntax_error")]
    SyntaxError,
    #[strum(serialize = "misc_errors")]
    MiscErrors,
    #[strum(serialize = "wrong_values")]
    WrongValues,
    #[strum(serialize = "relevant_but_incorrect")]
    RelevantButIncorrect,
    #[strum(serialize = "exactly_same_meaning")]
    ExactlySameMeaning,
    #[strum(serialize = "language_mismatch_wrong_values")]
    LanguageMismatchWrongValues,
    #[strum(serialize = "language_mismatch_relevant_but_incorrect")]
    LanguageMismatchRelevantButIncorrect,
    #[strum(serialize = "language_mismatch_exactly_same_meaning")]
    LanguageMismatchExactlySameMeaning,
    #[strum(serialize = "other_errors")]
    OtherErrors,
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
