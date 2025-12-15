use crate::{
    tool_bfcl_formats::{BfclDatasetEntry, BfclGroundTruthEntry, BfclOutputFunctionCall},
    tool_error_analysis::EvaluationError,
    tool_file_models::{EvaluationResultEntry, InferenceJsonEntry},
};

pub fn evaluate_entry(
    id: String,
    inference_entry: &InferenceJsonEntry,
    test_case_entry: &BfclDatasetEntry,
    ground_truth_entry: &BfclGroundTruthEntry,
) -> EvaluationResultEntry {
    let functions = match &inference_entry.result {
        Ok(funcs) => funcs,
        Err(e) => {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(e.clone()),
            };
        }
    };
    if functions.len() != 1 {
        return EvaluationResultEntry {
            id,
            valid: false,
            error: Some(EvaluationError::InvalidEntryCount {
                expected_count: 1,
                actual_count: functions.len(),
                decoded_output: serde_json::to_string(functions)
                    .expect("Should serialize correctly"),
            }),
        };
    }
    let function = &functions[0];
    let ground_truth_functions = &ground_truth_entry.ground_truth;
    assert_eq!(
        ground_truth_functions.len(),
        1,
        "Each ground truth entry should have exactly one function call"
    );
    let ground_truth_function = &ground_truth_functions[0];
    let output_function_name = &function.0.key;
    let ground_truth_function_name = &ground_truth_function.0.key;
    if output_function_name != ground_truth_function_name {
        return EvaluationResultEntry {
            id,
            valid: false,
            error: Some(EvaluationError::WrongFuncName {
                expected_name: ground_truth_function_name.clone(),
                actual_name: output_function_name.clone(),
                decoded_output: serde_json::to_string(functions)
                    .expect("Should serialize correctly"),
            }),
        };
    }
    let target_test_case_function = test_case_entry
        .function
        .iter()
        .find(|f| f.name == *output_function_name)
        .expect("The test case should contain the target function");
    // let target_function_required_parameters = &target_test_case_function.required;
    let prarameters = &function.0.value;
    // TODO: refactor the required parameters handling
    for required_param in target_function_required_parameters {
        if !prarameters.contains_key(required_param) {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::MissingRequiredParam {
                    missing_param: required_param.clone(),
                    required_params: target_function_required_parameters.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                }),
            };
        }
    }
    let ground_truth_parameters = &ground_truth_function.parameters;
    for (param, value) in prarameters.iter() {
        let Some(ground_truth_parameter_values) = ground_truth_parameters.get(param) else {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::UnexpectedParam {
                    unexpected_param: param.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                    expected_params: ground_truth_parameters.keys().cloned().collect(),
                }),
            };
        };
        if !value_matches_list(value, ground_truth_parameter_values) {
            return EvaluationResultEntry {
                id,
                valid: false,
                error: Some(EvaluationError::InvalidParamValue {
                    param: param.clone(),
                    actual_value: value.clone(),
                    expected_values: ground_truth_parameter_values.clone(),
                    decoded_output: serde_json::to_string(functions)
                        .expect("Should serialize correctly"),
                }),
            };
        }
    }
    EvaluationResultEntry {
        id,
        valid: true,
        error: None,
    }
}

fn value_matches_list(value: &serde_json::Value, expected_list: &Vec<serde_json::Value>) -> bool {
    for expected in expected_list {
        if value_matches_any(value, expected) {
            return true;
        }
    }
    false
}

fn value_matches_any(value: &serde_json::Value, expected: &serde_json::Value) -> bool {
    match (value, expected) {
        (serde_json::Value::Number(num1), serde_json::Value::Number(num2)) => {
            let num1 = num1.as_f64().expect("Should be convertible to f64");
            let num2 = num2.as_f64().expect("Should be convertible to f64");
            if (num1 - num2).abs() < 0.0001 {
                return true;
            }
        }
        (serde_json::Value::Null, serde_json::Value::Null) => return true,
        (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) => return b1 == b2,
        (serde_json::Value::String(s1), serde_json::Value::String(s2)) => return s1 == s2,
        (serde_json::Value::Array(arr1), serde_json::Value::Array(arr2)) => {
            if arr1.len() != arr2.len() {
                return false;
            }
            for (v1, v2) in arr1.iter().zip(arr2.iter()) {
                if !value_matches_any(v1, v2) {
                    return false;
                }
            }
            return true;
        }
        (serde_json::Value::Object(map1), serde_json::Value::Object(map2)) => {
            if map1.len() != map2.len() {
                return false;
            }
            for (k, v1) in map1.iter() {
                let Some(v2) = map2.get(k) else {
                    return false;
                };
                if !value_matches_any(v1, v2) {
                    return false;
                }
            }
            return true;
        }
        _ => {}
    }
    // value matches elements in expected array
    if let serde_json::Value::Array(expected_arry) = expected {
        for expected_item in expected_arry {
            if value_matches_any(value, expected_item) {
                return true;
            }
        }
    }
    return false;
}
