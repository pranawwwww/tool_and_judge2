use indexmap::IndexMap;

use crate::tool::{bfcl_formats::{
        BfclDatasetEntry, BfclGroundTruthEntry, BfclParameter,
    }, error_analysis::EvaluationError, file_models::EvaluationResultEntry, passes::pass_parse_output::ParseOutputEntry};

pub fn evaluate_entry(
    id: String,
    parse_output_entry: &ParseOutputEntry,
    test_case_entry: &BfclDatasetEntry,
    ground_truth_entry: &BfclGroundTruthEntry,
) -> EvaluationResultEntry {
    let functions = match &parse_output_entry.result {
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
    let test_case_function_def = test_case_entry
        .function
        .iter()
        .find(|f| f.name == *output_function_name)
        .expect("The test case should contain the target function");

    let test_case_outermost_properties = test_case_function_def
        .parameters
        .properties
        .as_ref()
        .expect("The outermost parameter should have a properties field");
    let test_case_outermost_required = &test_case_function_def
        .parameters
        .required
        .as_ref()
        .expect("The outermost parameter should have a required field");
    let decoded_output = serde_json::to_string(functions).expect("Should serialize correctly");
    // let target_function_required_parameters = &target_test_case_function.required;
    let prarameters = &function.0.value;
    // Checking required parameters and unexpected parameters recursively
    if let Err(error) = check_recursively_for_required_and_unexpected(
        prarameters,
        test_case_outermost_properties,
        test_case_outermost_required,
        &decoded_output,
    ) {
        return EvaluationResultEntry {
            id,
            valid: false,
            error: Some(error),
        };
    }

    let ground_truth_parameters = &ground_truth_function.0.value;
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

// enum RecursiveCheckResult {
//     Valid,
//     MissingRequiredParam {
//         missing_param: String,
//         required_params: Vec<String>,
//     },
//     UnexpectedParam {
//         unexpected_param: String,
//         expected_params: Vec<String>,
//     },
//     InvalidParamType,
//     InvalidParamValue {
//         param: String,
//         actual_value: serde_json::Value,
//         expected_values: Vec<serde_json::Value>,
//     },
// }

fn check_recursively_for_required_and_unexpected(
    parameters: &IndexMap<String, serde_json::Value>,
    // parameters_def: &BfclParameter,
    properties_def: &IndexMap<String, BfclParameter>,
    required_params: &Vec<String>,
    decoded_output: &String,
    // required_params: &Vec<String>,
) -> Result<(), EvaluationError> {
    for required_param in required_params {
        if !parameters.contains_key(required_param) {
            return Err(EvaluationError::MissingRequiredParam {
                missing_param: required_param.clone(),
                required_params: required_params.clone(),
                decoded_output: decoded_output.clone(),
            });
        }
    }
    // recursively check the sub-parameters
    for (parameter_name, parameter_value) in parameters.iter() {
        let Some(parameter_def) = properties_def.get(parameter_name) else {
            return Err(EvaluationError::UnexpectedParam {
                unexpected_param: parameter_name.clone(),
                expected_params: properties_def.keys().cloned().collect(),
                decoded_output: decoded_output.clone(),
            });
        };
        let (Some(sub_properties), Some(sub_required_params)) =
            (&parameter_def.properties, &parameter_def.required)
        else {
            // assert!(
            //     parameter_def.properties.is_none() && parameter_def.required.is_none(),
            //     "If one of required or properties is None, both should be None, but got required is {:?} and properties is {:?}",
            //     parameter_def.required,
            //     parameter_def.properties
            // );
            // if there are no required parameters for this property, skip to next
            continue;
        };
        // this parameter shoud be a dict / object
        let serde_json::Value::Object(sub_parameters_map) = parameter_value else {
            // The parameter value does not have the correct type, stop doing recursive checking for required and unexpected sub-parameters.
            // The type error will be caught at the invalid value checking stage.
            return Ok(());
        };
        // Convert the Map<String, Value> to IndexMap<String, Value>
        let sub_parameters_map: IndexMap<String, serde_json::Value> =
            sub_parameters_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        check_recursively_for_required_and_unexpected(
            &sub_parameters_map,
            sub_properties,
            sub_required_params,
            decoded_output,
        )?; // "?" is a syntax sugar that returns the function's return value if error occurs
    }
    Ok(())
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
    if let (serde_json::Value::Null, serde_json::Value::String(expected_str)) = (value, expected) {
        if expected_str == "" {
            return true;
        }
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
