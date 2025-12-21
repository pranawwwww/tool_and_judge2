use std::{collections::HashMap, sync::Arc};

use crate::{
    models::{
        backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    one_entry_map::KeyValuePair,
    single_or_list::SingleOrList,
    tool::bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall, BfclParameter},
    tool::error_analysis::EvaluationError,
};
use atomic_refcell::AtomicRefCell;
use indexmap::IndexMap;
use pyo3::types::PyList;
use pyo3::{Python, types::PyAnyMethods};
use serde::{Deserialize, Serialize};

/// Response structure from Python postprocess_tool_calls function
#[derive(Deserialize)]
struct PostprocessResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    calls: Option<Vec<HashMap<String, IndexMap<String, serde_json::Value>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

/// DeepSeek uses Python function call syntax for tool responses
/// Reference: https://api-docs.deepseek.com/api/create-chat-completion/
#[derive(Serialize)]
pub struct DeepSeekTool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: DeepSeekFunction,
}

#[derive(Serialize)]
pub struct DeepSeekFunction {
    pub name: String,
    pub description: String,
    pub parameters: DeepSeekParameter,
}

/// JSON Schema for DeepSeek function parameters
#[derive(Serialize)]
pub struct DeepSeekParameter {
    #[serde(rename = "type", skip_serializing_if = "type_is_any")]
    pub ty: SingleOrList<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, DeepSeekParameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<DeepSeekParameter>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<serde_json::Value>,
}

fn type_is_any(s: &SingleOrList<String>) -> bool {
    match s {
        SingleOrList::Single(val) => val == "any",
        SingleOrList::List(vals) => vals.iter().any(|v| v == "any"),
    }
}

#[derive(Copy, Clone)]
pub struct DeepSeekInterface;

impl DeepSeekInterface {
    pub fn map_type_hint(ty: &str) -> String {
        match ty {
            "dict" => "object".to_string(),
            "float" => "number".to_string(),
            "tuple" => "array".to_string(),
            _ => ty.to_string(),
        }
    }
}

fn bfcl_param_to_deepseek_param(
    bfcl_parameter: &BfclParameter,
    required: bool,
) -> DeepSeekParameter {
    let BfclParameter {
        ty: bfcl_type,
        properties: bfcl_properties,
        items: bfcl_items,
        r#enum: bfcl_enum,
        description: bfcl_description,
        format: bfcl_format,
        required: bfcl_required,
        default: bfcl_default,
        optional: _, // optional is unused because we use 'required' field to determine if a parameter is required
        maximum: bfcl_maximum,
    } = bfcl_parameter;
    let deepseek_type = DeepSeekInterface::map_type_hint(bfcl_type);
    let deepseek_type = match required {
        true => SingleOrList::Single(deepseek_type),
        false => SingleOrList::List(vec![deepseek_type, "null".to_string()]),
    };
    let bfcl_required = bfcl_required.as_ref();
    let deepseek_properties = bfcl_properties.as_ref().map(|props| {
        let mut deepseek_props = IndexMap::new();
        for (prop_name, prop_value) in props.iter() {
            let sub_required = bfcl_required.map_or(false, |reqs| reqs.contains(prop_name));
            let deepseek_prop_value = bfcl_param_to_deepseek_param(prop_value, sub_required);
            deepseek_props.insert(prop_name.clone(), deepseek_prop_value);
        }
        deepseek_props
    });

    let deepseek_items = bfcl_items.as_ref().map(|item| {
        let deepseek_item = bfcl_param_to_deepseek_param(item, true); // array items are always required
        Box::new(deepseek_item)
    });
    let deepseek_enum = bfcl_enum.clone();
    let deepseek_required: Option<Vec<String>> = bfcl_properties
        .as_ref()
        .map(|props| props.keys().cloned().collect());
    DeepSeekParameter {
        ty: deepseek_type,
        properties: deepseek_properties,
        description: bfcl_description.clone(),
        items: deepseek_items,
        r#enum: deepseek_enum,
        required: deepseek_required,
        default: bfcl_default.clone(),
        format: bfcl_format.clone(),
        maximum: bfcl_maximum.clone(),
    }
}

#[async_trait::async_trait]
impl ModelInterface for DeepSeekInterface {
    fn generate_tool_definitions(
        &self,
        bfcl_functions: &Vec<BfclFunctionDef>,
        name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value {
        // let sanitized_bfcl_functions = name_mapper.map_function_names(bfcl_functions);
        let mut deepseek_tools = Vec::new();
        for bfcl_func in bfcl_functions.iter() {
            let name = name_mapper
                .original_to_sanitized
                .get(&bfcl_func.name)
                .expect("Function name mapper does not contain key")
                .clone();
            let bfcl_param = &bfcl_func.parameters;
            let deepseek_params = bfcl_param_to_deepseek_param(bfcl_param, true);
            let description = bfcl_func.description.clone();
            deepseek_tools.push(DeepSeekTool {
                ty: "function".to_string(),
                function: DeepSeekFunction {
                    name,
                    description,
                    parameters: deepseek_params,
                },
            });
        }
        serde_json::to_value(deepseek_tools).expect("Failed to serialize DeepSeek tools")
    }
    // async fn generate_tool_call_async(
    //     &self,
    //     backend: Arc<ModelBackend>,
    //     raw_functions: Vec<BfclFunctionDef>,
    //     user_question: String,
    //     prompt_passing_in_english: bool,
    //     name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    // ) -> String {
    //     // downcast backend to api backend
    //     let ModelBackend::Api(api_backend) = backend.as_ref() else {
    //         panic!("deepseek interface should use ApiBackend");
    //     };
    //     let client = &api_backend.client;
    //     let deepseek_tools = {
    //         let mut name_mapper_borrow = name_mapper.borrow_mut();
    //         DeepSeekInterface::sanitize_and_convert_function_format(
    //             &raw_functions,
    //             &mut *name_mapper_borrow,
    //         )
    //     };

    //     let deepseek_tools_serialized =
    //         serde_json::to_string(&deepseek_tools).expect("Failed to serialize DeepSeek tools");

    //     let fut = Python::attach(|py| {
    //         let deepseek_backend_module = py
    //             .import("src_py.deepseek_backend")
    //             .expect("Failed to import src_py.deepseek_backend module");
    //         let generate_tool_call_async_fn = deepseek_backend_module
    //             .getattr("generate_tool_call_async")
    //             .expect("Failed to get generate_tool_call_async function");
    //         let model_name = api_backend.model.to_string();
    //         let json = py.import("json").expect("failed to import json");
    //         let deepseek_tools_obj = json
    //             .call_method("loads", (deepseek_tools_serialized,), None)
    //             .expect("Failed to parse DeepSeek tools JSON");
    //         assert!(deepseek_tools_obj.is_instance_of::<PyList>());
    //         let arguments = (
    //             model_name,
    //             client,
    //             user_question,
    //             deepseek_tools_obj,
    //             prompt_passing_in_english,
    //         );
    //         let fut = generate_tool_call_async_fn
    //             .call1(arguments)
    //             .expect("Failed to call generate_tool_call_async");
    //         pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
    //     });
    //     let response_str = fut.await.expect("DeepSeek tool call generation failed");
    //     let response_str = Python::attach(|py| {
    //         response_str
    //             .extract::<String>(py)
    //             .expect("Failed to extract response string")
    //     });
    //     response_str
    // }

    async fn translate_tool_question_async(
        &self,
        backend: Arc<ModelBackend>,
        user_question: String,
    ) -> String {
        let ModelBackend::Api(api_backend) = backend.as_ref() else {
            panic!("deepseek interface should use ApiBackend");
        };
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let deepseek_backend_module = py
                .import("src_py.deepseek_backend")
                .expect("Failed to import src_py.deepseek_backend module");
            let translate_tool_question_async_fn = deepseek_backend_module
                .getattr("translate_tool_question_async")
                .expect("Failed to get translate_tool_question_async function");
            let model_name = api_backend.model.to_string();
            let arguments = (model_name, client, user_question);
            let fut = translate_tool_question_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_question_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut
            .await
            .expect("DeepSeek tool question translation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }

    async fn translate_tool_answer_async(
        &self,
        backend: Arc<ModelBackend>,
        parameter_value: String,
    ) -> String {
        let ModelBackend::Api(api_backend) = backend.as_ref() else {
            panic!("deepseek interface should use ApiBackend");
        };
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let deepseek_backend_module = py
                .import("src_py.deepseek_backend")
                .expect("Failed to import src_py.deepseek_backend module");
            let translate_tool_answer_async_fn = deepseek_backend_module
                .getattr("translate_tool_answer_async")
                .expect("Failed to get translate_tool_answer_async function");
            let model_name = api_backend.model.to_string();
            let arguments = (model_name, client, parameter_value);
            let fut = translate_tool_answer_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_answer_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("DeepSeek tool answer translation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        // DeepSeek outputs Python function call syntax: [func_name(param=value)]
        // We need to parse this using Python's AST parser
        let json_result = Python::attach(|py| {
            let deepseek_backend_module = py
                .import("src_py.deepseek_backend")
                .expect("Failed to import src_py.deepseek_backend module");
            let postprocess_fn = deepseek_backend_module
                .getattr("postprocess_tool_calls")
                .expect("Failed to get postprocess_tool_calls function");

            let result = postprocess_fn
                .call1((raw_output,))
                .expect("Failed to call postprocess_tool_calls");

            // Extract the JSON string from Python
            result
                .extract::<String>()
                .expect("Failed to extract JSON string from postprocess result")
        });

        // Deserialize the JSON string
        let response: PostprocessResponse =
            serde_json::from_str(&json_result).map_err(|e| EvaluationError::ParsingError {
                error_message: format!("Failed to deserialize postprocess response: {}", e),
                raw_output: raw_output.to_string(),
            })?;

        if response.success {
            let calls_list = response
                .calls
                .ok_or_else(|| EvaluationError::ParsingError {
                    error_message: "Success response missing 'calls' field".to_string(),
                    raw_output: raw_output.to_string(),
                })?;

            let mut bfcl_calls = Vec::new();
            for call_map in calls_list {
                // Each call is a single-entry map: {func_name: {arguments}}
                if call_map.len() != 1 {
                    return Err(EvaluationError::ParsingError {
                        error_message: format!(
                            "Expected single-entry function call map, got {} entries",
                            call_map.len()
                        ),
                        raw_output: raw_output.to_string(),
                    });
                }
                let (func_name, arguments) = call_map.iter().next().unwrap();

                let original_function_name = name_mapper
                    .sanitized_to_original
                    .get(func_name)
                    .expect("Function name mapper does not contain key")
                    .clone();

                let bfcl_output_function_call = BfclOutputFunctionCall(KeyValuePair {
                    key: original_function_name,
                    value: arguments.clone(),
                });
                bfcl_calls.push(bfcl_output_function_call);
            }
            Ok(bfcl_calls)
        } else {
            let error_type = response
                .error_type
                .unwrap_or_else(|| "UNKNOWN_ERROR".to_string());
            let metadata = response.metadata.unwrap_or_default();

            // Map Python error type to Rust EvaluationError
            match error_type.as_str() {
                "PARSING_ERROR" => Err(EvaluationError::ParsingError {
                    error_message: metadata
                        .get("error_message")
                        .unwrap_or(&"Unknown parsing error".to_string())
                        .clone(),
                    raw_output: raw_output.to_string(),
                }),
                "JSON_DECODE_ERROR" => Err(EvaluationError::JsonDecodeError {
                    error_message: metadata
                        .get("error_message")
                        .unwrap_or(&"Unknown JSON decode error".to_string())
                        .clone(),
                    raw_output: raw_output.to_string(),
                }),
                "NO_FUNCTION_CALLS_FOUND" => Err(EvaluationError::NoFunctionCallsFound {
                    raw_output: raw_output.to_string(),
                }),
                _ => Err(EvaluationError::ParsingError {
                    error_message: format!("Unknown error type: {}", error_type),
                    raw_output: raw_output.to_string(),
                }),
            }
        }
    }
}
