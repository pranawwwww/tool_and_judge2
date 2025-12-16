use std::{any::Any, collections::HashMap, sync::Arc};

use crate::{
    models::{
        api_backend::ApiBackend, backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    one_entry_map::KeyValuePair,
    single_or_list::SingleOrList,
    tool_bfcl_formats::{
        BfclFunctionDef, BfclGroundTruthFunctionCall, BfclOutputFunctionCall, BfclParameter,
    },
    tool_error_analysis::EvaluationError,
};
use atomic_refcell::AtomicRefCell;
use indexmap::IndexMap;
use pyo3::{Py, PyAny, Python, types::PyAnyMethods};
use pyo3::{prelude::*, types::PyList};
use serde::{Deserialize, Serialize};

use serde_json::Value;
use serde_json::json;

/// https://platform.openai.com/docs/guides/function-calling?strict-mode=enabled#defining-functions
#[derive(Serialize)]
pub struct Gpt5Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
    pub description: String,
    pub strict: bool,
    pub parameters: Gpt5Parameter,
}
/// JSON Schema for GPT-5 function parameters
#[derive(Serialize)]
pub struct Gpt5Parameter {
    #[serde(rename = "type", skip_serializing_if = "type_is_any")]
    pub ty: SingleOrList<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Gpt5Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Gpt5Parameter>>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub r#enum: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    // #[serde(rename = "additionalProperties")]
    // pub additional_properties: bool,
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

// #[derive(Serialize)]
// pub struct Gpt5PropertyValue {
//     #[serde(rename = "type", skip_serializing_if = "type_is_any")]
//     pub ty: String,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub items: Option<IndexMap<String, String>>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub default: Option<serde_json::Value>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     pub properties: Option<IndexMap<String, Gpt5PropertyValue>>,
//     pub description: String,
// }

#[derive(Deserialize, Clone)]
pub struct Gpt5OutputFunctionCall {
    name: String,
    arguments: IndexMap<String, serde_json::Value>,
}

impl Gpt5OutputFunctionCall {
    // pub fn try_deserialize_from_json(
    //     json_value: &serde_json::Value,
    //     raw_output: &str,
    // ) -> Result<Gpt5OutputFunctionCall, EvaluationError> {
    //     let json_obj = json_value
    //         .as_object()
    //         .expect("This is tested before calling this function");
    //     let name = json_obj
    //         .get("name")
    //         .and_then(|s| s.as_str())
    //         .ok_or_else(|| EvaluationError::ParsingError {
    //             error_message: "Missing parameter 'name' or 'name' is not of type 'str'".into(),
    //             raw_output: raw_output.into(),
    //         })?
    //         .to_string();
    //     let arguments_str = json_obj
    //         .get("arguments")
    //         .and_then(|s| s.as_str())
    //         .ok_or_else(|| EvaluationError::ParsingError {
    //             error_message: "Missing parameter 'arguments' or 'arguments' is not of type 'str'"
    //                 .into(),
    //             raw_output: raw_output.into(),
    //         })?;
    //     let arguments = serde_json::from_str::<IndexMap<String, serde_json::Value>>(arguments_str)
    //         .map_err(|e| EvaluationError::JsonDecodeError {
    //             error_message: e.to_string(),
    //             raw_output: raw_output.into(),
    //         })?;
    //     Ok(Gpt5OutputFunctionCall { name, arguments })
    // }
}

#[derive(Copy, Clone)]
pub struct Gpt5Interface;

impl Gpt5Interface {
    pub fn map_type_hint(ty: &str) -> String {
        match ty {
            "dict" => "object".to_string(),
            "float" => "number".to_string(),
            "tuple" => "array".to_string(),
            _ => ty.to_string(),
        }
    }
    pub fn sanitize_and_convert_function_format(
        bfcl_functions: &Vec<BfclFunctionDef>,
        name_mapper: &mut FunctionNameMapper,
    ) -> Vec<Gpt5Tool> {
        let sanitized_bfcl_functions = name_mapper.map_function_names(bfcl_functions);
        let mut gpt5_tools = Vec::new();
        for bfcl_func in &sanitized_bfcl_functions {
            let bfcl_param = &bfcl_func.parameters;
            let gpt5_params = bfcl_param_to_gpt5_param(bfcl_param, true);
            let description = bfcl_func.description.clone();
            gpt5_tools.push(Gpt5Tool {
                ty: "function".to_string(),
                name: bfcl_func.name.clone(),
                description,
                parameters: gpt5_params,
                strict: false,
            });
        }
        gpt5_tools
    }
}

fn bfcl_param_to_gpt5_param(bfcl_parameter: &BfclParameter, required: bool) -> Gpt5Parameter {
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
    let gpt5_type = Gpt5Interface::map_type_hint(bfcl_type);
    // see https://platform.openai.com/docs/guides/function-calling#strict-mode
    let gpt5_type = match required {
        true => SingleOrList::Single(gpt5_type),
        false => SingleOrList::List(vec![gpt5_type, "null".to_string()]),
    };
    let bfcl_required = bfcl_required.as_ref();
    let gpt5_properties = bfcl_properties.as_ref().map(|props| {
        let mut gpt5_props = IndexMap::new();
        for (prop_name, prop_value) in props.iter() {
            let sub_required = bfcl_required.map_or(false, |reqs| reqs.contains(prop_name));
            let gpt5_prop_value = bfcl_param_to_gpt5_param(prop_value, sub_required);
            gpt5_props.insert(prop_name.clone(), gpt5_prop_value);
        }
        gpt5_props
    });

    let gpt5_items = bfcl_items.as_ref().map(|item| {
        let gpt5_item = bfcl_param_to_gpt5_param(item, true); // array items are always required
        Box::new(gpt5_item)
    });
    let gpt5_enum = bfcl_enum.clone();
    // in strict mode, all parameters must be set to be required
    // but strict mode is not compatible with some of the optional parameters in BFCL
    // so we keep the strict mode format but does not fully follow strict requirement
    let gpt5_required: Option<Vec<String>> = bfcl_properties
        .as_ref()
        .map(|props| props.keys().cloned().collect());
    Gpt5Parameter {
        ty: gpt5_type,
        properties: gpt5_properties,
        description: bfcl_description.clone(),
        items: gpt5_items,
        r#enum: gpt5_enum,
        required: gpt5_required,
        default: bfcl_default.clone(),
        // additional_properties: false, // in strict mode, additional properties are not allowed
        format: bfcl_format.clone(),
        maximum: bfcl_maximum.clone(),
    }
}

// Sample tool format for GPT-5
// tools = [
//     {
//         "type": "function",
//         "name": "get_horoscope",
//         "description": "Get today's horoscope for an astrological sign.",
//         "parameters": {
//             "type": "object",
//             "properties": {
//                 "sign": {
//                     "type": "string",
//                     "description": "An astrological sign like Taurus or Aquarius",
//                 },
//             },
//             "required": ["sign"],
//         },
//     },
// ]

#[async_trait::async_trait]
impl ModelInterface for Gpt5Interface {
    async fn generate_tool_call_async(
        &self,
        backend: Arc<ModelBackend>,
        raw_functions: Vec<BfclFunctionDef>,
        user_question: String,
        prompt_passing_in_english: bool,
        name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    ) -> String {
        // downcast backend to api backend
        let api_backend = (backend.as_ref() as &dyn Any)
            .downcast_ref::<ApiBackend>()
            .expect("Failed to downcast to ApiBackend");
        let client = &api_backend.client;
        let gpt5_tools = {
            let mut name_mapper_borrow = name_mapper.borrow_mut();
            Gpt5Interface::sanitize_and_convert_function_format(
                &raw_functions,
                &mut *name_mapper_borrow,
            )
        };
        // println!("{}", serde_json::to_string(&gpt5_tools).unwrap());
        let gpt5_tools_serialized =
            serde_json::to_string(&gpt5_tools).expect("Failed to serialize GPT-5 tools");

        // println!("{}", gpt5_tools_serialized);

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let generate_tool_call_async_fn = gpt5_backend_module
                .getattr("generate_tool_call_async")
                .expect("Failed to get generate_tool_call_async function");
            let model_name = api_backend.model.to_string();
            let json = py.import("json").expect("failed to import json");
            let gpt5_tools_obj = json
                .call_method("loads", (gpt5_tools_serialized,), None)
                .expect("Failed to parse GPT-5 tools JSON");
            assert!(gpt5_tools_obj.is_instance_of::<PyList>());
            let arguments = (
                model_name,
                client,
                user_question,
                gpt5_tools_obj,
                prompt_passing_in_english,
            );
            let fut = generate_tool_call_async_fn
                .call1(arguments)
                .expect("Failed to call generate_tool_call_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("GPT-5 tool call generation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }

    async fn translate_tool_question_async(
        &self,
        backend: Arc<ModelBackend>,
        user_question: String,
    ) -> String {
        // downcast backend to api backend
        // let api_backend = (backend.as_ref() as &dyn Any)
        //     .downcast_ref::<ApiBackend>()
        //     .expect("Failed to downcast to ApiBackend");
        let ModelBackend::Api(api_backend) = backend.as_ref() else {
            panic!("gpt5 interface should use ApiBackend");
        };
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let translate_tool_question_async_fn = gpt5_backend_module
                .getattr("translate_tool_question_async")
                .expect("Failed to get translate_tool_question_async function");
            let model_name = api_backend.model.to_string();
            let arguments = (model_name, client, user_question);
            let fut = translate_tool_question_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_question_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("GPT-5 tool question translation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }
    fn postprocess_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        let output_json = serde_json::from_str::<Value>(raw_output).map_err(|e| {
            EvaluationError::JsonDecodeError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            }
        })?;
        let output_list = output_json
            .as_array()
            .ok_or(EvaluationError::ParsingError {
                error_message: "Expected JSON array but got different format".into(),
                raw_output: raw_output.to_string(),
            })?;
        let mut func_calls = Vec::new();
        for potential_func_call in output_list {
            let json_obj =
                potential_func_call
                    .as_object()
                    .ok_or(EvaluationError::ParsingError {
                        error_message: "Expected JSON object for function call".into(),
                        raw_output: raw_output.to_string(),
                    })?;
            let ty = json_obj.get("type").and_then(|v| v.as_str()).ok_or(
                EvaluationError::ParsingError {
                    error_message: "Missing 'type' field in function call".into(),
                    raw_output: raw_output.to_string(),
                },
            )?;
            if ty != "function_call" {
                // if ty != "reasoning" {
                //     println!("Warning: Gpt5 outputs an item with unexpected type: {}", ty);
                // }
                continue; // skip non-function_call entries
            }

            let func_call= parse_gpt5_function_call(potential_func_call).map_err(|e| {
                    EvaluationError::ParsingError {
                        error_message: format!("Failed to parse Gpt5OutputFunctionCall: {}", e),
                        raw_output: raw_output.to_string(),
                    }
                })?;
            let original_function_name = {
                let name_mapper_borrow = name_mapper.borrow();
                name_mapper_borrow.get_original_name(&func_call.name)
            };
            let bfcl_output_function_call = BfclOutputFunctionCall(KeyValuePair {
                key: original_function_name,
                value: func_call.arguments,
            });
            func_calls.push(bfcl_output_function_call);
        }
        Ok(func_calls)
    }
    async fn translate_tool_answer_async(
        &self,
        backend: Arc<ModelBackend>,
        parameter_value: String,
    ) -> String {
        // downcast backend to api backend
        let ModelBackend::Api(api_backend) = backend.as_ref() else {
            panic!("gpt5 interface should use ApiBackend");
        };
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let translate_tool_answer_async_fn = gpt5_backend_module
                .getattr("translate_tool_answer_async")
                .expect("Failed to get translate_tool_answer_async function");
            let model_name = api_backend.model.to_string();
            let arguments = (model_name, client, parameter_value);
            let fut = translate_tool_answer_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_answer_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("GPT-5 tool answer translation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }
}

fn parse_gpt5_function_call(
    function_call: &serde_json::Value,
) -> Result<Gpt5OutputFunctionCall, String> {
    let mut function_call = function_call.clone();
    // retrieve the "arguments" field
    let arguments_value = function_call
        .get("arguments")
        .ok_or("Missing 'arguments' field in function call")?;
    let arguments_str = arguments_value
        .as_str()
        .ok_or("'arguments' field is not a string")?;
    // parse the arguments string as JSON
    let arguments_json: serde_json::Value = serde_json::from_str(arguments_str)
        .map_err(|e| format!("Failed to parse 'arguments' JSON string: {}", e))?;
    // replace the "arguments" field with the parsed JSON object
    function_call["arguments"] = arguments_json;
    // deserialize the modified function call into Gpt5OutputFunctionCall
    let gpt5_output_function_call: Gpt5OutputFunctionCall = serde_json::from_value(function_call)
        .map_err(|e| format!("Failed to deserialize Gpt5OutputFunctionCall: {}", e))?;
    Ok(gpt5_output_function_call)
}

// def postprocess_tool_calls(
//         self,
//         raw_output: str,
//         name_mapper: Optional['FunctionNameMapper'] = None
//     ) -> Union[List[Dict[str, Dict[str, Any]]], Tuple[EvaluationError, Dict[str, Any]]]:
//         """
//         Postprocess raw output from GPT-5's structured response format.

//         Args:
//             raw_output: Raw JSON string output from the model
//             name_mapper: External name mapper to convert sanitized names back to original

//         Returns:
//             On success: List of function calls
//             On error: Tuple of (EvaluationError, metadata dict with error details)
//         """
//         try:
//             # Parse the JSON response
//             response_data = json.loads(raw_output)

//             # Handle case where response_data is a list (new format)
//             if isinstance(response_data, list):
//                 function_calls = response_data
//             # Handle case where response_data is a dict with error
//             elif isinstance(response_data, dict) and "error" in response_data:
//                 print("Error: Unreachable! gpt5_interface line 236")
//                 exit(1)
//                 # return (EvaluationError.MODEL_ERROR, {
//                 #     "error_message": response_data['error'],
//                 #     "raw_output": raw_output
//                 # })
//             # Handle case where response_data is a dict with function_calls (old format)
//             elif isinstance(response_data, dict) and "function_calls" in response_data:
//                 function_calls = response_data["function_calls"]
//             # Fallback: no function calls
//             elif isinstance(response_data, dict) and "output_text" in response_data:
//                 return (EvaluationError.NO_FUNCTION_CALLS_FOUND, {
//                     "raw_output": raw_output
//                 })
//                 # return (EvaluationError.TEXT_INSTEAD_OF_FUNCTION_CALLS, {
//                 #     "output_text": str(response_data['output_text'])[:200],
//                 #     "raw_output": raw_output
//                 # })
//             else:
//                 print("Error: Unreachable! gpt5_interface line 254")
//                 exit(1)
//                 # return (EvaluationError.UNEXPECTED_RESPONSE_FORMAT, {
//                 #     "response_preview": json.dumps(response_data)[:200],
//                 #     "raw_output": raw_output
//                 # })

//             # Convert function calls to standard format
//             extracted = []
//             for func_call in function_calls:
//                 if func_call.get("type") == "function_call":
//                     sanitized_name = func_call.get("name")
//                     arguments = func_call.get("arguments", {})

//                     # Parse arguments if they come as a JSON string
//                     if isinstance(arguments, str):
//                         arguments = json.loads(arguments)
//                     if sanitized_name:
//                         # Convert sanitized name back to original using external name_mapper
//                         if name_mapper:
//                             original_name = name_mapper.get_original_name(sanitized_name)
//                         else:
//                             # Fallback: if no name_mapper, assume no sanitization was done
//                             original_name = sanitized_name

//                         # Convert to standard format
//                         extracted.append({original_name: arguments})

//             if extracted:
//                 return extracted
//             else:
//                 return (EvaluationError.NO_FUNCTION_CALLS_FOUND, {
//                     "raw_output": raw_output
//                 })

//         except json.JSONDecodeError as e:
//             return (EvaluationError.JSON_DECODE_ERROR, {
//                 "error_message": str(e),
//                 "raw_output": raw_output
//             })
//         except Exception as e:
//             return (EvaluationError.PARSING_ERROR, {
//                 "error_message": str(e),
//                 "exception_type": type(e).__name__,
//                 "raw_output": raw_output
//             })
