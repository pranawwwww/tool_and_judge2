use std::{any::Any, collections::HashMap, sync::Arc};

use crate::{
    models::{
        api_backend::ApiBackend, backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    tool_bfcl_decl::BfclFunctionDef, tool_error_analysis::EvaluationError, tool_file_models::ToolCallParsingResult,
};
use atomic_refcell::AtomicRefCell;
use pyo3::{Py, PyAny, Python, types::PyAnyMethods};
use pyo3::{prelude::*, types::PyList};
use serde::Serialize;

use serde_json::Value;
use serde_json::json;

#[derive(Serialize)]
pub struct Gpt5Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
    pub description: String,
    pub parameters: Gpt5Parameters,
}

#[derive(Serialize)]
pub struct Gpt5Parameters {
    #[serde(rename = "type")]
    pub ty: String,
    properties: HashMap<String, Gpt5PropertyValue>,
    required: Vec<String>,
}

#[derive(Serialize)]
pub struct Gpt5PropertyValue {
    #[serde(rename = "type")]
    pub ty: String,
    pub description: String,
}

#[derive(Copy, Clone)]
pub struct Gpt5Interface;

impl Gpt5Interface {
    fn sanitize_and_convert_function_format(
        functions: &Vec<BfclFunctionDef>,
        prompt_passing_in_english: bool,
        name_mapper: &mut FunctionNameMapper,
    ) -> Vec<Gpt5Tool> {
        let sanitized_functions = name_mapper.map_function_names(functions);
        let mut gpt5_tools = Vec::new();
        for func in &sanitized_functions {
            let mut properties = HashMap::new();
            let required = func.required.clone();
            for param in &func.parameters {
                properties.insert(
                    param.name.clone(),
                    Gpt5PropertyValue {
                        ty: param.ty.clone(),
                        description: param.description.clone(),
                    },
                );
            }
            let description = if prompt_passing_in_english {
                format!(
                    "{} (IMPORTANT: PASS PARAMETER VALUES IN ENGLISH!)",
                    func.description
                )
            } else {
                func.description.clone()
            };
            gpt5_tools.push(Gpt5Tool {
                ty: "function".to_string(),
                name: func.name.clone(),
                description,
                parameters: Gpt5Parameters {
                    ty: "object".to_string(),
                    properties,
                    required,
                },
            });
        }
        gpt5_tools
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
        backend: Arc<dyn ModelBackend>,
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
                prompt_passing_in_english,
                &mut *name_mapper_borrow,
            )
        };
        let gpt5_tools_serialized =
            serde_json::to_string(&gpt5_tools).expect("Failed to serialize GPT-5 tools");

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let generate_tool_call_async_fn = gpt5_backend_module
                .getattr("generate_tool_call_async")
                .expect("Failed to get generate_tool_call_async function");
            let model_name = backend.get_model_info().to_string();
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
        backend: Arc<dyn ModelBackend>,
        user_question: String,
    ) -> String {
        // downcast backend to api backend
        let api_backend = (backend.as_ref() as &dyn Any)
            .downcast_ref::<ApiBackend>()
            .expect("Failed to downcast to ApiBackend");
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let translate_tool_question_async_fn = gpt5_backend_module
                .getattr("translate_tool_question_async")
                .expect("Failed to get translate_tool_question_async function");
            let model_name = backend.get_model_info().to_string();
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
    ) -> ToolCallParsingResult {
        let output_json = serde_json::from_str::<Value>(raw_output);
        
        // match serde_json::from_str::<Value>(raw_output) {
        //     Ok(response_data) => {
        //         let function_calls = if response_data.is_array() {
        //             response_data.as_array().unwrap().clone()
        //         } else if response_data.is_object() && response_data.get("function_calls").is_some()
        //         {
        //             response_data
        //                 .get("function_calls")
        //                 .unwrap()
        //                 .as_array()
        //                 .unwrap()
        //                 .clone()
        //         } else {
        //             return ToolCallParsingResult::Failure(
        //                 EvaluationError::NoFunctionCallsFound { raw_output: raw_output.to_string() },
        //             );
        //         };

        //         let mut extracted = Vec::new();
        //         for func_call in function_calls {
        //             if func_call.get("type") == Some(&json!("function_call")) {
        //                 let sanitized_name = func_call.get("name").and_then(|v| v.as_str());
        //                 let arguments = func_call.get("arguments").cloned().unwrap_or(json!({}));

        //                 if let Some(sanitized_name) = sanitized_name {
        //                     let original_name = {
        //                         let name_mapper_borrow = name_mapper.borrow();
        //                         name_mapper_borrow
        //                             .sanitized_to_original
        //                             .get(sanitized_name)
        //                             .cloned()
        //                             .unwrap_or(sanitized_name.to_string())
        //                     };
        //                     extracted.push(json!({original_name: arguments}));
        //                 }
        //             }
        //         }

        //         if !extracted.is_empty() {
        //             ToolCallParsingResult::Success(extracted)
        //         } else {
        //             ToolCallParsingResult::Failure(EvaluationError::NoFunctionCallsFound { raw_output: () })
        //         }
        //     }
        //     Err(_) => ToolCallParsingResult::Failure(EvaluationError::JsonDecodeError { error_message: (), raw_output: () }),
        // }
        todo!()
    }
    async fn translate_tool_answer_async(
        &self,
        backend: Arc<dyn ModelBackend>,
        parameter_value: String,
    ) -> String {
        // downcast backend to api backend
        let api_backend = (backend.as_ref() as &dyn Any)
            .downcast_ref::<ApiBackend>()
            .expect("Failed to downcast to ApiBackend");
        let client = &api_backend.client;

        let fut = Python::attach(|py| {
            let gpt5_backend_module = py
                .import("src_py.gpt5_backend")
                .expect("Failed to import src_py.gpt5_backend module");
            let translate_tool_answer_async_fn = gpt5_backend_module
                .getattr("translate_tool_answer_async")
                .expect("Failed to get translate_tool_answer_async function");
            let model_name = backend.get_model_info().to_string();
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
