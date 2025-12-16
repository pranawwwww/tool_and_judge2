use std::{any::Any, sync::Arc};

use crate::{
    models::{
        backend::ModelBackend, function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface, vllm_backend::VllmBackend,
    },
    one_entry_map::KeyValuePair,
    tool_bfcl_formats::{
        BfclFunctionDef, BfclOutputFunctionCall, BfclParameter,
    },
    tool_error_analysis::EvaluationError,
};
use atomic_refcell::AtomicRefCell;
use indexmap::IndexMap;
use pyo3::{Python, types::PyAnyMethods};
use pyo3::{prelude::*, types::PyList};
use serde::{Deserialize, Serialize};

use serde_json::Value;

/// Llama 3.1 tool format following the Transformers chat template convention
/// https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct#tool-use-with-transformers
#[derive(Serialize)]
pub struct Llama3_1Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: Llama3_1Function,
}

#[derive(Serialize)]
pub struct Llama3_1Function {
    pub name: String,
    pub description: String,
    pub parameters: Llama3_1Parameter,
}

/// JSON Schema for Llama 3.1 function parameters
#[derive(Serialize)]
pub struct Llama3_1Parameter {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Llama3_1Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Llama3_1Parameter>>,
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

#[derive(Deserialize, Clone)]
pub struct Llama3_1OutputFunctionCall {
    name: String,
    arguments: IndexMap<String, serde_json::Value>,
}

#[derive(Copy, Clone)]
pub struct Llama3_1Interface;

impl Llama3_1Interface {
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
    ) -> Vec<Llama3_1Tool> {
        let sanitized_bfcl_functions = name_mapper.map_function_names(bfcl_functions);
        let mut llama3_1_tools = Vec::new();
        for bfcl_func in &sanitized_bfcl_functions {
            let bfcl_param = &bfcl_func.parameters;
            let llama3_1_params = bfcl_param_to_llama3_1_param(bfcl_param);
            let description = bfcl_func.description.clone();
            llama3_1_tools.push(Llama3_1Tool {
                ty: "function".to_string(),
                function: Llama3_1Function {
                    name: bfcl_func.name.clone(),
                    description,
                    parameters: llama3_1_params,
                },
            });
        }
        llama3_1_tools
    }
}

fn bfcl_param_to_llama3_1_param(bfcl_parameter: &BfclParameter) -> Llama3_1Parameter {
    let BfclParameter {
        ty: bfcl_type,
        properties: bfcl_properties,
        items: bfcl_items,
        r#enum: bfcl_enum,
        description: bfcl_description,
        format: bfcl_format,
        required: bfcl_required,
        default: bfcl_default,
        optional: _,
        maximum: bfcl_maximum,
    } = bfcl_parameter;

    let llama3_1_type = Llama3_1Interface::map_type_hint(bfcl_type);
    let bfcl_required = bfcl_required.as_ref();

    let llama3_1_properties = bfcl_properties.as_ref().map(|props| {
        let mut llama3_1_props = IndexMap::new();
        for (prop_name, prop_value) in props.iter() {
            let llama3_1_prop_value = bfcl_param_to_llama3_1_param(prop_value);
            llama3_1_props.insert(prop_name.clone(), llama3_1_prop_value);
        }
        llama3_1_props
    });

    let llama3_1_items = bfcl_items.as_ref().map(|item| {
        let llama3_1_item = bfcl_param_to_llama3_1_param(item);
        Box::new(llama3_1_item)
    });

    let llama3_1_enum = bfcl_enum.clone();
    let llama3_1_required: Option<Vec<String>> = bfcl_required.map(|reqs| reqs.to_vec());

    Llama3_1Parameter {
        ty: llama3_1_type,
        properties: llama3_1_properties,
        description: bfcl_description.clone(),
        items: llama3_1_items,
        r#enum: llama3_1_enum,
        required: llama3_1_required,
        default: bfcl_default.clone(),
        format: bfcl_format.clone(),
        maximum: bfcl_maximum.clone(),
    }
}

#[async_trait::async_trait]
impl ModelInterface for Llama3_1Interface {
    async fn generate_tool_call_async(
        &self,
        backend: Arc<ModelBackend>,
        raw_functions: Vec<BfclFunctionDef>,
        user_question: String,
        prompt_passing_in_english: bool,
        name_mapper: Arc<AtomicRefCell<FunctionNameMapper>>,
    ) -> String {
        // downcast backend to vllm backend
        // let vllm_backend = (backend.as_ref() as &dyn Any)
        //     .downcast_ref::<VllmBackend>()
        //     .expect("Failed to downcast to VllmBackend");
        let ModelBackend::Vllm(vllm_backend) = backend.as_ref() else {
            panic!("Llama3_1 interface generate_tool_call_async should use VllmBackend");
        };
        let engine = &vllm_backend.engine;
        let tokenizer = &vllm_backend.tokenizer;

        let llama3_1_tools = {
            let mut name_mapper_borrow = name_mapper.borrow_mut();
            Llama3_1Interface::sanitize_and_convert_function_format(
                &raw_functions,
                &mut *name_mapper_borrow,
            )
        };

        let llama3_1_tools_serialized =
            serde_json::to_string(&llama3_1_tools).expect("Failed to serialize Llama 3.1 tools");

        let fut = Python::attach(|py| {
            let llama3_1_backend_module = py
                .import("src_py.llama3_1_backend")
                .expect("Failed to import src_py.llama3_1_backend module");
            let generate_tool_call_async_fn = llama3_1_backend_module
                .getattr("generate_tool_call_async")
                .expect("Failed to get generate_tool_call_async function");
            let model_name = vllm_backend.model.to_string();
            let json = py.import("json").expect("failed to import json");
            let llama3_1_tools_obj = json
                .call_method("loads", (llama3_1_tools_serialized,), None)
                .expect("Failed to parse Llama 3.1 tools JSON");
            assert!(llama3_1_tools_obj.is_instance_of::<PyList>());
            let arguments = (
                model_name,
                engine,
                tokenizer,
                user_question,
                llama3_1_tools_obj,
                prompt_passing_in_english,
            );
            let fut = generate_tool_call_async_fn
                .call1(arguments)
                .expect("Failed to call generate_tool_call_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("Llama 3.1 tool call generation failed");
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
        // downcast backend to vllm backend
        let ModelBackend::Vllm(vllm_backend) = backend.as_ref() else {
            panic!("Llama3_1 interface translate_tool_question_async should use VllmBackend");
        };
        let engine = &vllm_backend.engine;
        let tokenizer = &vllm_backend.tokenizer;

        let fut = Python::attach(|py| {
            let llama3_1_backend_module = py
                .import("src_py.llama3_1_backend")
                .expect("Failed to import src_py.llama3_1_backend module");
            let translate_tool_question_async_fn = llama3_1_backend_module
                .getattr("translate_tool_question_async")
                .expect("Failed to get translate_tool_question_async function");
            let model_name = vllm_backend.model.to_string();
            let arguments = (model_name, engine, tokenizer, user_question);
            let fut = translate_tool_question_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_question_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("Llama 3.1 tool question translation failed");
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
            if ty != "function" {
                continue; // skip non-function entries
            }

            let func_call = parse_llama3_1_function_call(potential_func_call).map_err(|e| {
                EvaluationError::ParsingError {
                    error_message: format!("Failed to parse Llama3_1OutputFunctionCall: {}", e),
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
        // downcast backend to vllm backend
        let ModelBackend::Vllm(vllm_backend) = backend.as_ref() else {
            panic!("Llama3_1 interface translate_tool_answer_async should use VllmBackend");
        };
        let engine = &vllm_backend.engine;
        let tokenizer = &vllm_backend.tokenizer;

        let fut = Python::attach(|py| {
            let llama3_1_backend_module = py
                .import("src_py.llama3_1_backend")
                .expect("Failed to import src_py.llama3_1_backend module");
            let translate_tool_answer_async_fn = llama3_1_backend_module
                .getattr("translate_tool_answer_async")
                .expect("Failed to get translate_tool_answer_async function");
            let model_name = vllm_backend.model.to_string();
            let arguments = (model_name, engine, tokenizer, parameter_value);
            let fut = translate_tool_answer_async_fn
                .call1(arguments)
                .expect("Failed to call translate_tool_answer_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("Llama 3.1 tool answer translation failed");
        let response_str = Python::attach(|py| {
            response_str
                .extract::<String>(py)
                .expect("Failed to extract response string")
        });
        response_str
    }
}

fn parse_llama3_1_function_call(
    function_call: &serde_json::Value,
) -> Result<Llama3_1OutputFunctionCall, String> {
    // let mut function_call = function_call.clone();

    // Llama 3.1 format: {"type": "function", "function": {"name": "...", "arguments": {...}}}
    let function_obj = function_call
        .get("function")
        .ok_or("Missing 'function' field in function call")?;

    let name = function_obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or("Missing 'name' field in function object")?
        .to_string();

    let arguments_value = function_obj
        .get("arguments")
        .ok_or("Missing 'arguments' field in function object")?;

    // Arguments might be a string or already an object
    let arguments_json = if arguments_value.is_string() {
        let arguments_str = arguments_value
            .as_str()
            .ok_or("'arguments' field is not a string")?;
        serde_json::from_str::<serde_json::Value>(arguments_str)
            .map_err(|e| format!("Failed to parse 'arguments' JSON string: {}", e))?
    } else {
        arguments_value.clone()
    };

    let arguments: IndexMap<String, serde_json::Value> = serde_json::from_value(arguments_json)
        .map_err(|e| format!("Failed to deserialize arguments: {}", e))?;

    Ok(Llama3_1OutputFunctionCall { name, arguments })
}
