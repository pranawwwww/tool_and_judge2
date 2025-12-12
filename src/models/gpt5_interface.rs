use std::{any::Any, collections::HashMap};

use crate::{
    models::{
        api_backend::ApiBackend, backend::ModelBackend, function_name_mapper::FunctionNameMapper, model_interface::ModelInterface
    },
    tool_bfcl_decl::BfclFunctionDef,
};
use pyo3::{prelude::*, types::PyList};
use pyo3::{Py, PyAny, Python, types::PyAnyMethods};
use serde::Serialize;

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
        backend: &dyn ModelBackend,
        raw_functions: &Vec<BfclFunctionDef>,
        user_question: &str,
        prompt_passing_in_english: bool,
        name_mapper: &mut FunctionNameMapper,
    ) -> String {
        // downcast backend to api backend
        let api_backend = (backend as &dyn Any).downcast_ref::<ApiBackend>().expect("Failed to downcast to ApiBackend");
        let client = &api_backend.client;
        let gpt5_tools = Gpt5Interface::sanitize_and_convert_function_format(
            raw_functions,
            prompt_passing_in_english,
            name_mapper,
        );
        let gpt5_tools_serialized = serde_json::to_string(&gpt5_tools).expect("Failed to serialize GPT-5 tools");
        
        let fut = Python::attach(|py|{
            let gpt5_backend_module = py.import("src_py.gpt5_backend").expect("Failed to import src_py.gpt5_backend module");
            let generate_tool_call_async_fn = gpt5_backend_module.getattr("generate_tool_call_async").expect("Failed to get generate_tool_call_async function");
            let model_name = backend.get_model_info().to_string();
            let json = py.import("json").expect("failed to import json");
            let gpt5_tools_obj = json.call_method("loads", (gpt5_tools_serialized,), None).expect("Failed to parse GPT-5 tools JSON");
            assert!(gpt5_tools_obj.is_instance_of::<PyList>());
            let arguments = (
                model_name,
                client,
                user_question,
                gpt5_tools_obj,
                prompt_passing_in_english,
            );
            let fut = generate_tool_call_async_fn.call1(arguments).expect("Failed to call generate_tool_call_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("GPT-5 tool call generation failed");
        let response_str = Python::attach(|py| {
            response_str.extract::<String>(py).expect("Failed to extract response string")
        });
        response_str
    }

    async fn translate_tool_question_async(
        &self,
        backend: &dyn ModelBackend,
        user_question: &str,
    ) -> String {
        // downcast backend to api backend
        let api_backend = (backend as &dyn Any).downcast_ref::<ApiBackend>().expect("Failed to downcast to ApiBackend");
        let client = &api_backend.client;

        let fut = Python::attach(|py|{
            let gpt5_backend_module = py.import("src_py.gpt5_backend").expect("Failed to import src_py.gpt5_backend module");
            let translate_tool_question_async_fn = gpt5_backend_module.getattr("translate_tool_question_async").expect("Failed to get translate_tool_question_async function");
            let model_name = backend.get_model_info().to_string();
            let arguments = (
                model_name,
                client,
                user_question,
            );
            let fut = translate_tool_question_async_fn.call1(arguments).expect("Failed to call translate_tool_question_async");
            pyo3_async_runtimes::tokio::into_future(fut).expect("Failed to convert to Rust future")
        });
        let response_str = fut.await.expect("GPT-5 tool question translation failed");
        let response_str = Python::attach(|py| {
            response_str.extract::<String>(py).expect("Failed to extract response string")
        });
        response_str
    }
}
