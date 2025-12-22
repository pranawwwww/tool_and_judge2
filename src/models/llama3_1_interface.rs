use std::sync::Arc;

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
use indexmap::IndexMap;
use pyo3::{Python, types::PyAnyMethods};
use serde::{Deserialize, Serialize};

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
    parameters: IndexMap<String, serde_json::Value>,
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
}

/// See https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
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
    /// See https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    fn generate_tool_definitions(
        &self,
        bfcl_functions: &Vec<BfclFunctionDef>,
        name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value {
        let mut llama3_1_tools = Vec::new();
        for bfcl_func in bfcl_functions.iter() {
            let sanitized_name = name_mapper
                .original_to_sanitized
                .get(&bfcl_func.name)
                .expect("Function name mapper does not contain key")
                .clone();
            let bfcl_param = &bfcl_func.parameters;
            let llama3_1_params = bfcl_param_to_llama3_1_param(bfcl_param);
            let description = bfcl_func.description.clone();
            llama3_1_tools.push(Llama3_1Tool {
                ty: "function".to_string(),
                function: Llama3_1Function {
                    name: sanitized_name,
                    description,
                    parameters: llama3_1_params,
                },
            });
        }
        serde_json::to_value(llama3_1_tools).expect("Failed to serialize Llama 3.1 tools")
    }

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        // convert string to json value
        let output_json = serde_json::from_str::<serde_json::Value>(raw_output).map_err(|e| {
            EvaluationError::JsonDecodeError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            }
        })?;
        // directly parse json to llama3.1 output function call struct
        let parsed_output: SingleOrList<Llama3_1OutputFunctionCall> =
            serde_json::from_value(output_json).map_err(|e| EvaluationError::ParsingError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            })?;
        // unify to a list
        let parsed_output_vec = match parsed_output {
            SingleOrList::Single(item) => vec![item],
            SingleOrList::List(items) => items,
        };
        // parse from llama3.1 format to bfcl format
        // let name_mapper_borrow = name_mapper.borrow();
        let bfcl_calls: Vec<BfclOutputFunctionCall> = parsed_output_vec
            .into_iter()
            .map(|llama3_1_call| llama3_1_call_to_bfcl_call(llama3_1_call, name_mapper))
            .collect();
        Ok(bfcl_calls)
    }   
}
fn llama3_1_call_to_bfcl_call(
    llama3_1_call: Llama3_1OutputFunctionCall,
    name_mapper: &FunctionNameMapper,
) -> BfclOutputFunctionCall {
    // let mapped_name = name_mapper.get_original_name(&llama3_1_call.name);
    let mapped_name = name_mapper
        .sanitized_to_original
        .get(&llama3_1_call.name)
        .expect("Function name mapper does not contain key")
        .clone();
    BfclOutputFunctionCall(KeyValuePair {
        key: mapped_name,
        value: llama3_1_call.parameters,
    })
}
