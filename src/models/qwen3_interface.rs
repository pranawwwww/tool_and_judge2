
use crate::{
    models::{function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    one_entry_map::KeyValuePair,
    tool::bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall, BfclParameter},
    tool::error_analysis::EvaluationError,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Qwen3 tool format following the Qwen documentation
/// https://qwen.readthedocs.io/en/latest/framework/function_call.html
#[derive(Serialize)]
pub struct Qwen3Tool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: Qwen3Function,
}

#[derive(Serialize)]
pub struct Qwen3Function {
    pub name: String,
    pub description: String,
    pub parameters: Qwen3Parameter,
}

/// JSON Schema for Qwen3 function parameters
#[derive(Serialize)]
pub struct Qwen3Parameter {
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<IndexMap<String, Qwen3Parameter>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Qwen3Parameter>>,
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
pub struct Qwen3OutputFunctionCall {
    name: String,
    arguments: String, // JSON string that needs parsing
}

#[derive(Copy, Clone)]
pub struct Qwen3Interface;

impl Qwen3Interface {
    pub fn map_type_hint(ty: &str) -> String {
        match ty {
            "dict" => "object".to_string(),
            "float" => "number".to_string(),
            "tuple" => "array".to_string(),
            _ => ty.to_string(),
        }
    }
}

fn bfcl_param_to_qwen3_param(bfcl_parameter: &BfclParameter) -> Qwen3Parameter {
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

    let qwen3_type = Qwen3Interface::map_type_hint(bfcl_type);
    let bfcl_required = bfcl_required.as_ref();

    let qwen3_properties = bfcl_properties.as_ref().map(|props| {
        let mut qwen3_props = IndexMap::new();
        for (prop_name, prop_value) in props.iter() {
            let qwen3_prop_value = bfcl_param_to_qwen3_param(prop_value);
            qwen3_props.insert(prop_name.clone(), qwen3_prop_value);
        }
        qwen3_props
    });

    let qwen3_items = bfcl_items.as_ref().map(|item| {
        let qwen3_item = bfcl_param_to_qwen3_param(item);
        Box::new(qwen3_item)
    });

    let qwen3_enum = bfcl_enum.clone();
    let qwen3_required: Option<Vec<String>> = bfcl_required.map(|reqs| reqs.to_vec());

    Qwen3Parameter {
        ty: qwen3_type,
        properties: qwen3_properties,
        description: bfcl_description.clone(),
        items: qwen3_items,
        r#enum: qwen3_enum,
        required: qwen3_required,
        default: bfcl_default.clone(),
        format: bfcl_format.clone(),
        maximum: bfcl_maximum.clone(),
    }
}

#[async_trait::async_trait]
impl ModelInterface for Qwen3Interface {
    fn generate_tool_definitions(
        &self,
        bfcl_functions: &Vec<BfclFunctionDef>,
        name_mapper: &FunctionNameMapper,
    ) -> serde_json::Value {
        let mut qwen3_tools = Vec::new();
        for bfcl_func in bfcl_functions.iter() {
            let sanitized_name = name_mapper
                .original_to_sanitized
                .get(&bfcl_func.name)
                .expect("Function name mapper does not contain key")
                .clone();
            let bfcl_param = &bfcl_func.parameters;
            let qwen3_params = bfcl_param_to_qwen3_param(bfcl_param);
            let description = bfcl_func.description.clone();
            qwen3_tools.push(Qwen3Tool {
                ty: "function".to_string(),
                function: Qwen3Function {
                    name: sanitized_name,
                    description,
                    parameters: qwen3_params,
                },
            });
        }
        serde_json::to_value(qwen3_tools).expect("Failed to serialize Qwen3 tools")
    }

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        let stripped_output = raw_output.strip_prefix("<tool_call>").unwrap_or(raw_output);
        let stripped_output = stripped_output.strip_suffix("</tool_call>").unwrap_or(stripped_output);
        let stripped_output = stripped_output.trim();
        // Parse the raw output as JSON
        let output_json = serde_json::from_str::<serde_json::Value>(stripped_output).map_err(|e| {
            EvaluationError::JsonDecodeError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            }
        })?;

        // Check if it's a single function call or an array
        let qwen3_calls: Vec<Qwen3OutputFunctionCall> = if output_json.is_array() {
            // Parse as array of function calls
            serde_json::from_value(output_json).map_err(|e| EvaluationError::ParsingError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            })?
        } else if output_json.is_object() {
            // Try to parse as single function call
            let single_call: Qwen3OutputFunctionCall =
                serde_json::from_value(output_json).map_err(|e| EvaluationError::ParsingError {
                    error_message: e.to_string(),
                    raw_output: raw_output.to_string(),
                })?;
            vec![single_call]
        } else {
            return Err(EvaluationError::ParsingError {
                error_message: "Expected JSON object or array".to_string(),
                raw_output: raw_output.to_string(),
            });
        };

        // Convert Qwen3 format to BFCL format
        let mut bfcl_calls = Vec::new();
        for qwen3_call in qwen3_calls {
            // Parse the arguments string as JSON
            let arguments: IndexMap<String, serde_json::Value> =
                serde_json::from_str(&qwen3_call.arguments).map_err(|e| {
                    EvaluationError::JsonDecodeError {
                        error_message: format!(
                            "Failed to parse arguments JSON string: {}",
                            e
                        ),
                        raw_output: raw_output.to_string(),
                    }
                })?;

            // Map the function name back to original
            let original_name = name_mapper
                .sanitized_to_original
                .get(&qwen3_call.name)
                .expect("Function name mapper does not contain key")
                .clone();

            bfcl_calls.push(BfclOutputFunctionCall(KeyValuePair {
                key: original_name,
                value: arguments,
            }));
        }

        Ok(bfcl_calls)
    }
}
