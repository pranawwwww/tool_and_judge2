use crate::{
    models::{ function_name_mapper::FunctionNameMapper,
        model_interface::ModelInterface,
    },
    one_entry_map::KeyValuePair,
    single_or_list::SingleOrList,
    tool::bfcl_formats::{BfclFunctionDef, BfclOutputFunctionCall, BfclParameter},
    tool::error_analysis::EvaluationError,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// DeepSeek output function call structure (OpenAI-compatible)
#[derive(Deserialize, Clone)]
pub struct DeepSeekOutputFunctionCall {
    name: String,
    arguments: IndexMap<String, serde_json::Value>,
}

/// DeepSeek uses native function calling API with structured responses
/// Reference: https://api-docs.deepseek.com/guides/function_calling
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

    fn parse_tool_calls(
        &self,
        raw_output: &str,
        name_mapper: &FunctionNameMapper,
    ) -> Result<Vec<BfclOutputFunctionCall>, EvaluationError> {
        // DeepSeek now uses structured JSON format (OpenAI-compatible)
        // The raw_output is a JSON object with tool_calls array
        let output_json = serde_json::from_str::<Value>(raw_output).map_err(|e| {
            EvaluationError::JsonDecodeError {
                error_message: e.to_string(),
                raw_output: raw_output.to_string(),
            }
        })?;

        // Extract tool_calls array from the message object
        let tool_calls = output_json
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .ok_or(EvaluationError::ParsingError {
                error_message: "Expected 'tool_calls' array in response".into(),
                raw_output: raw_output.to_string(),
            })?;

        let mut func_calls = Vec::new();
        for tool_call in tool_calls {
            let tool_type = tool_call
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or(EvaluationError::ParsingError {
                    error_message: "Missing 'type' field in tool call".into(),
                    raw_output: raw_output.to_string(),
                })?;

            if tool_type != "function" {
                continue; // skip non-function tool calls
            }

            let function_obj = tool_call
                .get("function")
                .ok_or(EvaluationError::ParsingError {
                    error_message: "Missing 'function' field in tool call".into(),
                    raw_output: raw_output.to_string(),
                })?;

            let func_call = parse_deepseek_function_call(function_obj).map_err(|e| {
                EvaluationError::ParsingError {
                    error_message: format!("Failed to parse DeepSeekOutputFunctionCall: {}", e),
                    raw_output: raw_output.to_string(),
                }
            })?;

            let original_function_name = name_mapper
                .sanitized_to_original
                .get(&func_call.name)
                .ok_or_else(|| EvaluationError::ParsingError {
                    error_message: format!("Function name '{}' not found in mapper", func_call.name),
                    raw_output: raw_output.to_string(),
                })?
                .clone();

            let bfcl_output_function_call = BfclOutputFunctionCall(KeyValuePair {
                key: original_function_name,
                value: func_call.arguments,
            });
            func_calls.push(bfcl_output_function_call);
        }

        Ok(func_calls)
    }
}

fn parse_deepseek_function_call(
    function_obj: &serde_json::Value,
) -> Result<DeepSeekOutputFunctionCall, String> {
    let mut function_call = function_obj.clone();

    // Retrieve the "arguments" field (it's a JSON string, not an object)
    let arguments_value = function_call
        .get("arguments")
        .ok_or("Missing 'arguments' field in function call")?;
    let arguments_str = arguments_value
        .as_str()
        .ok_or("'arguments' field is not a string")?;

    // Parse the arguments string as JSON
    let arguments_json: serde_json::Value = serde_json::from_str(arguments_str)
        .map_err(|e| format!("Failed to parse 'arguments' JSON string: {}", e))?;

    // Replace the "arguments" field with the parsed JSON object
    function_call["arguments"] = arguments_json;

    // Deserialize the modified function call into DeepSeekOutputFunctionCall
    let deepseek_output_function_call: DeepSeekOutputFunctionCall =
        serde_json::from_value(function_call)
            .map_err(|e| format!("Failed to deserialize DeepSeekOutputFunctionCall: {}", e))?;

    Ok(deepseek_output_function_call)
}
