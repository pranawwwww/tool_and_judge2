use std::collections::HashMap;

use indexmap::IndexMap;
use serde_json::json;

#[derive(Clone)]
pub struct BfclDatasetEntry {
    pub id: String,
    pub question_content: String,
    pub functions: Vec<BfclFunctionDef>,
    pub raw_entry: serde_json::Value,
}

impl BfclDatasetEntry {
    pub fn deserialize_from_json(raw_entry: serde_json::Value) -> Result<Self, String> {
        let id = raw_entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'id' field")?
            .to_string();

        let question_content = raw_entry
            .get("question")
            .and_then(|v| v.get(0))
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("content"))
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'question[0][0].content' field")?
            .to_string();

        let functions_array = raw_entry
            .get("function")
            .and_then(|v| v.as_array())
            .ok_or("Missing or invalid 'function' field")?;

        let mut functions = Vec::new();
        for func_val in functions_array {
            let name = func_val
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid 'name' field in function")?
                .to_string();

            let description = func_val
                .get("description")
                .and_then(|v| v.as_str())
                .ok_or("Missing or invalid 'description' field in function")?
                .to_string();

            let params_obj = func_val
                .get("parameters")
                .and_then(|v| v.get("properties"))
                .and_then(|v| v.as_object())
                .ok_or("Missing or invalid 'parameters.properties' field")?;

            let required_array = func_val
                .get("parameters")
                .and_then(|v| v.get("required"))
                .and_then(|v| v.as_array())
                .ok_or("Missing or invalid 'parameters.required' field")?;

            let required = required_array
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>();

            let mut parameters = Vec::new();
            for (param_name, param_val) in params_obj {
                let param_type = param_val
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing or invalid 'type' field in parameter")?
                    .to_string();

                let param_description = param_val
                    .get("description")
                    .and_then(|v| v.as_str())
                    .ok_or("Missing or invalid 'description' field in parameter")?
                    .to_string();
                let items_ty = match param_val.get("items") {
                    Some(items_val) => {
                        let items_obj = items_val
                            .as_object()
                            .ok_or("items field should be an object")?;
                        let items_type = items_obj
                            .get("type")
                            .and_then(|v| v.as_str())
                            .ok_or("Missing or invalid 'type' field in items")?
                            .to_string();
                        Some(items_type)
                    }
                    None => None,
                };

                parameters.push(BfclParameter {
                    name: param_name.clone(),
                    ty: param_type,
                    items_ty,
                    description: param_description,
                });
            }

            functions.push(BfclFunctionDef {
                name,
                description,
                parameters,
                required,
            });
        }

        Ok(BfclDatasetEntry {
            id,
            question_content,
            functions,
            raw_entry,
        })
    }
    pub fn modify_question_content(&mut self, new_content: &str) -> Result<(), String> {
        let raw_entry = &mut self.raw_entry;
        let question_array = raw_entry
            .get_mut("question")
            .and_then(|v| v.as_array_mut())
            .ok_or("Missing or invalid 'question' field")?;
        let first_question = question_array
            .get_mut(0)
            .and_then(|v| v.as_array_mut())
            .ok_or("Missing or invalid 'question[0]' field")?;
        let first_content = first_question
            .get_mut(0)
            .and_then(|v| v.as_object_mut())
            .ok_or("Missing or invalid 'question[0][0]' field")?;
        first_content.insert(
            "content".to_string(),
            serde_json::Value::String(new_content.to_string()),
        );
        self.question_content = new_content.to_string();
        Ok(())
    }
}

#[derive(Clone)]
pub struct BfclFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<BfclParameter>,
    pub required: Vec<String>,
}

#[derive(Clone)]
pub struct BfclParameter {
    pub name: String,
    pub ty: String,
    pub items_ty: Option<String>,
    pub description: String,
}

#[derive(Clone)]
pub struct BfclGroundTruthFunctionCall {
    pub function_name: String,
    pub parameters: IndexMap<String, Vec<serde_json::Value>>,
}

#[derive(Clone)]
pub struct BfclOutputFunctionCall {
    pub function_name: String,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}
impl BfclOutputFunctionCall {
    pub fn new(
        function_name: String,
        parameters: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        Self {
            function_name,
            parameters,
        }
    }
    pub fn serialize_to_json(self) -> serde_json::Value {
        let parameters_json = self
            .parameters
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        json![{
            self.function_name: serde_json::Value::Object(parameters_json)
        }]
    }
    pub fn deserialize_from_json(value: &serde_json::Value) -> Result<Self, String> {
        let obj = value.as_object().ok_or("Expected a JSON object")?;
        if obj.len() != 1 {
            return Err("Expected exactly one function call".to_string());
        }
        let (function_name, params_value) = obj.iter().next().unwrap();
        let params_obj = params_value
            .as_object()
            .ok_or("Expected parameters to be a JSON object")?;

        let mut parameters = serde_json::Map::new();
        for (param_name, param_value) in params_obj {
            parameters.insert(param_name.clone(), param_value.clone());
        }

        Ok(BfclOutputFunctionCall {
            function_name: function_name.clone(),
            parameters,
        })
    }
}

impl BfclGroundTruthFunctionCall {
    pub fn serialize_to_json(self) -> serde_json::Value {
        let parameters_json = self
            .parameters
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::Array(v)))
            .collect::<serde_json::Map<String, serde_json::Value>>();
        json![{
            self.function_name: serde_json::Value::Object(parameters_json)
        }]
    }
    pub fn deserialize_from_json(value: &serde_json::Value) -> Result<Self, String> {
        let obj = value.as_object().ok_or("Expected a JSON object")?;
        if obj.len() != 1 {
            return Err("Expected exactly one function call".to_string());
        }
        let (function_name, params_value) = obj.iter().next().unwrap();
        let params_obj = params_value
            .as_object()
            .ok_or("Expected parameters to be a JSON object")?;

        let mut parameters = IndexMap::new();
        for (param_name, param_value) in params_obj {
            let param_array = param_value
                .as_array()
                .ok_or("Expected parameter value to be a JSON array")?;
            parameters.insert(param_name.clone(), param_array.clone());
        }

        Ok(BfclGroundTruthFunctionCall {
            function_name: function_name.clone(),
            parameters,
        })
    }
}

#[derive(Clone)]
pub struct BfclGroundTruthEntry {
    pub id: String,
    pub ground_truth: Vec<BfclGroundTruthFunctionCall>,
}

impl BfclGroundTruthEntry {
    pub fn deserialize_from_json(raw_entry: serde_json::Value) -> Result<Self, String> {
        let id = raw_entry
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing or invalid 'id' field")?
            .to_string();

        let gt_array = raw_entry
            .get("ground_truth")
            .and_then(|v| v.as_array())
            .ok_or("Missing or invalid 'ground_truth' field")?;

        let mut ground_truth = Vec::new();
        for gt_val in gt_array {
            let function_call = BfclGroundTruthFunctionCall::deserialize_from_json(gt_val)?;
            ground_truth.push(function_call);
        }

        Ok(BfclGroundTruthEntry { id, ground_truth })
    }
}
// sample ground truth function call:
// {"triangle_properties.get": {"side1": [5], "side2": [4], "side3": [3], "get_area": ["", true], "get_perimeter": ["", true], "get_angles": ["", true]}}
