"""
DeepSeek backend for tool calling.

This backend supports DeepSeek's API which uses OpenAI-compatible interface
but returns function calls in Python syntax: [func_name(param=value)]

Reference: https://api-docs.deepseek.com/api/create-chat-completion/
"""

import ast
import json
from typing import Any, List, Dict, Tuple, Union


async def generate_tool_call_async(
    model_name: str,
    client: Any,
    question: str,
    tools: List[Dict],
    prompt_passing_in_english: bool
) -> str:
    """
    Generate tool calls using DeepSeek API.

    Args:
        model_name: DeepSeek model name (e.g., "deepseek-chat")
        client: OpenAI-compatible async client
        question: User query
        tools: List of tool definitions in OpenAI format
        prompt_passing_in_english: Whether to request English parameter passing

    Returns:
        Raw model output as string (Python function call syntax)
    """
    system_prompt = _generate_system_prompt(tools, prompt_passing_in_english)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content


async def translate_tool_question_async(model_name: str, client: Any, question: str) -> str:
    """
    Translate a user question to English using DeepSeek.

    Args:
        model_name: DeepSeek model name
        client: OpenAI-compatible async client
        question: Question to translate

    Returns:
        Translated question as string
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following question to English. Only output the translated question, nothing else:\n\n{question}"
        }
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


async def translate_tool_answer_async(model_name: str, client: Any, parameter_value: str) -> str:
    """
    Translate a parameter value to English using DeepSeek.

    Args:
        model_name: DeepSeek model name
        client: OpenAI-compatible async client
        parameter_value: Parameter value to translate

    Returns:
        Translated parameter value as string
    """
    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the given text to English accurately. If the given text is already in English or is language agnostic, return it unchanged."
        },
        {
            "role": "user",
            "content": f"Translate the following text to English. Only output the translated text, nothing else:\n\n{parameter_value}"
        }
    ]

    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()


def postprocess_tool_calls(raw_output: str) -> str:
    """
    Postprocess raw output from DeepSeek model to extract function calls.

    DeepSeek outputs in Python function call syntax:
    [func_name1(param1=value1, param2=value2), func_name2(param3=value3)]

    Parses using AST (Abstract Syntax Tree).

    Args:
        raw_output: Raw string output from the model

    Returns:
        JSON string with either:
        - Success: {"success": true, "calls": [{func_name: {arguments}}]}
        - Error: {"success": false, "error_type": "...", "metadata": {...}}
    """
    # Strip backticks and whitespace
    cleaned_output = raw_output.strip("`\n ")

    # Add brackets if missing
    if not cleaned_output.startswith("["):
        cleaned_output = "[" + cleaned_output
    if not cleaned_output.endswith("]"):
        cleaned_output = cleaned_output + "]"

    # Remove wrapping quotes
    cleaned_input = cleaned_output.strip().strip("'")

    try:
        # Parse as Python AST
        parsed = ast.parse(cleaned_input, mode="eval")
    except SyntaxError as e:
        return json.dumps({
            "success": False,
            "error_type": "PARSING_ERROR",
            "metadata": {
                "error_message": f"Invalid Python syntax: {str(e)}",
                "raw_output": raw_output
            }
        })

    # Extract function calls from AST
    extracted = []
    try:
        if isinstance(parsed.body, ast.Call):
            extracted.append(_resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    return json.dumps({
                        "success": False,
                        "error_type": "PARSING_ERROR",
                        "metadata": {
                            "error_message": f"Expected AST Call node, but got {type(elem)}",
                            "raw_output": raw_output
                        }
                    })
                extracted.append(_resolve_ast_call(elem))
    except Exception as e:
        return json.dumps({
            "success": False,
            "error_type": "PARSING_ERROR",
            "metadata": {
                "error_message": str(e),
                "exception_type": type(e).__name__,
                "raw_output": raw_output
            }
        })

    if extracted:
        return json.dumps({
            "success": True,
            "calls": extracted
        })
    else:
        return json.dumps({
            "success": False,
            "error_type": "NO_FUNCTION_CALLS_FOUND",
            "metadata": {
                "raw_output": raw_output
            }
        })


def _generate_system_prompt(tools: List[Dict], prompt_passing_in_english: bool = True) -> str:
    """
    Generate system prompt for DeepSeek based on available tools.

    Args:
        tools: List of available tool definitions
        prompt_passing_in_english: Whether to request English parameter passing

    Returns:
        System prompt as a string
    """
    # Extract just the function definitions from tools
    functions = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            func_def = tool["function"]
            functions.append({
                "name": func_def["name"],
                "description": func_def["description"],
                "parameters": func_def["parameters"]
            })

    function_calls_json = json.dumps(functions, ensure_ascii=False, indent=2)
    passing_in_english_prompt = (
        " IMPORTANT: Pass in all parameters in function calls in English."
        if prompt_passing_in_english
        else ""
    )

    return f'''You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. You SHOULD NOT include any other text in the response.{passing_in_english_prompt}

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in json format that you can invoke.
{function_calls_json}
'''


def _resolve_ast_call(elem: ast.Call) -> Dict[str, Dict[str, Any]]:
    """
    Resolve an AST Call node to function call dictionary.

    Args:
        elem: AST Call node

    Returns:
        Dictionary in format: {func_name: {arguments}}
    """
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))

    # Extract arguments
    args_dict = {}
    for arg in elem.keywords:
        output = _resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output

    return {func_name: args_dict}


def _resolve_ast_by_type(value: ast.expr) -> Any:
    """
    Resolve AST expression to Python value.

    Args:
        value: AST expression node

    Returns:
        Resolved Python value
    """
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            return "..."
        else:
            return value.value
    elif isinstance(value, ast.UnaryOp):
        return -value.operand.value
    elif isinstance(value, ast.List):
        return [_resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        return {
            _resolve_ast_by_type(k): _resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):
        return value.value
    elif isinstance(value, ast.BinOp):
        return eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        # Convert lowercase "true" and "false" to Python's True and False
        if value.id == "true":
            return True
        elif value.id == "false":
            return False
        else:
            return value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            return ast.unparse(value)
        else:
            return _resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        # Convert tuple to list to match ground truth
        return [_resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Lambda):
        return eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        return "..."
    elif isinstance(value, ast.Subscript):
        try:
            return ast.unparse(value.body[0].value)
        except:
            return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
