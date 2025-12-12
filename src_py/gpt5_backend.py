



async def generate_tool_call_async(model_name: str, client: any, question: str, tools: list, prompt_passing_in_english: bool) -> str:
    developer_message = {
        "role": "developer",
        "content": (
            "You are an expert in composing functions. "
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out. "
            "If the given question lacks the parameters required by the function, also point it out.\n\n"
            "You should ONLY return function calls in your response. "
            "You MUST NOT include any other text, explanations, or direct answers. "
            "If you decide to invoke any function(s), you MUST use the provided tools. "
            "Do NOT attempt to answer the question directly without using the available functions."
        )
    }
    input_messages = [
        developer_message,
        {"role": "user", "content": question}
    ]
    response = await client.responses.create(
        model = model_name,
        input = input_messages,
        tools = tools,
    )
    return response.output