from dotenv import load_dotenv
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage
from callbacks import AgentCallbackHandler


load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters"""
    print(f"get_text_length: {text=}")
    text = text.strip("\n").strip('"')

    return len(text)


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]
    
    llm = ChatOllama(
        model="qwen2.5:7b-instruct",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a tool lookup dictionary
    tool_map = {tool.name: tool for tool in tools}
    
    # Initialize conversation with the user's question
    question = "What is the length of 'SNOOP DOG' in characters?"
    messages = [HumanMessage(content=question)]
    
    # Agent loop
    while True:
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        print(f"AI Response: {response.content}")
        print(f"Tool Calls: {response.tool_calls}")
        
        # Add AI response to messages
        messages.append(response)
        
        # If no tool calls, we're done
        if not response.tool_calls:
            print(f"\nFinal Answer: {response.content}")
            break
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"\nExecuting tool: {tool_name} with input: {tool_input}")
            
            # Find and execute the tool
            if tool_name in tool_map:
                tool_to_use = tool_map[tool_name]
                observation = tool_to_use.invoke(tool_input)
                print(f"Tool result: {observation}")
                
                # Add tool result to messages
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_id,
                    )
                )
            else:
                error_msg = f"Tool {tool_name} not found"
                print(f"Error: {error_msg}")
                messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_id,
                    )
                )
