import json
import json
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List
import re

class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]

# Fixed system prompt for MCP router/caller role (never changes)
MCP_ROUTER_SYSTEM_PROMPT = """You are an MCP router and caller. Your role is to:

1. Receive tool definitions and a user request
2. Choose the correct tool based on the user's intent
3. Extract parameters from the user's request
4. Output ONLY valid JSON in this exact format: {"tool": "tool_name", "arguments": {"key": "value"}}

RULES:
- Output ONLY the JSON tool call. No text before or after.
- Use the EXACT tool name as provided in the MCP tool documentation.
- Extract all required parameters from the user's request.
- If a required parameter is missing, output JSON with empty string or null for that parameter (the system will handle validation).
- Do NOT explain, do NOT chat, do NOT add commentary. Just output the JSON.

OUTPUT FORMAT:
{"tool": "exact_tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
"""

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
YOUR GOAL: Execute the user's intent as EFFICIENTLY as possible.

RESPONSE GUIDELINES:
1. START WITH THE TOOL CALL. If the user asks for something that requires a tool, call it IMMEDIATELY.
2. DO NOT CHAT if you can act. Do not say "I will now..." or "Let me check...". Just return the JSON.
3. BE CONCISE. After getting a tool result, summarize the answer in 1-2 sentences unless asked for details.

TOOL USAGE FORMAT:
You MUST output a VALID JSON object in this exact format: {"tool": "tool_name", "arguments": {"key": "value"}}
Do NOT write any text before or after the JSON when calling a tool.
Do NOT return error messages like "Tool not found" - if a tool is listed in Available Tools, it exists and you MUST call it using the JSON format above.

IMPORTANT:
- Before calling a tool, check the tool definition for required arguments.
- If a required argument is missing, ask the user for it.
- NEVER simulate tool outputs. ALWAYS run the tool.
- If the user gives you a CONDITIONAL instruction (e.g., "if X then do Y"), you MUST evaluate the condition FIRST.
- Only execute the action if the condition is TRUE. If the condition is FALSE, do NOT execute the action.
- Example: "if temperature > 30 then search flights" means: check temperature, and ONLY search flights if temp > 30.

SEARCHING/FILTERING:
- If a tool supports a specific filter argument, use it! This saves time.
- If not, get the list and filter in your final response.

REMINDER:
- If you see `@webdav` in the user prompt, use `webdav__...` tools.
- If you see `@fabricstudio` in the user prompt, use `fabricstudio__...` tools.
- If you see `@Booking` or `@booking` in the user prompt, use `Booking__...` tools.
- Use the EXACT tool names as listed in the Available Tools section (including the server prefix like `Booking__`).
- If a tool fails (e.g. "not found"), try `execute_shell_command` as a fallback if appropriate.

Think: "Can I do this in one step?" If yes, output the JSON tool call NOW.
"""

import httpx

def retrieve_relevant_tools(user_query: str, all_tools: List[dict], top_k: int = 5) -> List[dict]:
    """
    Tiny RAG: Retrieve the most relevant tools based on user query.
    Uses simple keyword matching and scoring.
    
    Args:
        user_query: The user's natural language request
        all_tools: List of all available tool definitions
        top_k: Number of top tools to return
    
    Returns:
        List of top-k most relevant tools
    """
    if not all_tools:
        return []
    
    # Extract keywords from user query (simple tokenization)
    query_lower = user_query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Score each tool based on keyword matches
    scored_tools = []
    for tool in all_tools:
        score = 0
        
        # Get tool text to search
        tool_name = (tool.get("name") or "").lower()
        tool_desc = (tool.get("description") or "").lower()
        
        # Check inputSchema properties and descriptions
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        schema_text = json.dumps(properties).lower()
        
        # Combine all searchable text
        searchable_text = f"{tool_name} {tool_desc} {schema_text}"
        searchable_words = set(re.findall(r'\b\w+\b', searchable_text))
        
        # Score based on:
        # 1. Exact tool name match (highest weight)
        if any(word in tool_name for word in query_words):
            score += 10
        
        # 2. Description matches
        for word in query_words:
            if word in tool_desc:
                score += 3
            if word in schema_text:
                score += 2
        
        # 3. Word overlap
        overlap = len(query_words & searchable_words)
        score += overlap
        
        scored_tools.append((score, tool))
    
    # Sort by score (descending) and return top-k
    scored_tools.sort(key=lambda x: x[0], reverse=True)
    return [tool for _, tool in scored_tools[:top_k]]

async def query_ollama(messages: list, system_prompt: str, model_url: str, model_name: str = "qwen3:8b") -> str:
    """
    Queries a local Ollama instance.
    
    Args:
        messages: List of message dicts
        system_prompt: System prompt to use
        model_url: Ollama server URL
        model_name: Model name to use (e.g., qwen3:8b, gemma3:8b, etc.)
    """
    if not model_url:
        return "Error: Ollama URL is not set."
        
    # Ensure URL ends with /api/chat
    # If the user provides "http://10.3.0.7:11434", we append "/api/chat"
    # If they include it, we respect it.
    api_endpoint = model_url
    if not api_endpoint.endswith("/api/chat"):
        if api_endpoint.endswith("/"):
            api_endpoint += "api/chat"
        else:
            api_endpoint += "/api/chat"
            
    # Prep messages
    # Custom System Prompt for Ollama to force tool usage
    # We append the tools here instead of assuming they are in the passed system_prompt
    # This allows us to format them specifically for the local model
    
    final_system_prompt = system_prompt
    if "Available Tools" not in final_system_prompt:
         # Need to be able to pass tools to formatting
         pass 

    print(f"DEBUG: Ollama System Prompt Length: {len(final_system_prompt)}")
    print(f"DEBUG: Using Ollama model: {model_name}")
    
    ollama_messages = [{"role": "system", "content": final_system_prompt}]
    
    for msg in messages:
        # Map roles if necessary, but "user" and "assistant" are standard
        role = msg["role"]
        if role == "model": role = "assistant" # Gemini uses 'model', Ollama uses 'assistant'
        ollama_messages.append({"role": role, "content": msg["content"]})
        
    payload = {
        "model": model_name,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": 0 # Low temp for tool execution
        }
    }
    
    try:
        # Increase timeout to 300s (5 mins) because loading a model for the first time can be slow
        timeout = httpx.Timeout(300.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        print(f"Ollama HTTP Error: {error_body}")
        return f"Ollama Error: {error_body}"
    except Exception as e:
        print(f"Ollama Error Details: {repr(e)}")
        return f"Error communicating with Ollama at {model_url}: {str(e)}"

import time

# Global Rate Limiter
LAST_REQUEST_TIME = 0
RATE_LIMIT_INTERVAL = 15  # 15 seconds (4 requests/min) to be safe under 5 RPM limit

async def query_llm(messages: list, tools: list = None, api_key: str = None, provider: str = "openai", model_url: str = None, model_name: str = "qwen3:8b", use_qwen_rag: bool = False) -> str:
    """
    Queries the selected LLM provider.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: List of tool definitions
        api_key: API key for providers that need it
        provider: 'openai' or 'ollama'
        model_url: URL for Ollama instance
        model_name: Model name for Ollama (e.g., qwen3:8b, gemma3:8b)
        use_qwen_rag: If True, use the new Qwen RAG approach (fixed prompt + retrieved tools)
    """
    global LAST_REQUEST_TIME
    
    # Dispatch based on provider
    if provider == "ollama":
        if use_qwen_rag and tools:
            # New Qwen RAG approach:
            # 1. Fixed system prompt (MCP_ROUTER_SYSTEM_PROMPT)
            # 2. Retrieve relevant tools using RAG
            # 3. Build context with retrieved tools
            # 4. Send to Qwen
            
            # Get user query from last message
            user_query = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break
            
            # Retrieve top-k relevant tools
            relevant_tools = retrieve_relevant_tools(user_query, tools, top_k=5)
            
            # Build context with retrieved tool documentation
            tool_context = "## MCP TOOL DOCUMENTATION:\n\n"
            if relevant_tools:
                for tool in relevant_tools:
                    tool_context += f"Tool: {tool.get('name', 'unknown')}\n"
                    tool_context += f"Description: {tool.get('description', 'No description')}\n"
                    tool_context += f"Input Schema: {json.dumps(tool.get('inputSchema', {}), indent=2)}\n\n"
            else:
                # Fallback: include all tools if RAG found nothing
                tool_context += json.dumps(tools, indent=2)
            
            # Build final system prompt: fixed instructions + tool context
            qwen_system_prompt = f"{MCP_ROUTER_SYSTEM_PROMPT}\n\n{tool_context}\n\nRemember: Output ONLY the JSON tool call, nothing else."
            
            return await query_ollama(messages, qwen_system_prompt, model_url, model_name=model_name)
        else:
            # Legacy Ollama approach
            ollama_system_prompt = SYSTEM_PROMPT
            if tools:
                tool_descriptions = json.dumps(tools, indent=2)
                ollama_system_prompt += f"\n\n## AVAILABLE TOOLS (JSON Format):\n{tool_descriptions}\n\nYou MUST use these tools to answer queries. Do not say you cannot access them. Just output the JSON to call them."
            
            return await query_ollama(messages, ollama_system_prompt, model_url, model_name=model_name)

    # Construct the full prompt including system instructions (for OpenAI)
    current_system_prompt = SYSTEM_PROMPT
    if tools:
        tool_descriptions = json.dumps(tools, indent=2)
        current_system_prompt += f"\n\nAvailable Tools:\n{tool_descriptions}"

    if provider == "openai":
        from openai import AsyncOpenAI
        if not api_key:
            return "Error: OPENAI_API_KEY is not set. Please provide it in the UI."
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Prepare messages
        openai_messages = [{"role": "system", "content": current_system_prompt}]
        for msg in messages:
             # Map 'model' to 'assistant' if needed
             role = msg["role"]
             if role == "model": role = "assistant"
             openai_messages.append({"role": role, "content": msg["content"]})

        try:
            print("Sending request to OpenAI (GPT-4o Mini)...")
            # Log request details
            print(f"[DEBUG] OpenAI Request - Messages count: {len(openai_messages)}")
            if openai_messages:
                system_msg = next((m for m in openai_messages if m.get("role") == "system"), None)
                if system_msg:
                    sys_content = system_msg.get("content", "")
                    print(f"[DEBUG] System prompt length: {len(sys_content)} chars")
                    if "Available Tools" in sys_content:
                        # Extract tool count from system prompt
                        import re
                        tool_matches = re.findall(r'"name":\s*"([^"]+)"', sys_content)
                        if tool_matches:
                            print(f"[DEBUG] Tools in system prompt: {len(tool_matches)} tools")
                            print(f"[DEBUG] Tool names: {', '.join(tool_matches[:5])}{'...' if len(tool_matches) > 5 else ''}")
                # Log last user message preview
                user_msgs = [m for m in openai_messages if m.get("role") == "user"]
                if user_msgs:
                    last_user = user_msgs[-1].get("content", "")[:200]
                    print(f"[DEBUG] Last user message preview: {last_user}...")
            print(f"[DEBUG] OpenAI Request - Model: gpt-4o-mini, Temperature: 0")
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return f"Error communicating with OpenAI: {str(e)}"
    return "Error: Unsupported LLM provider. Use 'openai' or 'ollama'."

def parse_llm_response(response_content: str) -> dict:
    """
    Parses the LLM response. 
    Returns a dict with 'type': 'tool_call' or 'text', and relevant data.
    """
    print(f"DEBUG: Raw LLM Response: {repr(response_content)}")
    try:
        # Attempt to find JSON object using regex
        import re
        # Look for a JSON object structure: { ... }
        # This regex is simple and might need refinement for nested braces, 
        # but for simple tool calls it usually works.
        # We look for the first '{' and the last '}'
        match = re.search(r'(\{.*\})', response_content.replace('\n', ' '), re.DOTALL)
        
        json_str = ""
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try cleaning markdown code blocks
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            json_str = clean_content.strip()

        data = json.loads(json_str)
        
        # Validate with Pydantic
        tool_call = ToolCall(**data)
        return {"type": "tool_call", "data": tool_call}
        
    except (json.JSONDecodeError, ValidationError):
        # If it's not valid JSON or doesn't match the schema, treat as text
        # But if it looks like it tried to be JSON (starts with {), return error
        if response_content.strip().startswith("{"):
             return {"type": "error", "message": "System Error: Failed to parse tool call."}
        
        return {"type": "text", "content": response_content}
