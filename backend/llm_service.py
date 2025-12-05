import json
import json
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any

class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

RESPONSE GUIDELINES:
1. Be CONCISE and DIRECT. Avoid conversational filler.
2. Use MARKDOWN for structure:
   - Use HEADERS (##) to separate sections.
   - Use BULLET POINTS for lists.
   - Use CODE BLOCKS for code or structured data.
   - Use BOLD for key terms.
3. When presenting data, use TABLES or structured lists.

ACCURACY AND HONESTY:
1. DO NOT HALLUCINATE. If you do not know the answer, admit it.
2. If you are unsure about a user request or lack necessary information, ASK CLARIFYING QUESTIONS. Do not guess.
3. NEVER fabricate facts, code, or tool outputs.

TOOL USAGE:
To use a tool, you MUST output a VALID JSON object in this exact format: {"tool": "tool_name", "arguments": {"key": "value"}}
Do NOT write any text before or after the JSON when calling a tool.

IMPORTANT: Before calling a tool, check the tool definition for required arguments. 
If a required argument is missing, ask the user for it.

STRATEGY FOR SEARCHING/FILTERING:
1. If a tool supports a specific filter argument, use it.
2. If not, call the tool WITHOUT arguments to get the full list, then filter the results yourself in the final response.

CRITICAL: NEVER simulate tool outputs. ALWAYS run the tool.
If you receive a validation error, fix the arguments and retry.

When you receive a Tool Result, analyze it and present ONLY the relevant information in a structured format. Do not dump raw JSON.

TOOL NAMESPACING & ROUTING:
Tools are prefixed with their server name to allow you to select the correct server.
Format: `server_name__tool_name` (double underscore separator).

Examples:
- `webdav__list_files`: Lists files on the WebDAV server.
- `fabricstudio__list_repositories`: Lists repos on the FabricStudio server.

USER INSTRUCTIONS:
- If the user says `@webdav`, you MUST use tools starting with `webdav__`.
- If the user says `@fabricstudio`, you MUST use tools starting with `fabricstudio__`.
- If the user says "save to @webdav", use `webdav__write_file`.

Do not complain that you don't have a tool named `@webdav`. Instead, look for `webdav__...` tools.

TROUBLESHOOTING & FALLBACKS:
- If a tool fails to retrieve a file or says "File not found", but provides the path (especially absolute paths like `/app/...`), YOU SHOULD TRY to read it yourself using any available local file tools.
- Be helpful and proactive. If a remote server can't see a file, maybe you can.
"""

import google.generativeai as genai

async def query_llm(messages: list, tools: list = None, api_key: str = None) -> str:
    """
    Queries the Gemini Pro model.
    """
    # Use provided key
    if not api_key:
        return "Error: GEMINI_API_KEY is not set. Please provide it in the UI."

    genai.configure(api_key=api_key)
    
    # Gemini uses a different message format. We need to adapt.
    # But for simplicity in this "text-in, text-out" architecture, 
    # we can just construct a prompt or use the chat session.
    
    # Construct the full prompt including system instructions
    # Gemini supports system instructions in the model config, but prepending is safer for now.
    
    current_system_prompt = SYSTEM_PROMPT
    if tools:
        tool_descriptions = json.dumps(tools, indent=2)
        current_system_prompt += f"\n\nAvailable Tools:\n{tool_descriptions}"

    # Convert messages to Gemini format or just a single string for simplicity
    # Since we are managing the loop manually, we can just feed the history as text
    # or use the chat object. Let's use the chat object for better context handling.
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Convert our message format to Gemini's
    # our format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # Gemini format: history=[{"role": "user", "parts": ["..."]}, {"role": "model", "parts": ["..."]}]
    
    gemini_history = []
    # Add system prompt as the first user message? Or just prepend to the first message?
    # Gemini Pro doesn't have a dedicated "system" role in the chat history yet (in some versions).
    # Best practice: Prepend system prompt to the first user message.
    
    first_message_content = current_system_prompt
    
    for i, msg in enumerate(messages):
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        
        if i == 0 and role == "user":
            content = f"{first_message_content}\n\nUser: {content}"
        elif i == 0 and role == "model":
             # This shouldn't happen usually, but if it does, we prepend to it? 
             # Or we insert a dummy user message.
             gemini_history.append({"role": "user", "parts": [first_message_content]})
        
        gemini_history.append({"role": role, "parts": [content]})
        
    # The last message is the new prompt, so pop it
    last_message = gemini_history.pop()
    prompt = last_message["parts"][0]
    
    chat = model.start_chat(history=gemini_history)
    
    # Configure safety settings to block nothing for the demo
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    try:
        response = await chat.send_message_async(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        print(f"LLM Error Details: {repr(e)}")
        return f"Error communicating with LLM: {str(e)}"

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
