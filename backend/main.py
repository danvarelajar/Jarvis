from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from .mcp_client import connection_manager, parse_server_route
from .llm_service import query_llm, parse_llm_response

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

class ConnectRequest(BaseModel):
    server_name: str
    url: str
    headers: Optional[Dict[str, str]] = None
    transport: str = "sse"

@app.post("/api/connect")
async def connect_server(request: ConnectRequest):
    connection_manager.add_server(request.server_name, request.url, request.headers, request.transport, save=True)
    return {"status": "connected", "server": request.server_name}

@app.get("/api/config")
async def get_config():
    # Construct config from active connections
    config = {"mcpServers": {}}
    for name, conn in connection_manager.connections.items():
        config["mcpServers"][name] = {
            "url": conn.url,
            "headers": conn.headers,
            "transport": conn.transport
        }
    return config

class ConfigRequest(BaseModel):
    mcpServers: Dict[str, Dict[str, Any]]

@app.post("/api/config")
async def update_config(request: ConfigRequest):
    # This endpoint replaces the current config with the new one
    # For simplicity, we'll just add new ones. 
    # To fully replace, we'd need to stop existing connections, which we haven't implemented.
    # So we'll just add/update.
    for name, details in request.mcpServers.items():
        connection_manager.add_server(
            name, 
            details["url"], 
            details.get("headers"), 
            details.get("transport", "sse"),
            save=True
        )
    return {"status": "updated", "count": len(request.mcpServers)}

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    user_message = request.messages[-1]["content"]
    
    # 1. Smart Routing
    target_server = parse_server_route(user_message)
    
    # 2. Tool Discovery
    tools = await connection_manager.list_tools(target_server)
    
    # 3. Agent Loop
    # We allow up to 5 turns to prevent infinite loops
    current_messages = request.messages.copy()
    
    for _ in range(5):
        # Query LLM
        response_content = await query_llm(current_messages, tools)
        
        # Parse Response
        parsed_response = parse_llm_response(response_content)
        
        if parsed_response["type"] == "text":
            return {"role": "assistant", "content": parsed_response["content"]}
            
        elif parsed_response["type"] == "error":
            return {"role": "assistant", "content": parsed_response["message"]}
            
        elif parsed_response["type"] == "tool_call":
            tool_call = parsed_response["data"]
            
            # Prevent infinite loops: Check if we already called this tool with these args
            # We need to serialize args to check for equality
            import json
            tool_signature = (tool_call.tool, json.dumps(tool_call.arguments, sort_keys=True))
            
            # Initialize history if not present (using a local variable outside the loop would be better, 
            # but we can just check the conversation history too? 
            # Actually, let's use a set for this request scope)
            if 'tool_call_history' not in locals():
                tool_call_history = set()
            
            if tool_signature in tool_call_history:
                error_msg = f"System Error: You have already called tool '{tool_call.tool}' with these arguments. Do not call it again. Analyze the results you already have or ask the user for clarification."
                print(f"Loop detected: {error_msg}")
                current_messages.append({"role": "assistant", "content": response_content})
                current_messages.append({"role": "user", "content": error_msg})
                continue
            
            tool_call_history.add(tool_signature)
            
            # Execute tool logic (Routing, Validation, Execution)
            try:
                # Routing
                server_to_call = target_server
                if not server_to_call:
                    all_tools = await connection_manager.list_tools()
                    sessions = connection_manager.get_all_sessions()
                    for name, session in sessions.items():
                        try:
                            t_list = await session.list_tools()
                            if any(t.name == tool_call.tool for t in t_list.tools):
                                server_to_call = name
                                break
                        except:
                            continue
                
                # Validation
                tool_def = next((t for t in tools if t['name'] == tool_call.tool), None)
                if tool_def:
                    input_schema = tool_def.get('inputSchema', {})
                    required_args = input_schema.get('required', [])
                    allowed_args = input_schema.get('properties', {}).keys()
                    
                    # Check for missing required args
                    missing_args = [arg for arg in required_args if arg not in tool_call.arguments or tool_call.arguments[arg] in (None, "")]
                    if missing_args:
                        error_msg = f"Error: Missing required arguments for tool '{tool_call.tool}': {', '.join(missing_args)}. Please ask the user for these values."
                        print(f"Validation failed: {error_msg}. Retrying with LLM...")
                        current_messages.append({"role": "assistant", "content": response_content})
                        current_messages.append({"role": "user", "content": error_msg})
                        continue

                    # Check for unknown args
                    unknown_args = [arg for arg in tool_call.arguments if arg not in allowed_args]
                    if unknown_args:
                        error_msg = f"Error: Tool '{tool_call.tool}' does not accept arguments: {', '.join(unknown_args)}. Allowed arguments: {', '.join(allowed_args)}. SUGGESTION: Call the tool WITHOUT these arguments to get the full list, then filter the results yourself."
                        print(f"Validation failed: {error_msg}. Retrying with LLM...")
                        current_messages.append({"role": "assistant", "content": response_content})
                        current_messages.append({"role": "user", "content": error_msg})
                        continue

                if not server_to_call:
                    return {"role": "assistant", "content": f"Error: Tool '{tool_call.tool}' not found on any connected server."}

                print(f"Executing tool '{tool_call.tool}' on server '{server_to_call}' with args: {tool_call.arguments}")
                result = await connection_manager.call_tool(server_to_call, tool_call.tool, tool_call.arguments)
                
                    # Extract text content or serialize object
                tool_output = ""
                if hasattr(result, 'content'):
                    for item in result.content:
                        if item.type == 'text':
                            tool_output += item.text
                        elif item.type == 'image':
                            tool_output += "[Image Content]"
                else:
                    # Try to serialize as compact JSON if it's a list or dict
                    try:
                        import json
                        # If result is a Pydantic model or similar, try model_dump
                        if hasattr(result, 'model_dump'):
                            data = result.model_dump()
                        elif hasattr(result, '__dict__'):
                            data = result.__dict__
                        else:
                            data = result
                        
                        tool_output = json.dumps(data, separators=(',', ':'))
                    except:
                        tool_output = str(result)
                
                # Truncate if too long to prevent LLM timeout/context overflow
                # Gemini 2.5 Flash has a huge context window, so we can be very generous.
                MAX_TOOL_OUTPUT = 1000000 
                if len(tool_output) > MAX_TOOL_OUTPUT:
                    tool_output = tool_output[:MAX_TOOL_OUTPUT] + f"\n... (truncated, {len(tool_output) - MAX_TOOL_OUTPUT} chars omitted). Warning: Some data is missing."
                
                # Feed result back to LLM
                current_messages.append({"role": "assistant", "content": response_content})
                current_messages.append({"role": "user", "content": f"Tool Result: {tool_output}"})
                
                # Loop continues to let LLM process the result
                
            except Exception as e:
                return {"role": "assistant", "content": f"Error executing tool: {str(e)}"}
    
    return {"role": "assistant", "content": "Error: Maximum agent turns reached."}

# Mount static files
# We need to ensure the directory exists, even if empty, to avoid startup errors
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# SPA Fallback
@app.exception_handler(404)
async def not_found(request: Request, exc: HTTPException):
    return FileResponse(os.path.join(static_dir, "index.html"))
