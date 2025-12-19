import json
import json
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List
import re
from datetime import datetime, timedelta
import subprocess
import time
import os

def get_timestamp() -> str:
    """Returns a formatted timestamp for logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

def format_duration(start_time: float) -> str:
    """Formats a duration in seconds to a readable string."""
    duration = time.time() - start_time
    if duration < 1:
        return f"{duration*1000:.1f}ms"
    else:
        return f"{duration:.2f}s"

class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]

# Fixed system prompt for MCP router/caller role (never changes)
MCP_ROUTER_SYSTEM_PROMPT = """You are an MCP router and caller. Your role is to:

1. Receive tool definitions (if any) and a user request
2. Tools are ONLY available when the user explicitly requests them using @server_name prefix (e.g., @weather, @booking)
3. If NO tools are provided in the documentation, the user did NOT use @server_name - respond with TEXT only
4. If tools ARE provided, the user used @server_name - you can call tools if needed
5. Extract parameters from the user's request if calling a tool

CRITICAL RULES:
- If the "MCP TOOL DOCUMENTATION" section is empty or says "No tools available", respond with TEXT only (no JSON)
- If the user's question is conversational (greetings, "how are you", general questions without @server_name), respond with TEXT only (no JSON)
- ONLY call tools if: (1) tools are listed in the documentation AND (2) the user's request clearly requires a tool
- NEVER invent or hallucinate tool names - only use tools that are explicitly listed in the documentation
- Use the EXACT tool name as provided in the MCP tool documentation
- Use the EXACT parameter names from the Input Schema - do NOT use synonyms or alternative names
- Check the Input Schema properties to see the exact parameter names required (e.g., if schema says "from", use "from" not "origin")
- If you already have all the information needed, return a TEXT response (not JSON)
- If you see "System Error: You have already called tool..." - STOP calling tools and return a text response

OUTPUT FORMAT:
- For tool calls (ONLY when tools are available AND needed): {"tool": "exact_tool_name", "arguments": {"param1": "value1"}}
- For text responses (when no tools needed or no tools available): Just return plain text (no JSON, no code blocks)
"""

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
YOUR GOAL: Execute the user's intent as EFFICIENTLY as possible.

RESPONSE GUIDELINES:
1. CALL TOOLS IMMEDIATELY - no conversational preamble. If the user asks for something that requires a tool, output ONLY the JSON tool call (no "I'll help you..." or "Let me check..." text).
   (Reason: Prevents wasted turns on acknowledgments, gets results faster, better user experience)
   - This applies to intermediate turns when tools are needed (multi-turn scenarios)
   - The final turn (after tool execution) should return formatted text, not JSON
2. WHEN NO TOOL IS NEEDED, respond with plain text (not JSON). For conversational questions, greetings, or requests that don't require tools, just answer naturally.
3. FORMAT FINAL RESPONSES IN MARKDOWN. After tool execution completes, format your final answer using markdown (headers, lists, code blocks, etc.) for readability.

TOOL USAGE FORMAT:x
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

Think: "Can I do this in one step?" If yes, output the JSON tool call NOW.
"""

import httpx

def get_current_date(agent_mode: str = "defender") -> tuple[str, str]:
    """
    Gets the current date and time.
    
    VULNERABILITY (Naive Mode): Uses shell command execution with shell=True which is vulnerable to command injection.
    Defender Mode: Uses safe Python datetime module.
    
    Args:
        agent_mode: 'defender' (safe) or 'naive' (vulnerable)
    
    Returns:
        Tuple of (current_date, current_datetime) as strings
    """
    if agent_mode == "naive":
        # VULNERABLE: Command injection vulnerability
        # In naive mode, we use shell commands with shell=True and string construction
        # This allows attackers to inject commands via environment variables or format strings
        try:
            # VULNERABLE: Read date format from environment variable (could be controlled by attacker)
            # In a real attack scenario, an attacker might set DATE_FORMAT env var with malicious commands
            date_format = os.environ.get("DATE_FORMAT", "+%Y-%m-%d")
            datetime_format = os.environ.get("DATETIME_FORMAT", "+%Y-%m-%d %H:%M:%S")
            
            # VULNERABLE: Using shell=True with string concatenation allows command injection
            # If DATE_FORMAT contains "; command", the shell will execute both commands
            # Example: DATE_FORMAT="+%Y-%m-%d; echo 'INJECTED'" will execute both date and echo
            # Note: We don't quote the format string to allow injection (vulnerable by design)
            date_cmd = f"date {date_format}"
            datetime_cmd = f"date {datetime_format}"
            
            print(f"[VULNERABLE] Naive mode: Executing shell command: {date_cmd}")
            result = subprocess.run(
                date_cmd,
                shell=True,  # VULNERABLE: shell=True allows command injection
                capture_output=True,
                text=True,
                timeout=2
            )
            current_date = result.stdout.strip()
            
            print(f"[VULNERABLE] Naive mode: Executing shell command: {datetime_cmd}")
            result2 = subprocess.run(
                datetime_cmd,
                shell=True,  # VULNERABLE: shell=True allows command injection
                capture_output=True,
                text=True,
                timeout=2
            )
            current_datetime = result2.stdout.strip()
            
            # If commands fail, fall back to safe method
            if not current_date or not current_datetime:
                raise Exception("Command failed")
                
            return current_date, current_datetime
        except Exception as e:
            # Fallback to safe method if command fails
            print(f"[WARNING] Date command failed, using safe fallback: {e}")
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return current_date, current_datetime
    else:
        # Defender mode: Safe Python datetime
        # Even if DATE_FORMAT env var is set, we ignore it and use safe Python API
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_date, current_datetime

def calculate_specific_dates(user_query: str, current_date: str, today: datetime) -> str:
    """
    Detects date-related terms in the user query and calculates specific dates.
    
    Args:
        user_query: The user's query text
        current_date: Current date as string (YYYY-MM-DD)
        today: Current date as datetime object
    
    Returns:
        Enhanced date context string with specific date calculations
    """
    user_lower = user_query.lower()
    specific_dates = []
    
    # Day of week calculations
    days_of_week = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    # Check for "next [day]" patterns
    for day_name, day_num in days_of_week.items():
        if f'next {day_name}' in user_lower or f'next {day_name[:3]}' in user_lower:
            # Find next occurrence of this day
            days_ahead = (day_num - today.weekday()) % 7
            if days_ahead == 0:  # If today is that day, get next week's
                days_ahead = 7
            next_day = today + timedelta(days=days_ahead)
            specific_dates.append(f"- When the user says 'next {day_name}', use: {next_day.strftime('%Y-%m-%d')}")
    
    # Check for "this [day]" patterns
    for day_name, day_num in days_of_week.items():
        if f'this {day_name}' in user_lower or f'this {day_name[:3]}' in user_lower:
            # Find this week's occurrence
            days_ahead = (day_num - today.weekday()) % 7
            this_day = today + timedelta(days=days_ahead)
            specific_dates.append(f"- When the user says 'this {day_name}', use: {this_day.strftime('%Y-%m-%d')}")
    
    # Check for "in X days" patterns
    days_match = re.search(r'in (\d+) days?', user_lower)
    if days_match:
        days = int(days_match.group(1))
        future_date = today + timedelta(days=days)
        specific_dates.append(f"- When the user says 'in {days} days', use: {future_date.strftime('%Y-%m-%d')}")
    
    # Check for "X days from now"
    days_match = re.search(r'(\d+) days? from now', user_lower)
    if days_match:
        days = int(days_match.group(1))
        future_date = today + timedelta(days=days)
        specific_dates.append(f"- When the user says '{days} days from now', use: {future_date.strftime('%Y-%m-%d')}")
    
    # Check for "next week"
    if 'next week' in user_lower:
        next_week = today + timedelta(days=7)
        specific_dates.append(f"- When the user says 'next week', use: {next_week.strftime('%Y-%m-%d')} (7 days from today)")
    
    # Check for "in X weeks"
    weeks_match = re.search(r'in (\d+) weeks?', user_lower)
    if weeks_match:
        weeks = int(weeks_match.group(1))
        future_date = today + timedelta(weeks=weeks)
        specific_dates.append(f"- When the user says 'in {weeks} weeks', use: {future_date.strftime('%Y-%m-%d')}")
    
    if specific_dates:
        return "\n".join(specific_dates) + "\n"
    return ""

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

    print(f"[{get_timestamp()}] DEBUG: Ollama System Prompt Length: {len(final_system_prompt)}")
    print(f"[{get_timestamp()}] DEBUG: Using Ollama model: {model_name}")
    
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
        "keep_alive": "10m",  # Keep model loaded for 10 minutes after last use (prevents reloading from disk)
        "options": {
            "temperature": 0 # Low temp for tool execution
        }
    }
    
    # Log payload size for debugging
    import json as json_module
    payload_size = len(json_module.dumps(payload))
    total_chars = sum(len(str(msg.get("content", ""))) for msg in ollama_messages)
    print(f"[{get_timestamp()}] [LLM] Payload size: {payload_size} bytes, Total message chars: {total_chars}")
    
    try:
        # Check if model is already loaded before making request
        base_url = model_url
        if base_url.endswith("/api/chat"):
            base_url = base_url[:-9]
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        ps_endpoint = f"{base_url}/api/ps"
        
        try:
            check_start = time.time()
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as check_client:
                ps_response = await check_client.get(ps_endpoint)
                if ps_response.status_code == 200:
                    ps_data = ps_response.json()
                    models_loaded = ps_data.get("models", [])
                    model_loaded = any(m.get("name", "").startswith(model_name) for m in models_loaded)
                    if model_loaded:
                        print(f"[{get_timestamp()}] [LLM] ✓ Model '{model_name}' is already loaded in memory")
                    else:
                        print(f"[{get_timestamp()}] [LLM] ⚠️  Model '{model_name}' is NOT loaded - will need to load from disk (~4s delay)")
                else:
                    print(f"[{get_timestamp()}] [LLM] Could not check model status (status: {ps_response.status_code})")
        except Exception as e:
            print(f"[{get_timestamp()}] [LLM] Could not check if model is loaded: {e}")
        
        # Increase timeout to 300s (5 mins) because loading a model for the first time can be slow
        timeout = httpx.Timeout(300.0, connect=10.0)
        request_start = time.time()
        print(f"[{get_timestamp()}] [LLM] Sending request to Ollama (endpoint: {api_endpoint})...")
        
        # Track connection establishment time
        connect_start = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            connect_time = time.time() - connect_start
            if connect_time > 0.1:
                print(f"[{get_timestamp()}] [LLM] HTTP client created ({format_duration(connect_start)})")
            
            # Track actual HTTP request time
            http_start = time.time()
            print(f"[{get_timestamp()}] [LLM] HTTP POST request initiated...")
            print(f"[{get_timestamp()}] [LLM] Waiting for Ollama inference (this may take 2-3 minutes if model needs to load)...")
            response = await client.post(api_endpoint, json=payload)
            http_time = time.time() - http_start
            print(f"[{get_timestamp()}] [LLM] HTTP response received ({format_duration(http_start)}), status: {response.status_code}")
            
            # Performance analysis
            if http_time > 60:
                print(f"[{get_timestamp()}] [LLM] ⚠️  SLOW: Inference took {http_time:.1f}s - model may be loading from disk or underpowered")
            elif http_time > 30:
                print(f"[{get_timestamp()}] [LLM] ⚠️  MODERATE: Inference took {http_time:.1f}s - consider model preloading")
            else:
                print(f"[{get_timestamp()}] [LLM] ✓ Inference completed in {http_time:.1f}s")
            
            response.raise_for_status()
            
            # Track JSON parsing time
            parse_start = time.time()
            result = response.json()
            parse_time = time.time() - parse_start
            if parse_time > 0.1:
                print(f"[{get_timestamp()}] [LLM] JSON parsed ({format_duration(parse_start)})")
            
            total_time = time.time() - request_start
            print(f"[{get_timestamp()}] [LLM] Ollama response received (total: {format_duration(request_start)}, HTTP wait: {format_duration(http_start)})")
            
            # Log response size
            response_content = result.get("message", {}).get("content", "")
            if response_content:
                print(f"[{get_timestamp()}] [LLM] Response content length: {len(response_content)} chars")
            
            return response_content
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

async def query_llm(messages: list, tools: list = None, api_key: str = None, provider: str = "openai", model_url: str = None, model_name: str = "qwen3:8b", use_qwen_rag: bool = False, agent_mode: str = "defender", user_query: str = "") -> str:
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
        agent_mode: 'defender' (safe) or 'naive' (vulnerable) - affects date retrieval method
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
                    tool_name = tool.get('name', 'unknown')
                    tool_context += f"### Tool: {tool_name}\n"
                    tool_context += f"Description: {tool.get('description', 'No description')}\n\n"
                    
                    # Extract and format input schema with explicit parameter names
                    input_schema = tool.get('inputSchema', {})
                    properties = input_schema.get('properties', {})
                    required = input_schema.get('required', [])
                    
                    tool_context += "**REQUIRED PARAMETERS (use EXACT names):**\n"
                    for param_name in required:
                        param_info = properties.get(param_name, {})
                        param_type = param_info.get('type', 'string')
                        param_desc = param_info.get('description', '')
                        tool_context += f"  - `{param_name}` ({param_type}): {param_desc}\n"
                    
                    if properties:
                        tool_context += "\n**ALL PARAMETERS (use EXACT names from this list):**\n"
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'string')
                            param_desc = param_info.get('description', '')
                            is_required = param_name in required
                            req_marker = " [REQUIRED]" if is_required else " [OPTIONAL]"
                            tool_context += f"  - `{param_name}` ({param_type}){req_marker}: {param_desc}\n"
                    
                    tool_context += f"\n**Full Input Schema (JSON):**\n```json\n{json.dumps(input_schema, indent=2)}\n```\n\n"
                    # List exact parameter names to emphasize
                    exact_params = list(properties.keys())
                    if exact_params:
                        param_list = ", ".join([f"'{p}'" for p in exact_params[:5]])
                        tool_context += f"**CRITICAL: Use EXACT parameter names from above. Do NOT use synonyms like 'origin'/'destination' - use the exact names: {param_list}**\n\n"
                    else:
                        tool_context += "**CRITICAL: Use EXACT parameter names from the Input Schema above. Do NOT invent or use synonyms.**\n\n"
            else:
                # Fallback: include all tools if RAG found nothing
                if tools:
                    tool_context += json.dumps(tools, indent=2)
                else:
                    tool_context += "No tools available. Respond with text only (no JSON, no tool calls)."
            
            # Get current date for context (vulnerable to command injection in naive mode)
            current_date, current_datetime = get_current_date(agent_mode)
            
            # Calculate tomorrow and day after tomorrow for explicit examples
            try:
                today = datetime.strptime(current_date, "%Y-%m-%d")
                tomorrow = today + timedelta(days=1)
                day_after = today + timedelta(days=2)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")
                day_after_str = day_after.strftime("%Y-%m-%d")
                current_year = today.year
            except Exception as e:
                tomorrow_str = "N/A"
                day_after_str = "N/A"
                current_year = current_date[:4] if len(current_date) >= 4 else "2024"
                today = datetime.now()
            
            # Build base date context
            date_context = (
                f"\n## CURRENT DATE AND TIME (CRITICAL - USE THESE DATES):\n"
                f"Today's date: {current_date}\n"
                f"Current date and time: {current_datetime}\n\n"
                f"DATE CALCULATIONS:\n"
                f"- When the user says 'today', use: {current_date}\n"
                f"- When the user says 'tomorrow', use: {tomorrow_str}\n"
                f"- When the user says 'day after tomorrow' or 'after tomorrow', use: {day_after_str}\n"
                f"- When the user says 'next week', add 7 days to {current_date}\n"
            )
            
            # Add specific date calculations if user query contains date-related terms
            if user_query:
                specific_dates = calculate_specific_dates(user_query, current_date, today)
                if specific_dates:
                    date_context += f"\nSPECIFIC DATE CALCULATIONS (based on user query):\n{specific_dates}"
            
            date_context += (
                f"\nIMPORTANT: The current year is {current_year}. "
                f"DO NOT use dates from 2023 or earlier. Always calculate relative dates from TODAY ({current_date}). "
                f"Example: If today is {current_date} and user says 'tomorrow', use {tomorrow_str}, NOT 2023-10-04.\n\n"
            )
            
            # Build final system prompt: fixed instructions + tool context
            if not tools or not relevant_tools:
                # No tools available - user didn't use @server_name prefix OR loop was detected
                # Make it very explicit - no JSON allowed
                qwen_system_prompt = f"{MCP_ROUTER_SYSTEM_PROMPT}\n{date_context}\n{tool_context}\n\nCRITICAL: No tools are available. You MUST respond with PLAIN TEXT ONLY. DO NOT output JSON. DO NOT output {{}}. DO NOT output code blocks with JSON. Write a natural language response. If you output any JSON, you are making an error."
            else:
                qwen_system_prompt = f"{MCP_ROUTER_SYSTEM_PROMPT}\n{date_context}\n{tool_context}\n\nIMPORTANT: Tools are available because the user used @server_name. After you receive tool results, if you have enough information to answer the user, return a TEXT response (not JSON). Only call tools if you still need more information."
            
            return await query_ollama(messages, qwen_system_prompt, model_url, model_name=model_name)
        else:
            # Legacy Ollama approach
            # Get current date for context (vulnerable to command injection in naive mode)
            current_date, current_datetime = get_current_date(agent_mode)
            
            # Calculate tomorrow and day after tomorrow for explicit examples
            try:
                today = datetime.strptime(current_date, "%Y-%m-%d")
                tomorrow = today + timedelta(days=1)
                day_after = today + timedelta(days=2)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")
                day_after_str = day_after.strftime("%Y-%m-%d")
                current_year = today.year
            except Exception as e:
                tomorrow_str = "N/A"
                day_after_str = "N/A"
                current_year = current_date[:4] if len(current_date) >= 4 else "2024"
                today = datetime.now()
            
            # Build base date context
            date_context = (
                f"\n## CURRENT DATE AND TIME (CRITICAL - USE THESE DATES):\n"
                f"Today's date: {current_date}\n"
                f"Current date and time: {current_datetime}\n\n"
                f"DATE CALCULATIONS:\n"
                f"- When the user says 'today', use: {current_date}\n"
                f"- When the user says 'tomorrow', use: {tomorrow_str}\n"
                f"- When the user says 'day after tomorrow' or 'after tomorrow', use: {day_after_str}\n"
                f"- When the user says 'next week', add 7 days to {current_date}\n"
            )
            
            # Add specific date calculations if user query contains date-related terms
            if user_query:
                specific_dates = calculate_specific_dates(user_query, current_date, today)
                if specific_dates:
                    date_context += f"\nSPECIFIC DATE CALCULATIONS (based on user query):\n{specific_dates}"
            
            date_context += (
                f"\nIMPORTANT: The current year is {current_year}. "
                f"DO NOT use dates from 2023 or earlier. Always calculate relative dates from TODAY ({current_date}). "
                f"Example: If today is {current_date} and user says 'tomorrow', use {tomorrow_str}, NOT 2023-10-04.\n\n"
            )
            
            ollama_system_prompt = SYSTEM_PROMPT + date_context
            if tools:
                tool_descriptions = json.dumps(tools, indent=2)
                ollama_system_prompt += f"\n\n## AVAILABLE TOOLS (JSON Format):\n{tool_descriptions}\n\nYou MUST use these tools to answer queries. Do not say you cannot access them. Just output the JSON to call them."
            else:
                # No tools available - emphasize conversational response
                ollama_system_prompt += "\n\n## AVAILABLE TOOLS:\nNo tools are available. Respond with plain text only. Do NOT output JSON. Do NOT try to call or invent tools."
            
            return await query_ollama(messages, ollama_system_prompt, model_url, model_name=model_name)

    # Construct the full prompt including system instructions (for OpenAI)
    # Get current date for context (vulnerable to command injection in naive mode)
    current_date, current_datetime = get_current_date(agent_mode)
    
    # Calculate tomorrow and day after tomorrow for explicit examples
    try:
        today = datetime.strptime(current_date, "%Y-%m-%d")
        tomorrow = today + timedelta(days=1)
        day_after = today + timedelta(days=2)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        day_after_str = day_after.strftime("%Y-%m-%d")
        current_year = today.year
    except Exception as e:
        tomorrow_str = "N/A"
        day_after_str = "N/A"
        current_year = current_date[:4] if len(current_date) >= 4 else "2024"
        today = datetime.now()
    
    # Build base date context
    date_context = (
        f"\n## CURRENT DATE AND TIME (CRITICAL - USE THESE DATES):\n"
        f"Today's date: {current_date}\n"
        f"Current date and time: {current_datetime}\n\n"
        f"DATE CALCULATIONS:\n"
        f"- When the user says 'today', use: {current_date}\n"
        f"- When the user says 'tomorrow', use: {tomorrow_str}\n"
        f"- When the user says 'day after tomorrow' or 'after tomorrow', use: {day_after_str}\n"
        f"- When the user says 'next week', add 7 days to {current_date}\n"
    )
    
    # Add specific date calculations if user query contains date-related terms
    if user_query:
        specific_dates = calculate_specific_dates(user_query, current_date, today)
        if specific_dates:
            date_context += f"\nSPECIFIC DATE CALCULATIONS (based on user query):\n{specific_dates}"
    
    date_context += (
        f"\nIMPORTANT: The current year is {current_year}. "
        f"DO NOT use dates from 2023 or earlier. Always calculate relative dates from TODAY ({current_date}). "
        f"Example: If today is {current_date} and user says 'tomorrow', use {tomorrow_str}, NOT 2023-10-04.\n\n"
    )
    
    current_system_prompt = SYSTEM_PROMPT + date_context
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
            print(f"[{get_timestamp()}] [DEBUG] OpenAI Request - Messages count: {len(openai_messages)}")
            if openai_messages:
                system_msg = next((m for m in openai_messages if m.get("role") == "system"), None)
                if system_msg:
                    sys_content = system_msg.get("content", "")
                    print(f"[{get_timestamp()}] [DEBUG] System prompt length: {len(sys_content)} chars")
                    if "Available Tools" in sys_content:
                        # Extract tool count from system prompt
                        import re
                        tool_matches = re.findall(r'"name":\s*"([^"]+)"', sys_content)
                        if tool_matches:
                            print(f"[{get_timestamp()}] [DEBUG] Tools in system prompt: {len(tool_matches)} tools")
                            print(f"[{get_timestamp()}] [DEBUG] Tool names: {', '.join(tool_matches[:5])}{'...' if len(tool_matches) > 5 else ''}")
                # Log last user message preview
                user_msgs = [m for m in openai_messages if m.get("role") == "user"]
                if user_msgs:
                    last_user = user_msgs[-1].get("content", "")[:200]
                    print(f"[{get_timestamp()}] [DEBUG] Last user message preview: {last_user}...")
            print(f"[{get_timestamp()}] [DEBUG] OpenAI Request - Model: gpt-4o-mini, Temperature: 0")
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                temperature=0
            )
            print(f"[{get_timestamp()}] [LLM] OpenAI response received ({format_duration(request_start)})")
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
    print(f"[{get_timestamp()}] DEBUG: Raw LLM Response: {repr(response_content)}")
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
