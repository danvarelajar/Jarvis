import json
import json
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any, List
import re
from datetime import datetime, timedelta
import subprocess
import time
import asyncio

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

LANGUAGE REQUIREMENT:
- You MUST respond in ENGLISH only. The user's language is English, and all responses must be in English.
- Do NOT respond in Arabic, Spanish, or any other language - ONLY English.

CRITICAL RULES:
- If the "MCP TOOL DOCUMENTATION" section is empty or says "No tools available", respond with TEXT only (no JSON)
- If the user's question is conversational (greetings, "how are you", general questions without @server_name), respond with TEXT only (no JSON)
- ONLY call tools if: (1) tools are listed in the documentation AND (2) the user's request clearly requires a tool
- NEVER invent or hallucinate tool names - only use tools that are explicitly listed in the documentation
- Use the EXACT tool name as provided in the MCP tool documentation
- Use the EXACT parameter names from the Input Schema - do NOT use synonyms or alternative names
- Check the Input Schema properties to see the exact parameter names required (e.g., if schema says "from", use "from" not "origin")
- DO NOT add parameters that are NOT listed in the Input Schema (e.g., if schema doesn't have "city", do NOT add it)
- If a parameter is not in the "ALL PARAMETERS" list, DO NOT include it in your tool call arguments
- If you already have all the information needed, return a TEXT response (not JSON)
- If you see "System Error: You have already called tool..." - STOP calling tools and return a text response

OUTPUT FORMAT:
- For tool calls (ONLY when tools are available AND needed): {"tool": "exact_tool_name", "arguments": {"param1": "value1"}}
  **CRITICAL: ALL parameters MUST be inside the "arguments" object. Do NOT put parameters at the top level.**
  **CORRECT:** {"tool": "weather__get_complete_forecast", "arguments": {"latitude": 40.4, "longitude": -3.7}}
  **WRONG:** {"tool": "weather__get_complete_forecast", "latitude": 40.4, "longitude": -3.7}
- For text responses (when no tools needed or no tools available): Just return plain text (no JSON, no code blocks)
"""

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.
YOUR GOAL: Execute the user's intent as EFFICIENTLY as possible.

LANGUAGE REQUIREMENT:
- You MUST respond in ENGLISH only. The user's language is English, and all responses must be in English.
- Do NOT respond in Arabic, Spanish, or any other language - ONLY English.

RESPONSE GUIDELINES:
1. CALL TOOLS IMMEDIATELY - no conversational preamble. If the user asks for something that requires a tool, output ONLY the JSON tool call (no "I'll help you..." or "Let me check..." text).
   (Reason: Prevents wasted turns on acknowledgments, gets results faster, better user experience)
   - This applies to intermediate turns when tools are needed (multi-turn scenarios)
   - The final turn (after tool execution) should return formatted text, not JSON
2. WHEN NO TOOL IS NEEDED, respond with plain text (not JSON). For conversational questions, greetings, or requests that don't require tools, just answer naturally.
3. FORMAT FINAL RESPONSES IN MARKDOWN. After tool execution completes, format your final answer using markdown (headers, lists, code blocks, etc.) for readability.

TOOL USAGE FORMAT:
You MUST output a VALID JSON object in this exact format: {"tool": "tool_name", "arguments": {"key": "value"}}
**CRITICAL: ALL parameters MUST be inside the "arguments" object. Do NOT put parameters at the top level.**
**CORRECT:** {"tool": "weather__get_complete_forecast", "arguments": {"latitude": 40.4, "longitude": -3.7}}
**WRONG:** {"tool": "weather__get_complete_forecast", "latitude": 40.4, "longitude": -3.7}
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

def get_current_date() -> tuple[str, str]:
    """
    Gets the current date and time via shell command.
    
    Returns:
        Tuple of (current_date, current_datetime) as strings
    """
    try:
        result = subprocess.run(
            ["date", "+%Y-%m-%d"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=2
        )
        current_date = result.stdout.strip()
        
        result2 = subprocess.run(
            ["date", "+%Y-%m-%d %H:%M:%S"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=2
        )
        current_datetime = result2.stdout.strip()
        
        if not current_date or not current_datetime:
            raise Exception("Command failed")
            
        return current_date, current_datetime
    except Exception as e:
        print(f"[WARNING] Date command failed, using fallback: {e}")
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
    import re
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
    
    # Check for "until [day]" or "[day]" patterns (e.g., "until Friday", "Friday")
    for day_name, day_num in days_of_week.items():
        # Pattern: "until [day]" or "until [day abbreviation]" (e.g., "until Friday", "until Fri")
        if f'until {day_name}' in user_lower or f'until {day_name[:3]}' in user_lower:
            # Find next occurrence of this day (including today if today is that day)
            days_ahead = (day_num - today.weekday()) % 7
            # If today is that day (days_ahead == 0), use today; otherwise use the calculated day
            target_day = today + timedelta(days=days_ahead)
            specific_dates.append(f"- When the user says 'until {day_name}', use: {target_day.strftime('%Y-%m-%d')}")
        
        # Pattern: standalone "[day]" (e.g., "Friday", "Fri") - only if not already matched
        # Check if day name appears as a standalone word (not part of "next", "this", "until")
        day_pattern = r'\b' + day_name + r'\b'
        if re.search(day_pattern, user_lower) and f'next {day_name}' not in user_lower and f'this {day_name}' not in user_lower and f'until {day_name}' not in user_lower:
            # Find next occurrence of this day (including today if today is that day)
            days_ahead = (day_num - today.weekday()) % 7
            # If today is that day (days_ahead == 0), use today; otherwise use the calculated day
            target_day = today + timedelta(days=days_ahead)
            specific_dates.append(f"- When the user says '{day_name}', use: {target_day.strftime('%Y-%m-%d')}")
    
    # Parse explicit date formats: "26th December", "26 December", "December 26"
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Pattern: "26th December" or "26 December" or "December 26"
    for month_name, month_num in months.items():
        # "26th December" or "26 December"
        pattern1 = re.search(r'(\d+)(?:st|nd|rd|th)?\s+' + month_name, user_lower)
        if pattern1:
            day = int(pattern1.group(1))
            year = today.year
            # If the date is in the past this year, assume next year
            try:
                parsed_date = datetime(year, month_num, day)
                if parsed_date < today:
                    year += 1
                    parsed_date = datetime(year, month_num, day)
                specific_dates.append(f"- When the user says '{pattern1.group(0)}', use: {parsed_date.strftime('%Y-%m-%d')}")
            except ValueError:
                pass  # Invalid date (e.g., Feb 30)
        
        # "December 26" or "December 26th"
        pattern2 = re.search(month_name + r'\s+(\d+)(?:st|nd|rd|th)?', user_lower)
        if pattern2:
            day = int(pattern2.group(1))
            year = today.year
            try:
                parsed_date = datetime(year, month_num, day)
                if parsed_date < today:
                    year += 1
                    parsed_date = datetime(year, month_num, day)
                specific_dates.append(f"- When the user says '{pattern2.group(0)}', use: {parsed_date.strftime('%Y-%m-%d')}")
            except ValueError:
                pass
    
    # Parse DD/MM/YYYY or MM/DD/YYYY format: "07/01/2026"
    date_pattern = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', user_query)  # Use original case for exact match
    if date_pattern:
        part1, part2, year = int(date_pattern.group(1)), int(date_pattern.group(2)), int(date_pattern.group(3))
        # Try both formats and use the one that makes sense (future date, not too far in past)
        dd_mm_yyyy_valid = False
        mm_dd_yyyy_valid = False
        dd_mm_date = None
        mm_dd_date = None
        
        # Try DD/MM/YYYY first
        try:
            dd_mm_date = datetime(year, part2, part1)
            if dd_mm_date >= today - timedelta(days=30):  # Allow dates up to 30 days in past
                dd_mm_yyyy_valid = True
        except ValueError:
            pass
        
        # Try MM/DD/YYYY
        try:
            mm_dd_date = datetime(year, part1, part2)
            if mm_dd_date >= today - timedelta(days=30):  # Allow dates up to 30 days in past
                mm_dd_yyyy_valid = True
        except ValueError:
            pass
        
        # Prefer the format that gives a future date (or closer to today if both are valid)
        # For checkout dates, prefer MM/DD/YYYY (more common in US/booking contexts)
        if dd_mm_yyyy_valid and mm_dd_yyyy_valid:
            # Both valid - check if this looks like a checkout date (after "checkout" or "check-out")
            is_checkout_date = "checkout" in user_lower or "check-out" in user_lower
            if is_checkout_date:
                # For checkout dates, prefer MM/DD/YYYY (Feb 1, 2026) over DD/MM/YYYY (Jan 2, 2026)
                # unless DD/MM makes more sense (much closer to today)
                days_diff_dd_mm = abs((dd_mm_date - today).days)
                days_diff_mm_dd = abs((mm_dd_date - today).days)
                # If MM/DD is within reasonable range (not too far), prefer it for checkout
                if mm_dd_date > today and days_diff_mm_dd < 60:
                    specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {mm_dd_date.strftime('%Y-%m-%d')} (interpreted as MM/DD/YYYY - checkout date)")
                elif days_diff_dd_mm < days_diff_mm_dd:
                    specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {dd_mm_date.strftime('%Y-%m-%d')} (interpreted as DD/MM/YYYY)")
                else:
                    specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {mm_dd_date.strftime('%Y-%m-%d')} (interpreted as MM/DD/YYYY)")
            else:
                # Not a checkout date - prefer the one closer to today
                if abs((dd_mm_date - today).days) < abs((mm_dd_date - today).days):
                    specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {dd_mm_date.strftime('%Y-%m-%d')} (interpreted as DD/MM/YYYY)")
                else:
                    specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {mm_dd_date.strftime('%Y-%m-%d')} (interpreted as MM/DD/YYYY)")
        elif dd_mm_yyyy_valid:
            specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {dd_mm_date.strftime('%Y-%m-%d')} (interpreted as DD/MM/YYYY)")
        elif mm_dd_yyyy_valid:
            specific_dates.append(f"- When the user says '{date_pattern.group(0)}', use: {mm_dd_date.strftime('%Y-%m-%d')} (interpreted as MM/DD/YYYY)")
    
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

def format_tool_registry(tools: List[dict]) -> str:
    """
    Formats tools as a concise semantic signature list for token efficiency.
    Returns: "tool_name(param1:type, param2:type) - description"
    """
    if not tools:
        return "No tools available."

    registry = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")

        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        params = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            marker = "*" if param_name in required else ""
            params.append(f"{param_name}{marker}:{param_type}")

        sig = f"{name}({', '.join(params)})" if params else name
        registry.append(f"{sig} - {desc}")

    return "\n".join(registry)

def _normalize_ollama_base_url(url: str) -> str:
    """Normalize Ollama URL to base (no path)."""
    base = url.strip()
    for suffix in ["/api/chat", "/api/generate", "/v1", "/v1/"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base.rstrip("/")


async def query_ollama(messages: list, system_prompt: str, model_url: str, model_name: str = "") -> str:
    """
    Queries a local Ollama instance via the OpenAI-compatible /v1/chat/completions API.
    
    Args:
        messages: List of message dicts
        system_prompt: System prompt to use
        model_url: Ollama server URL
        model_name: Model name to use
    """
    if not model_url:
        return "Error: Ollama URL is not set."
    
    if not model_name or model_name.strip() == "":
        return "Error: Model name is not set. Please select a model in the settings."

    # Build OpenAI-format messages
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        if role == "model":
            role = "assistant"
        ollama_messages.append({"role": role, "content": msg.get("content", "")})

    total_chars = sum(len(str(msg.get("content", ""))) for msg in ollama_messages)
    print(f"[{get_timestamp()}] DEBUG: Using Ollama model: {model_name}")
    print(f"[{get_timestamp()}] [LLM] Using OpenAI-compatible /v1/chat/completions API")
    print(f"[{get_timestamp()}] [LLM] Total prompt/message chars: {total_chars}")

    base_url = _normalize_ollama_base_url(model_url)
    openai_base = f"{base_url}/v1"

    try:
        from openai import AsyncOpenAI

        # Check if model is already loaded (Ollama-specific, optional)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as check_client:
                ps_response = await check_client.get(f"{base_url}/api/ps")
                if ps_response.status_code == 200:
                    ps_data = ps_response.json()
                    models_loaded = ps_data.get("models", [])
                    model_loaded = any(m.get("name", "").startswith(model_name) for m in models_loaded)
                    if model_loaded:
                        print(f"[{get_timestamp()}] [LLM] ✓ Model '{model_name}' is already loaded in memory")
                    else:
                        print(f"[{get_timestamp()}] [LLM] ⚠️  Model '{model_name}' is NOT loaded - will need to load from disk (~4s delay)")
        except Exception as e:
            print(f"[{get_timestamp()}] [LLM] Could not check if model is loaded: {e}")

        request_start = time.time()
        print(f"[{get_timestamp()}] [LLM] Sending request to Ollama via OpenAI spec (base: {openai_base})...")
        print(f"[{get_timestamp()}] [LLM] Waiting for Ollama inference (this may take 2-3 minutes if model needs to load)...")

        client = AsyncOpenAI(
            base_url=openai_base,
            api_key="ollama",  # required by client but ignored by Ollama
        )

        # Retry logic for 404 (model loading)
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=ollama_messages,
                    temperature=0,
                )
                http_time = time.time() - request_start
                if http_time > 60:
                    print(f"[{get_timestamp()}] [LLM] ⚠️  SLOW: Inference took {http_time:.1f}s")
                elif http_time > 30:
                    print(f"[{get_timestamp()}] [LLM] ⚠️  MODERATE: Inference took {http_time:.1f}s")
                else:
                    print(f"[{get_timestamp()}] [LLM] ✓ Inference completed in {http_time:.1f}s")
                print(f"[{get_timestamp()}] [LLM] Ollama response received (total: {format_duration(request_start)})")
                content = response.choices[0].message.content
                if content:
                    print(f"[{get_timestamp()}] [LLM] Response content length: {len(content)} chars")
                else:
                    print(f"[{get_timestamp()}] [LLM] WARNING: Response content is empty!")
                return content or ""
            except Exception as e:
                err_str = str(e).lower()
                if ("404" in err_str or "not found" in err_str) and attempt < max_retries - 1:
                    print(f"[{get_timestamp()}] [LLM] ⚠️  Got 404, retrying in 5 seconds (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(5)
                    continue
                raise
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "403" in err_msg:
            # OpenAI client may surface auth errors; Ollama ignores api_key
            pass
        print(f"Ollama Error: {err_msg}")
        return f"Error communicating with Ollama at {model_url}: {err_msg}"

import time

# Global Rate Limiter
LAST_REQUEST_TIME = 0
RATE_LIMIT_INTERVAL = 15  # 15 seconds (4 requests/min) to be safe under 5 RPM limit

async def query_llm(messages: list, tools: list = None, api_key: str = None, provider: str = "openai", model_url: str = None, model_name: str = "", use_qwen_rag: bool = False, user_query: str = "") -> str:
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
        # /api/chat approach: let Ollama apply model templates internally.
        # We provide a normal system prompt + message list (no manual control tokens).
            current_date, current_datetime = get_current_date()
            
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
            
            # Get user query for date calculations
            user_query_for_dates = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_query_for_dates = msg.get("content", "")
                    break
            
            # Calculate specific dates from user query
            specific_dates_context = calculate_specific_dates(user_query_for_dates, current_date, today)
            
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
            
            if specific_dates_context:
                date_context += f"\nSPECIFIC DATE CALCULATIONS FROM USER QUERY:\n{specific_dates_context}\n"
            
            date_context += (
                f"IMPORTANT: The current year is {current_year}. "
                f"DO NOT use dates from 2023 or earlier. Always calculate relative dates from TODAY ({current_date}). "
                f"Example: If today is {current_date} and user says 'tomorrow', use {tomorrow_str}, NOT 2023-10-04.\n\n"
            )
            
            ollama_system_prompt = SYSTEM_PROMPT + date_context
            if tools:
                # List exact tool names first to prevent hallucination
                exact_tool_names = [tool.get('name', 'unknown') for tool in tools]
                tool_names_list = "\n".join([f"  - `{name}`" for name in exact_tool_names])
                ollama_system_prompt += f"\n\n## AVAILABLE TOOLS:\n\n**CRITICAL: EXACT TOOL NAMES (use EXACTLY as shown):**\n{tool_names_list}\n\n"
                ollama_system_prompt += f"**YOU MUST use ONLY these exact tool names. Do NOT invent, modify, or hallucinate tool names.**\n"
                ollama_system_prompt += f"**Example: If you see 'weather__search_location', use EXACTLY 'weather__search_location', NOT 'weather__get_location' or 'weather__find_location'.**\n\n"
                
                tool_descriptions = json.dumps(tools, indent=2)
                ollama_system_prompt += f"**Full Tool Definitions (JSON Format):**\n```json\n{tool_descriptions}\n```\n\n"
                
                ollama_system_prompt += f"You MUST use these tools to answer queries. Use the EXACT tool names listed above. Do not say you cannot access them. Just output the JSON to call them."
                
                # Global tool rules (single place, to reduce prompt size and leverage recency for small models)
                ollama_system_prompt += (
                    "\n### GLOBAL TOOL RULES (MANDATORY)\n"
                    "1. ONLY use tools if the user used the @server_name prefix (e.g., @weather, @booking).\n"
                    "2. Use EXACT tool names and parameter names from the documentation. NO synonyms. NO extra parameters.\n"
                    "3. JSON format for tool calls: {\"tool\": \"exact_tool_name\", \"arguments\": {\"param\": \"value\"}}.\n"
                    "4. If no tools are available or the user did NOT use @server_name, respond with TEXT only (no JSON).\n"
                    "5. Do NOT add parameters that are not listed. Example forbidden extras: adults, guests, people, persons.\n"
                )
                # Weather flow guidance (two-step) for legacy path
                has_weather_tools = any("weather__" in (t.get("name") or "") for t in tools or [])
                if has_weather_tools:
                    ollama_system_prompt += (
                        "### WEATHER FLOW (TWO-STEP)\n"
                        "Step 1: Call weather__search_location with the city/location name from the user.\n"
                        "  Example: {\"tool\": \"weather__search_location\", \"arguments\": {\"city\": \"Madrid\"}}\n"
                        "Step 2: After you get coordinates, call weather__get_complete_forecast with EXACT latitude and longitude from step 1.\n"
                        "  Example: {\"tool\": \"weather__get_complete_forecast\", \"arguments\": {\"latitude\": 40.4168, \"longitude\": -3.7038}}\n"
                        "Rules: Do NOT hallucinate coordinates. Do NOT pass 'location' to weather__get_complete_forecast. Use only the coordinates returned by weather__search_location.\n\n"
                    )
            else:
                # No tools available - emphasize conversational response
                ollama_system_prompt += "\n\n## AVAILABLE TOOLS:\nNo tools are available. Respond with plain text only. Do NOT output JSON. Do NOT try to call or invent tools."
            
            return await query_ollama(messages, ollama_system_prompt, model_url, model_name=model_name)

    # Construct the full prompt including system instructions (for OpenAI)
    current_date, current_datetime = get_current_date()
    
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
    
    date_context = (
        f"\n## CURRENT DATE AND TIME (CRITICAL - USE THESE DATES):\n"
        f"Today's date: {current_date}\n"
        f"Current date and time: {current_datetime}\n\n"
        f"DATE CALCULATIONS:\n"
        f"- When the user says 'today', use: {current_date}\n"
        f"- When the user says 'tomorrow', use: {tomorrow_str}\n"
        f"- When the user says 'day after tomorrow' or 'after tomorrow', use: {day_after_str}\n"
        f"- When the user says 'next week', add 7 days to {current_date}\n\n"
        f"IMPORTANT: The current year is {current_year}. "
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
            openai_start = time.time()
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                temperature=0
            )
            print(f"[{get_timestamp()}] [LLM] OpenAI response received ({format_duration(openai_start)})")
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
    
    # Handle empty responses
    if not response_content or not response_content.strip():
        error_msg = "LLM returned an empty response. The model may not have generated any output. Please try again or check the model configuration."
        print(f"[{get_timestamp()}] [PARSE] ERROR: {error_msg}", flush=True)
        return {"type": "error", "message": error_msg}
    
    try:
        # Attempt to find JSON object using regex
        import re
        
        # RECOMMENDATION 1 & 4: Strip HTML comments BEFORE parsing JSON
        # This prevents embedded data (like LOCATIONS_DATA) from being parsed as tool calls
        clean_content = response_content.strip()
        # Remove HTML comments (e.g., <!-- LOCATIONS_DATA: [...] -->)
        clean_content = re.sub(r'<!--[\s\S]*?-->', '', clean_content)
        print(f"[{get_timestamp()}] [PARSE] Stripped HTML comments from response", flush=True)
        
        # First, clean markdown code blocks if present
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:].strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content[3:].strip()
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3].strip()
        
        # Check for <start_function_call> format (gemma3-mcp model)
        function_call_match = re.search(r'<start_function_call>(.*?)<end_function_call>', clean_content, re.DOTALL)
        if function_call_match:
            json_str = function_call_match.group(1).strip()
            print(f"[{get_timestamp()}] [PARSE] Detected <start_function_call> format", flush=True)
        else:
            # Look for a JSON object structure: { ... }
            # Use a more robust regex that handles nested braces
            # RECOMMENDATION 3: Since HTML comments are already stripped, any JSON found should be from the actual response
            # We'll validate it has "tool" and "arguments" fields after parsing
            
            # Find first JSON object by counting braces
            brace_start = clean_content.find('{')
            if brace_start != -1:
                # Find matching closing brace by counting braces
                brace_count = 0
                brace_end = brace_start
                for i in range(brace_start, len(clean_content)):
                    if clean_content[i] == '{':
                        brace_count += 1
                    elif clean_content[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i + 1
                            break
                if brace_count == 0:
                    json_str = clean_content[brace_start:brace_end]
                else:
                    # Fallback: use regex
                    content_for_regex = clean_content.replace('\n', ' ')
                    match = re.search(r'(\{.*\})', content_for_regex, re.DOTALL)
                    json_str = match.group(1) if match else clean_content.strip()
            else:
                json_str = clean_content.strip()

        data = json.loads(json_str)
        
        # RECOMMENDATION 2 & 3: Only treat JSON as tool call if it has both "tool" and "arguments" fields
        # Also detect embedded data patterns (like LOCATIONS_DATA) and treat as text, not error
        if "tool" not in data:
            # Check if this looks like embedded data (e.g., location data with "index", "name", "latitude", etc.)
            is_embedded_data = False
            if isinstance(data, list):
                # Arrays are likely embedded data (e.g., list of locations)
                is_embedded_data = True
                print(f"[{get_timestamp()}] [PARSE] Detected JSON array - likely embedded data, treating as text", flush=True)
            elif isinstance(data, dict):
                # Check for common embedded data patterns
                embedded_data_keys = ["index", "name", "latitude", "longitude", "country", "state", "full_data"]
                if any(key in data for key in embedded_data_keys):
                    is_embedded_data = True
                    print(f"[{get_timestamp()}] [PARSE] Detected embedded data pattern (has keys like 'index', 'name', 'latitude'), treating as text", flush=True)
            
            if is_embedded_data:
                # This is embedded data, not a tool call - treat as text response
                print(f"[{get_timestamp()}] [PARSE] JSON appears to be embedded data, not a tool call. Treating response as text.", flush=True)
                return {"type": "text", "content": response_content}
            
            # This is a common error - LLM returns JSON but not in tool call format
            error_msg = (
                f"LLM format error: You returned JSON but it's missing the required 'tool' and 'arguments' fields. "
                f"You returned: {json.dumps(data)[:200]}. "
                f"You MUST use this EXACT format: {{\"tool\": \"tool_name\", \"arguments\": {{\"param\": \"value\"}}}}. "
                f"Put ALL parameters inside the 'arguments' object. "
                f"Do NOT return JSON like {{\"origin\": \"...\", \"destination\": \"...\"}}. "
                f"You MUST wrap it as {{\"tool\": \"booking__create_itinerary\", \"arguments\": {{\"from\": \"...\", \"to\": \"...\"}}}}."
            )
            print(f"[{get_timestamp()}] [PARSE] {error_msg}", flush=True)
            return {"type": "error", "message": error_msg}
        
        # Check if tool call format is malformed (parameters at top level instead of in "arguments")
        if "tool" in data and "arguments" not in data:
            # Try to fix: move all non-"tool" fields into "arguments"
            tool_name = data.pop("tool")
            arguments = data  # Everything else becomes arguments
            data = {"tool": tool_name, "arguments": arguments}
            print(f"[{get_timestamp()}] [PARSE] Fixed malformed tool call: moved parameters into 'arguments' object", flush=True)
        
        # Validate with Pydantic
        try:
            tool_call = ToolCall(**data)
            return {"type": "tool_call", "data": tool_call}
        except ValidationError as ve:
            # If validation fails, provide helpful error
            print(f"[{get_timestamp()}] [PARSE] Tool call validation failed: {ve}", flush=True)
            # Note: "tool" not in data case is already handled above
            if "tool" in data:
                return {"type": "error", "message": f"Invalid tool call format. Expected {{'tool': 'name', 'arguments': {{...}}}}. Got: {json.dumps(data)[:200]}"}
            raise  # Re-raise to be caught by outer except
        
    except (json.JSONDecodeError, ValidationError):
        # RECOMMENDATION 3: Improved error handling - detect embedded data vs tool call attempts
        # Check if response contains HTML comments with embedded data
        if "<!--" in response_content and "LOCATIONS_DATA" in response_content:
            print(f"[{get_timestamp()}] [PARSE] Response contains embedded data in HTML comment, treating as text", flush=True)
            return {"type": "text", "content": response_content}
        
        # If it's not valid JSON or doesn't match the schema, treat as text
        # But if it looks like it tried to be JSON (starts with {), return error
        # However, only return error if it's clearly a tool call attempt (not embedded data)
        stripped = response_content.strip()
        if (stripped.startswith("{") or "```json" in response_content) and "LOCATIONS_DATA" not in response_content:
            error_msg = "LLM format error: expected tool call JSON like {\"tool\": \"tool_name\", \"arguments\": {...}}. Put ALL parameters inside the 'arguments' object."
            print(f"[{get_timestamp()}] [PARSE] {error_msg}", flush=True)
            return {"type": "error", "message": error_msg}
        
        return {"type": "text", "content": response_content}
