# How MCP Tools are Fed to the LLM

## Overview

This document explains how MCP (Model Context Protocol) tools are formatted and sent to the LLM, and why the LLM sometimes hallucinates parameters.

---

## Tool Flow

### 1. Tool Discovery (`backend/main.py`)

**Defender Mode** (Lines 559-570):
```python
if agent_mode == "defender" and tools:
    sanitized = []
    for t in tools:
        sanitized.append({
            "name": t.get("name"),
            "inputSchema": t.get("inputSchema", {})
        })
    tools = sanitized
```

**Key Point**: In defender mode, only `name` and `inputSchema` are kept. Descriptions are **stripped** to prevent prompt injection.

**Naive Mode**: Full tool object is kept (including descriptions).

### 2. Tool Formatting for LLM

There are **two approaches** depending on the provider and configuration:

#### A. Qwen RAG Approach (Ollama with `use_qwen_rag=True`)

**Location**: `backend/llm_service.py` lines 442-545

**When is it used?**
- Provider is `"ollama"` AND
- `use_qwen_rag=True` (automatically set when `llm_provider == "ollama"` in `main.py` line 643) AND
- `tools` list is not empty

**Process**:

**Step 1: RAG Retrieval** (Lines 456-457)
```python
# Retrieve top-k relevant tools based on user query
relevant_tools = retrieve_relevant_tools(user_query, tools, top_k=5)
```

The `retrieve_relevant_tools()` function (lines 218-276):
- Extracts keywords from user query
- Scores each tool based on:
  - Tool name matches (weight: 10)
  - Description matches (weight: 3)
  - Schema property matches (weight: 2)
  - Word overlap (weight: 1)
- Returns top 5 most relevant tools

**Why RAG?** Instead of sending ALL tools (which can be overwhelming), we only send the most relevant ones based on the user's query. This:
- Reduces token usage
- Focuses the LLM's attention
- Prevents information overload

**Step 2: Manual Formatting** (Lines 459-540)

Builds a structured markdown document with multiple sections:

**2a. Tool Names List** (Lines 462-468)
```python
tool_context += f"**CRITICAL: AVAILABLE TOOL NAMES (use EXACTLY as shown):**\n"
for name in exact_tool_names:
    tool_context += f"  - `{name}`\n"
```
**Purpose**: Prevents tool name hallucination by listing exact names upfront.

**2b. Per-Tool Documentation** (Lines 470-540)

For each retrieved tool, builds:
- **Tool Name & Description** (Lines 471-473)
- **Tool Sequence Warnings** (Lines 475-488) - Special handling for weather tools
- **Required Parameters List** (Lines 497-502)
- **All Parameters List** (Lines 504-511)
- **Full Input Schema JSON** (Line 513)
- **Critical Parameter Rules** (Lines 515-540) - Special restrictions for weather tools

**Step 3: System Prompt Assembly** (Lines 546-550)
```python
qwen_system_prompt = f"{MCP_ROUTER_SYSTEM_PROMPT}\n{date_context}\n{tool_context}\n\n..."
```

**Final Structure**:
```
MCP_ROUTER_SYSTEM_PROMPT (fixed instructions)
+
date_context (current date, tomorrow, etc.)
+
tool_context (retrieved tools documentation)
+
final instructions
```

**Advantages**:
- ✅ Only relevant tools are shown (RAG filtering)
- ✅ Highly structured and explicit
- ✅ Multiple reinforcement points
- ✅ Custom warnings for specific tools
- ✅ Better for instruction-following

**Disadvantages**:
- ❌ More complex code
- ❌ RAG might miss relevant tools
- ❌ More tokens used for formatting

**Example Output**:
```markdown
## MCP TOOL DOCUMENTATION:

**CRITICAL: AVAILABLE TOOL NAMES (use EXACTLY as shown):**
  - `weather__search_location`
  - `weather__get_complete_forecast`

### Tool: weather__get_complete_forecast
Description: No description

**🚨 CRITICAL TOOL SEQUENCE REQUIREMENT:**
1. This tool REQUIRES coordinates (latitude/longitude) - it does NOT accept location names
2. You MUST call 'weather__search_location' FIRST with the location name to get coordinates
...

**REQUIRED PARAMETERS (use EXACT names):**
  - `latitude` (number): Latitude coordinate
  - `longitude` (number): Longitude coordinate

**ALL PARAMETERS (use EXACT names from this list):**
  - `latitude` (number) [REQUIRED]: Latitude coordinate
  - `longitude` (number) [REQUIRED]: Longitude coordinate

**Full Input Schema (JSON):**
```json
{
  "type": "object",
  "properties": {
    "latitude": {"type": "number"},
    "longitude": {"type": "number"}
  },
  "required": ["latitude", "longitude"]
}
```

**🚨 CRITICAL PARAMETER RESTRICTIONS FOR THIS TOOL:**
1. This tool ONLY accepts these exact parameters: 'latitude', 'longitude'
2. This tool does NOT accept 'location', 'city', 'address', 'place', or any location name parameter
3. If you have a location name, you MUST call 'weather__search_location' FIRST to get coordinates
4. Do NOT add 'location' parameter - it will be rejected and cause an error
5. Do NOT invent or guess coordinates - you must get them from 'weather__search_location'
6. Use ONLY these parameters: 'latitude', 'longitude' - nothing else
```

#### B. Legacy Ollama Approach (Ollama with `use_qwen_rag=False`)

**Location**: `backend/llm_service.py` lines 603-660

**When is it used?**
- Provider is `"ollama"` AND
- `use_qwen_rag=False` OR `tools` is empty

**Note**: Currently, `use_qwen_rag` is always `True` for Ollama (line 643 in `main.py`), so this approach is rarely used. It's kept as a fallback.

**Process**:

**Step 1: Tool Names List** (Lines 621-626)
```python
exact_tool_names = [tool.get('name', 'unknown') for tool in tools]
tool_names_list = "\n".join([f"  - `{name}`" for name in exact_tool_names])
ollama_system_prompt += f"**CRITICAL: EXACT TOOL NAMES (use EXACTLY as shown):**\n{tool_names_list}\n\n"
```

**Step 2: Tool Sequence Warnings** (Lines 628-639)
```python
if has_weather_forecast and has_weather_search:
    ollama_system_prompt += f"**🚨 CRITICAL TOOL SEQUENCE REQUIREMENT:**\n"
    # ... sequence instructions
```

**Step 3: Full Tool JSON Dump** (Lines 641-642)
```python
tool_descriptions = json.dumps(tools, indent=2)
ollama_system_prompt += f"**Full Tool Definitions (JSON Format):**\n```json\n{tool_descriptions}\n```\n\n"
```

**Step 4: Parameter Restrictions** (Lines 644-657)
```python
# Add explicit parameter restrictions for weather__get_complete_forecast
weather_forecast_tool = next((t for t in tools if "weather__get_complete_forecast" in t.get('name', '')), None)
if weather_forecast_tool:
    # Extract allowed parameters and add restrictions
```

**Final Structure**:
```
SYSTEM_PROMPT (general instructions)
+
date_context (current date, tomorrow, etc.)
+
tool names list
+
tool sequence warnings
+
full tool JSON dump
+
parameter restrictions
+
final instructions
```

**Advantages**:
- ✅ Simpler code
- ✅ Shows ALL tools (no filtering)
- ✅ Full JSON schema visible
- ✅ Less token overhead

**Disadvantages**:
- ❌ Can be overwhelming with many tools
- ❌ Less structured than RAG approach
- ❌ LLM must parse JSON to understand tools
- ❌ No relevance filtering

#### C. OpenAI Approach

**Location**: `backend/llm_service.py` lines 585-603

**When is it used?**
- Provider is `"openai"`

**Process**:
1. Uses OpenAI's native function calling format
2. Tools are passed directly to OpenAI API as `tools` parameter
3. OpenAI handles tool documentation internally

**Code** (Lines 585-603):
```python
if provider == "openai":
    # ... prepare messages ...
    
    if tools:
        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            tools=openai_tools,  # OpenAI handles tool documentation
            tool_choice="auto"
        )
```

**Advantages**:
- ✅ Native OpenAI function calling (best support)
- ✅ OpenAI handles tool documentation internally
- ✅ Better instruction following
- ✅ Automatic tool selection

**Disadvantages**:
- ❌ Only works with OpenAI API
- ❌ Requires API key
- ❌ Costs money per request

**Example Output**:
```markdown
## AVAILABLE TOOLS:

**CRITICAL: EXACT TOOL NAMES (use EXACTLY as shown):**
  - `weather__search_location`
  - `weather__get_complete_forecast`

**🚨 CRITICAL TOOL SEQUENCE REQUIREMENT:**
1. When user asks for weather at a location name, you MUST call 'weather__search_location' FIRST
...

**Full Tool Definitions (JSON Format):**
```json
[
  {
    "name": "weather__search_location",
    "inputSchema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  },
  {
    "name": "weather__get_complete_forecast",
    "inputSchema": {
      "type": "object",
      "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      },
      "required": ["latitude", "longitude"]
    }
  }
]
```

**🚨 CRITICAL FOR 'weather__get_complete_forecast':**
- This tool ONLY accepts: 'latitude', 'longitude'
- This tool does NOT accept 'location', 'city', 'address', 'place', or any location name
- If you have a location name, call 'weather__search_location' FIRST
- Do NOT add 'location' parameter - it will be rejected
```

---

## Why LLM Hallucinates Parameters

### Root Causes

1. **Model Limitations**:
   - `gemma3:1B` is a very small model (1 billion parameters)
   - Small models struggle with following complex instructions
   - They tend to "fill in" parameters based on user query context

2. **Context Mismatch**:
   - User query: "what's the weather in Calera y Chozas, Spain"
   - LLM sees: location name in query → thinks it should pass it to the tool
   - LLM ignores: explicit instructions that tool only accepts coordinates

3. **Instruction Overload**:
   - Too many instructions can confuse small models
   - They may focus on user query context over system instructions

4. **Tool Schema Visibility**:
   - Even though we show exact parameters, the model may:
     - See "location" in user query
     - See "location" parameter in `weather__search_location` tool
     - Infer that `weather__get_complete_forecast` should also accept "location"

### What We're Doing to Fix It

1. **Explicit Parameter Restrictions**:
   - Added specific warnings: "This tool does NOT accept 'location' parameter"
   - Listed exact allowed parameters multiple times
   - Added negative examples (what NOT to do)

2. **Tool Sequence Documentation**:
   - Explicitly states: "call weather__search_location FIRST"
   - Explains why: "to get coordinates"
   - Warns: "Do NOT invent coordinates"

3. **Validation with Error Messages**:
   - When LLM adds invalid parameter, validation rejects it
   - Error message explicitly guides LLM to use correct tool sequence
   - Prevents retrying same invalid call

4. **Multiple Reinforcement Points**:
   - Tool name list
   - Tool sequence warnings
   - Parameter restrictions
   - Full schema JSON
   - Critical parameter rules
   - Error messages

---

## Current Tool Object Structure

### Full Tool Object (from MCP server):
```json
{
  "name": "weather__get_complete_forecast",
  "description": "Get complete weather forecast for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "latitude": {"type": "number", "description": "Latitude coordinate"},
      "longitude": {"type": "number", "description": "Longitude coordinate"}
    },
    "required": ["latitude", "longitude"]
  }
}
```

### Defender Mode (sanitized):
```json
{
  "name": "weather__get_complete_forecast",
  "inputSchema": {
    "type": "object",
    "properties": {
      "latitude": {"type": "number"},
      "longitude": {"type": "number"}
    },
    "required": ["latitude", "longitude"]
  }
}
```

**Note**: Description is stripped in defender mode.

---

## Recommendations

1. **For Better Results**:
   - Use larger models (e.g., `gemma3:8b` or `qwen3:8b` instead of `gemma3:1B`)
   - Use Qwen RAG approach (better tool documentation formatting)
   - Consider using OpenAI API (better instruction following)

2. **For Current Model**:
   - Keep explicit parameter restrictions
   - Add more negative examples
   - Simplify instructions (fewer but clearer)
   - Use validation to catch and correct errors

3. **Future Improvements**:
   - Add parameter validation before sending to LLM
   - Pre-process user query to detect location names and suggest tool sequence
   - Add few-shot examples showing correct tool usage

---

## Summary

**How tools are fed**:
- Defender mode: Only `name` and `inputSchema` (descriptions stripped)
- Qwen RAG: Manually formatted markdown with explicit parameter lists
- Legacy Ollama: JSON dump with tool sequence warnings

**Why hallucination happens**:
- Small model (`gemma3:1B`) struggles with complex instructions
- Context from user query overrides system instructions
- Model infers parameters from query context rather than schema

**What we're doing**:
- Multiple reinforcement points (warnings, restrictions, examples)
- Validation with corrective error messages
- Explicit tool sequence documentation
- Negative examples (what NOT to do)

---

## Detailed Comparison: Qwen RAG vs Legacy Ollama

### Example Scenario: User Query "what's the weather in Madrid, Spain"

**Input Tools** (from MCP server, after defender mode sanitization):
```json
[
  {
    "name": "weather__search_location",
    "inputSchema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  },
  {
    "name": "weather__get_complete_forecast",
    "inputSchema": {
      "type": "object",
      "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      },
      "required": ["latitude", "longitude"]
    }
  }
]
```

### Qwen RAG Approach - Step by Step

**Step 1: RAG Retrieval** (`retrieve_relevant_tools()`)
- **Input**: Query = "what's the weather in Madrid, Spain", Tools = [2 tools]
- **Process**:
  1. Extract keywords: `{"what", "weather", "madrid", "spain"}`
  2. Score `weather__search_location`:
     - Name match: "weather" in name → +10
     - Description match: (empty in defender mode) → +0
     - Schema match: "location" in schema → +2
     - Word overlap: 1 word → +1
     - **Total: 13**
  3. Score `weather__get_complete_forecast`:
     - Name match: "weather" in name → +10
     - Description match: (empty) → +0
     - Schema match: "weather" not in schema → +0
     - Word overlap: 1 word → +1
     - **Total: 11**
  4. Sort by score: `weather__search_location` (13) > `weather__get_complete_forecast` (11)
  5. Return top 2 (both tools)

**Step 2: Manual Formatting** (Lines 459-540)

**Output Structure** (~1500 tokens):
```markdown
## MCP TOOL DOCUMENTATION:

**CRITICAL: AVAILABLE TOOL NAMES (use EXACTLY as shown):**
  - `weather__search_location`
  - `weather__get_complete_forecast`

### Tool: weather__search_location
Description: No description

**🚨 CRITICAL: This is the FIRST tool you must call when the user asks for weather at a location name.**
1. Call this tool FIRST with the location name
2. Use the coordinates from the result to call 'weather__get_complete_forecast'
3. Do NOT skip this step - you cannot get weather without coordinates

**REQUIRED PARAMETERS (use EXACT names):**
  - `location` (string): 

**ALL PARAMETERS (use EXACT names from this list):**
  - `location` (string) [REQUIRED]: 

**Full Input Schema (JSON):**
```json
{
  "type": "object",
  "properties": {
    "location": {"type": "string"}
  },
  "required": ["location"]
}
```

**CRITICAL PARAMETER RULES:**
1. Use ONLY these exact parameter names: 'location'
2. Do NOT use synonyms (e.g., 'origin'/'destination'/'city' - use 'from'/'to')
3. Do NOT add parameters that are NOT in this list (e.g., do NOT add 'city' if it's not listed)
4. If a parameter is not in the list above, DO NOT include it in your tool call
5. Example: If the list shows 'from', 'to', 'departDate', 'returnDate', 'passengers' - use ONLY these 5 parameters, nothing else

### Tool: weather__get_complete_forecast
Description: No description

**🚨 CRITICAL TOOL SEQUENCE REQUIREMENT:**
1. This tool REQUIRES coordinates (latitude/longitude) - it does NOT accept location names
2. You MUST call 'weather__search_location' FIRST with the location name to get coordinates
3. Then use the coordinates from that result to call this tool
4. Do NOT invent, hallucinate, or guess coordinates
5. Do NOT add 'location' parameter to this tool - it only accepts latitude and longitude
**If you have a location name but no coordinates, call 'weather__search_location' first!**

**REQUIRED PARAMETERS (use EXACT names):**
  - `latitude` (number): 
  - `longitude` (number): 

**ALL PARAMETERS (use EXACT names from this list):**
  - `latitude` (number) [REQUIRED]: 
  - `longitude` (number) [REQUIRED]: 

**Full Input Schema (JSON):**
```json
{
  "type": "object",
  "properties": {
    "latitude": {"type": "number"},
    "longitude": {"type": "number"}
  },
  "required": ["latitude", "longitude"]
}
```

**🚨 CRITICAL PARAMETER RESTRICTIONS FOR THIS TOOL:**
1. This tool ONLY accepts these exact parameters: 'latitude', 'longitude'
2. This tool does NOT accept 'location', 'city', 'address', 'place', or any location name parameter
3. If you have a location name, you MUST call 'weather__search_location' FIRST to get coordinates
4. Do NOT add 'location' parameter - it will be rejected and cause an error
5. Do NOT invent or guess coordinates - you must get them from 'weather__search_location'
6. Use ONLY these parameters: 'latitude', 'longitude' - nothing else
```

**Key Characteristics**:
- ✅ **Highly structured**: Markdown with clear sections
- ✅ **Multiple reinforcements**: Same info repeated 6+ times in different formats
- ✅ **Explicit negatives**: "Do NOT add 'location' parameter" stated 3 times
- ✅ **Custom warnings**: Special handling for weather tools
- ✅ **Token count**: ~1500 tokens for 2 tools

### Legacy Ollama Approach - Step by Step

**Output Structure** (~800 tokens):
```markdown
## AVAILABLE TOOLS:

**CRITICAL: EXACT TOOL NAMES (use EXACTLY as shown):**
  - `weather__search_location`
  - `weather__get_complete_forecast`

**🚨 CRITICAL TOOL SEQUENCE REQUIREMENT:**
1. When user asks for weather at a location name, you MUST call 'weather__search_location' FIRST
2. 'weather__search_location' accepts location names and returns coordinates
3. 'weather__get_complete_forecast' ONLY accepts coordinates (latitude/longitude) - it does NOT accept location names
4. Use the coordinates from 'weather__search_location' result to call 'weather__get_complete_forecast'
5. Do NOT invent, hallucinate, or guess coordinates
6. Do NOT add 'location' parameter to 'weather__get_complete_forecast' - it will be rejected
**If you have a location name but no coordinates, call 'weather__search_location' first!**

**Full Tool Definitions (JSON Format):**
```json
[
  {
    "name": "weather__search_location",
    "inputSchema": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  },
  {
    "name": "weather__get_complete_forecast",
    "inputSchema": {
      "type": "object",
      "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      },
      "required": ["latitude", "longitude"]
    }
  }
]
```

**🚨 CRITICAL FOR 'weather__get_complete_forecast':**
- This tool ONLY accepts: 'latitude', 'longitude'
- This tool does NOT accept 'location', 'city', 'address', 'place', or any location name
- If you have a location name, call 'weather__search_location' FIRST
- Do NOT add 'location' parameter - it will be rejected
```

**Key Characteristics**:
- ✅ **Simpler**: Less code, more compact
- ✅ **Full JSON**: Complete schema visible in one place
- ✅ **Token efficient**: ~800 tokens (almost half of RAG approach)
- ❌ **Less structured**: LLM must parse JSON
- ❌ **Fewer reinforcements**: Info stated 2-3 times vs 6+ times
- ❌ **No relevance filtering**: Shows all tools even if not relevant

### Why Qwen RAG is Better for Small Models Like `gemma3:1B`

**1. Explicit Structure vs JSON Parsing**:
- **RAG**: Markdown lists are easier for small models to parse
- **Legacy**: JSON requires model to understand structure and extract info

**2. Multiple Reinforcements**:
- **RAG**: Same restriction stated 6+ times in different formats:
  1. Tool sequence warning
  2. Required parameters list
  3. All parameters list
  4. Full schema JSON
  5. Critical parameter restrictions (6 points)
  6. Final instructions
- **Legacy**: Stated 2-3 times

**3. Negative Examples**:
- **RAG**: "Do NOT add 'location' parameter" appears 3 times explicitly
- **Legacy**: Appears 1-2 times

**4. Relevance Filtering**:
- **RAG**: Only shows tools relevant to query (reduces cognitive load)
- **Legacy**: Shows all tools (can be overwhelming)

**Trade-off**: RAG uses ~2x more tokens, but significantly better instruction following for small models.

### Token Usage Comparison

For 2 weather tools:

| Approach | Tokens | Structure | Reinforcements | Best For |
|----------|--------|-----------|----------------|----------|
| **Qwen RAG** | ~1500 | Markdown | 6+ times | Small models, complex tools |
| **Legacy Ollama** | ~800 | JSON | 2-3 times | Large models, simple tools |
| **OpenAI** | ~400* | Native | Internal | OpenAI models |

*OpenAI handles tool documentation internally, so token count is lower.

---

## Code Flow Diagram

```
User Query: "what's the weather in Madrid"
    ↓
main.py: chat() endpoint
    ↓
1. Tool Discovery (main.py:559-570)
   - Defender mode: Strip descriptions → Keep only name + inputSchema
   - Naive mode: Keep full tool object
    ↓
2. Determine Formatting Approach (main.py:643)
   - use_qwen_rag = (provider == "ollama")  # Always True for Ollama
    ↓
3a. Qwen RAG Path (llm_service.py:442-545)
   ├─ RAG Retrieval (retrieve_relevant_tools)
   │  └─ Score tools → Return top 5
   ├─ Manual Formatting
   │  ├─ Tool names list
   │  ├─ Per-tool documentation
   │  │  ├─ Sequence warnings
   │  │  ├─ Parameter lists
   │  │  ├─ Full schema JSON
   │  │  └─ Critical restrictions
   │  └─ Build markdown context
   └─ Assemble system prompt
      └─ MCP_ROUTER_SYSTEM_PROMPT + date_context + tool_context
    ↓
3b. Legacy Ollama Path (llm_service.py:587-660)
   ├─ Tool names list
   ├─ Tool sequence warnings
   ├─ Full JSON dump
   └─ Parameter restrictions
    ↓
4. Send to LLM (query_ollama)
   - Format: System message with full prompt
   - User messages appended
    ↓
5. LLM Response
   - Parse JSON tool call
   - Validate against schema
   - Execute tool
   - Return result
```

---

## Why Full Tool Object Isn't Sent

**Question**: Why don't we just send the full tool object directly?

**Answer**: We do send the full tool object, but:

1. **In Defender Mode**: Descriptions are stripped for security (prevent prompt injection)
2. **Formatting Matters**: Raw JSON is harder for small models to parse than structured markdown
3. **Reinforcement**: Multiple formats help small models understand constraints
4. **Relevance**: RAG filters to only relevant tools (reduces noise)

**What "Full Tool Object" means**:
- ✅ We send the complete `inputSchema` (all properties, types, required fields)
- ✅ We send tool names
- ❌ In defender mode, we strip `description` (security)
- ✅ We add extra warnings and restrictions (helpful context)

So we're not hiding information - we're **enhancing** it with better formatting and explicit restrictions.

