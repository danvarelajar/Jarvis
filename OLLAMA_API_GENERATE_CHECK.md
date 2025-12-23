# Ollama `/api/generate` Migration Checklist

## Current Implementation (`/api/chat`)

### What We Have:
1. ✅ **Structured Prompt Builder** (`build_structured_prompt_gemma()`)
   - Already builds prompts with Gemma control tokens (`<start_of_turn>developer`, `<end_of_turn>`)
   - Returns a single prompt string (perfect for `/api/generate`!)

2. ✅ **Message Format Conversion**
   - Currently converts messages array to Ollama format
   - Handles role mapping (model → assistant)

3. ✅ **Response Parsing**
   - Currently: `result.get("message", {}).get("content", "")`
   - For `/api/generate`: `result.get("response", "")`

### Current Endpoint Usage:
- **Endpoint**: `/api/chat`
- **Payload Format**:
  ```json
  {
    "model": "gemma3-mcp:1b",
    "messages": [
      {"role": "system", "content": "<start_of_turn>developer\n..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "stream": false,
    "keep_alive": "10m",
    "options": {"temperature": 0}
  }
  ```

## What's Needed for `/api/generate`

### Required Changes:

1. **Convert Messages to Single Prompt String**
   - System prompt: Already has `<start_of_turn>developer` and `<end_of_turn>`
   - User messages: Need `<start_of_turn>user\n{content}<end_of_turn>`
   - Assistant messages: Need `<start_of_turn>model\n{content}<end_of_turn>`
   - Final turn: Add `<start_of_turn>model\n` at the end (for LLM to respond)

2. **Change Endpoint**
   - From: `/api/chat`
   - To: `/api/generate`

3. **Change Payload Format**
   ```json
   {
     "model": "gemma3-mcp:1b",
     "prompt": "<start_of_turn>developer\n...<end_of_turn>\n<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n",
     "stream": false,
     "keep_alive": "10m",
     "options": {"temperature": 0}
   }
   ```

4. **Change Response Parsing**
   - From: `result.get("message", {}).get("content", "")`
   - To: `result.get("response", "")`

### Implementation Plan:

```python
def convert_messages_to_prompt(messages: list, system_prompt: str) -> str:
    """
    Convert messages array to single prompt string with Gemma tokens.
    """
    prompt_parts = []
    
    # System prompt (already has Gemma tokens)
    prompt_parts.append(system_prompt)
    
    # Convert messages
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            # System already handled above
            continue
        elif role == "user":
            prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant" or role == "model":
            prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    
    # Add final turn marker for LLM to respond
    prompt_parts.append("<start_of_turn>model\n")
    
    return "\n".join(prompt_parts)
```

### Benefits of `/api/generate`:

1. **More Direct**: Single prompt string matches our structured prompt approach
2. **Better Control**: We control exact token placement
3. **Simpler**: No need for Ollama to convert messages to prompt
4. **Token Efficiency**: We can optimize token usage directly

### Potential Issues:

1. **Multi-turn Conversations**: Need to ensure conversation history is properly formatted
2. **Tool Results**: Need to format tool results as user messages with proper tokens
3. **Response Format**: Ensure response parsing handles both formats during migration

### Testing Checklist:

- [ ] Single-turn conversation (user → assistant)
- [ ] Multi-turn conversation (user → assistant → user → assistant)
- [ ] Tool calls (user → tool call → tool result → summary)
- [ ] System prompt with Gemma tokens
- [ ] Response parsing from `response` field
- [ ] Error handling for both endpoints (fallback)

## Recommendation:

**✅ We have everything needed!** Our structured prompt builder already creates the right format. We just need to:
1. Add message-to-prompt conversion function
2. Update endpoint and payload format
3. Update response parsing
4. Test thoroughly

The migration should be straightforward since we're already building structured prompts with Gemma tokens.

