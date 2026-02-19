# Worker Configuration Analysis

## Current Configuration

The application is configured to run with **4 workers** in `Dockerfile`:
```dockerfile
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]
```

**Rationale (from comment)**: "Run the application with multiple workers to handle concurrent requests. This prevents new requests from being queued behind long-running Ollama inference."

## Problems with Multiple Workers

### 1. **State Management Issues**
- Each worker has its own `GlobalConnectionManager` instance (module-level singleton)
- Each worker maintains separate MCP connections
- Each worker runs its own config watcher
- **Result**: Duplicate connections, multiple config reloads, inconsistent state

### 2. **Resource Waste**
- 4x the MCP connections (each worker connects to all MCP servers)
- 4x the config watchers (each worker watches the config file)
- 4x the memory usage for connection state
- **Result**: Unnecessary resource consumption

### 3. **Race Conditions**
- Multiple workers detecting config changes simultaneously
- Multiple workers trying to reload config at the same time
- **Result**: Cascading reloads, duplicate connections

### 4. **Debugging Complexity**
- Logs from 4 different processes interleaved
- Hard to trace which worker handled which request
- **Result**: Difficult troubleshooting

## Why Multiple Workers Are NOT Needed

### 1. **All Operations Are Async (I/O Bound)**
- ✅ LLM API calls: `async with httpx.AsyncClient()`
- ✅ MCP tool calls: `await session.call_tool()`
- ✅ Database/file operations: Async I/O
- ✅ FastAPI endpoints: All `async def`

**Key Point**: Async operations don't block the event loop. While one request waits for Ollama, other requests can be processed concurrently.

### 2. **FastAPI Handles Concurrency Well**
- FastAPI's async support allows handling many concurrent requests with a single worker
- The event loop can manage hundreds of concurrent async operations
- **Example**: While request A waits for Ollama (async), request B can be processed immediately

### 3. **No CPU-Bound Tasks**
- ❌ No heavy computation
- ❌ No image processing
- ❌ No synchronous blocking operations
- **Result**: No benefit from multiple processes

### 4. **Stateful Application**
- MCP connections are long-lived and stateful
- Config watcher needs to track file changes
- Connection state should be shared (not duplicated)
- **Result**: Single worker is architecturally better

## When Multiple Workers ARE Appropriate

Multiple workers are beneficial for:
- **CPU-bound tasks**: Heavy computation, image processing, data transformation
- **Blocking synchronous code**: Legacy code that can't be made async
- **Very high throughput**: Thousands of requests per second (not applicable here)
- **Stateless applications**: APIs without shared state (not our case)

## Recommendation: Single Worker

### Benefits
1. ✅ **Simpler architecture**: One process, one set of connections, one config watcher
2. ✅ **Better state management**: Shared state across all requests
3. ✅ **Easier debugging**: Single process logs
4. ✅ **Resource efficiency**: No duplicate connections or watchers
5. ✅ **No race conditions**: Single config watcher, single reload process

### Performance Impact
- **Concurrent requests**: ✅ Still handled efficiently (async)
- **Long-running operations**: ✅ Don't block other requests (async)
- **Throughput**: ✅ Sufficient for typical usage (chat interface)

### Configuration Change
```dockerfile
# Before
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "4"]

# After
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000"]
```

## Testing Recommendations

After switching to a single worker:
1. ✅ Test concurrent chat requests (should work fine)
2. ✅ Test config changes (should reload once)
3. ✅ Test MCP connections (should connect once per server)
4. ✅ Monitor resource usage (should be lower)

## Conclusion

**Single worker is the correct choice** for this application because:
- All operations are async (I/O bound)
- Stateful connections need to be shared
- No CPU-bound tasks require multiple processes
- Simpler architecture with better state management

The original rationale ("prevent requests from being queued behind long-running Ollama inference") is incorrect - async operations don't block the event loop, so concurrent requests are handled efficiently by a single worker.
