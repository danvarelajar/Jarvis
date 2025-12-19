import asyncio
import re
from typing import Dict, List, Optional, Callable, Any
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CreateMessageResult



# Since sse_client is a context manager, we need a way to manage multiple of them.
# A common pattern is to have a background task that manages the lifecycle.

from mcp.client.streamable_http import streamablehttp_client

import time
from datetime import datetime

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

class PersistentConnection:
    def __init__(
        self,
        server_key: str,
        display_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        transport: str = "sse",
        sampling_callback: Optional[Callable[[Any], Any]] = None,
    ):
        # server_key is the normalized internal key (typically lowercase).
        # display_name is what the user configured (preserve case), and is used for tool name prefixes.
        self.server_key = server_key
        self.display_name = display_name
        self.url = url
        self.headers = headers or {}
        self.transport = transport
        self.sampling_callback = sampling_callback
        self.session: Optional[ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        
        # Caching
        self.tools_cache = None
        self.tools_cache_timestamp = 0
        
        self.resources_cache = None
        self.resources_cache_timestamp = 0
        
        self.prompts_cache = None
        self.prompts_cache_timestamp = 0
        
        self.CACHE_TTL = 300  # 5 minutes

    async def start(self):
        # Exponential backoff parameters
        backoff_delay = 1
        max_backoff = 60
        
        while True:
            try:
                if self.transport == "http":
                    # HTTP Transport (Streamable HTTP)
                    async with streamablehttp_client(self.url, headers=self.headers) as (read, write, _):
                        async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                            self.session = session
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Connected via HTTP")
                            init_start = time.time()
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Request: initialize()")
                            await session.initialize()
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Response: initialize() -> success ({format_duration(init_start)})")
                            
                            # Prefetch tools to populate cache
                            asyncio.create_task(self.get_tools())
                            
                            # Reset backoff on successful connection
                            backoff_delay = 1
                            
                            while True:
                                await asyncio.sleep(1)
                else:
                    # SSE Transport
                    # Set timeout to None to prevent read timeouts on long-lived connections
                    
                    import logging
                    import sys
                    
                    async def log_response_body(response):
                        print(f"HOOK CALLED for {response.url}", flush=True)
                        if response.status_code >= 400:
                            body = await response.aread()
                            print(f"Response Body for {response.url}: {body}", flush=True)
                            logging.error(f"Response Body for {response.url}: {body}")

                    def custom_client_factory(headers, auth, timeout):
                        # Enforce infinite timeout for SSE connections to prevent read timeouts
                        # httpx defaults to 5s if not specified, and sometimes timeout=None isn't enough depending on context
                        return httpx.AsyncClient(
                            headers=headers, 
                            auth=auth, 
                            timeout=httpx.Timeout(None, connect=5.0),
                            event_hooks={'response': [log_response_body]}
                        )

                    async with sse_client(self.url, headers=self.headers, timeout=None, httpx_client_factory=custom_client_factory) as (read, write):
                        async with ClientSession(read, write, sampling_callback=self.sampling_callback) as session:
                            self.session = session
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Connected via SSE")
                            init_start = time.time()
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Request: initialize()")
                            await session.initialize()
                            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Response: initialize() -> success ({format_duration(init_start)})")
                            
                            # Prefetch tools to populate cache
                            asyncio.create_task(self.get_tools())
                            
                            # Reset backoff on successful connection
                            backoff_delay = 1
                            
                            while True:
                                await asyncio.sleep(1)
            except Exception as e:
                import traceback
                import sys
                
                error_msg = str(e)
                is_connection_error = False
                
                # Helper to check for connection errors in an exception
                def check_connection_error(exc):
                    msg = str(exc).lower()
                    name = type(exc).__name__
                    return "ConnectError" in name or "os error" in msg or "connection refused" in msg or "connect call failed" in msg or "session terminated" in msg

                # Recursively flatten exceptions
                def get_all_exceptions(exc):
                    if hasattr(exc, 'exceptions'):
                        excs = []
                        for error in exc.exceptions:
                            excs.extend(get_all_exceptions(error))
                        return excs
                    else:
                        return [exc]

                all_exceptions = get_all_exceptions(e)
                connections_errors = [exc for exc in all_exceptions if check_connection_error(exc)]
                
                if connections_errors:
                    is_connection_error = True
                    # Use the first connection error as the main message
                    error_msg = str(connections_errors[0])
                else:
                    # If multiple generic errors, join them
                    if len(all_exceptions) > 1:
                        error_msg = f"TaskGroup errors: {'; '.join([str(exc) for exc in all_exceptions])}"
                        traceback.print_exc() # Print full trace for generic errors

                if is_connection_error:
                     print(f"Connection error for {self.display_name}: {error_msg}")
                     
                     # Fallback logic: If HTTP fails, try SSE
                     if self.transport == "http":
                         print(f"Attempting fallback to SSE transport for {self.display_name}...")
                         self.transport = "sse"
                         backoff_delay = 1 # Reset backoff for immediate retry
                         continue # Retry immediately
                else:
                     print(f"Connection error for {self.display_name}: {error_msg}")
                     if not isinstance(e, asyncio.CancelledError):
                        traceback.print_exc()

                self.session = None
                # Clear cache on disconnection
                self.tools_cache = None
                self.resources_cache = None
                self.prompts_cache = None
                
                print(f"Reconnecting to {self.display_name} in {backoff_delay} seconds...")
                await asyncio.sleep(backoff_delay)
                
                # Increase backoff
                backoff_delay = min(backoff_delay * 2, max_backoff)

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            print(f"Stopped connection to {self.display_name}")

    async def get_tools(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.tools_cache and (current_time - self.tools_cache_timestamp < self.CACHE_TTL):
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Using cached tools ({len(self.tools_cache)} tools)")
            return self.tools_cache
            
        try:
            list_start = time.time()
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Request: list_tools()")
            result = await self.session.list_tools()
            self.tools_cache = []
            for t in result.tools:
                tool_dump = t.model_dump()
                # Prefix tool name with server name
                # Use display_name to preserve case expected by labs (e.g., Booking__search_flights)
                tool_dump['name'] = f"{self.display_name}__{t.name}"
                self.tools_cache.append(tool_dump)
                
            self.tools_cache_timestamp = current_time
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Response: list_tools() -> {len(self.tools_cache)} tools ({format_duration(list_start)})")
            return self.tools_cache
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Error listing tools: {e}")
            return []

    async def get_resources(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.resources_cache and (current_time - self.resources_cache_timestamp < self.CACHE_TTL):
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Using cached resources ({len(self.resources_cache)} resources)")
            return self.resources_cache
            
        try:
            list_start = time.time()
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Request: list_resources()")
            result = await self.session.list_resources()
            self.resources_cache = [r.model_dump() for r in result.resources]
            self.resources_cache_timestamp = current_time
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Response: list_resources() -> {len(self.resources_cache)} resources ({format_duration(list_start)})")
            return self.resources_cache
        except Exception as e:
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Error listing resources: {e}")
            return []

    async def get_prompts(self):
        if not self.session:
            return []
            
        current_time = time.time()
        if self.prompts_cache and (current_time - self.prompts_cache_timestamp < self.CACHE_TTL):
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Using cached prompts ({len(self.prompts_cache)} prompts)")
            return self.prompts_cache
            
        try:
            list_start = time.time()
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Request: list_prompts()")
            result = await self.session.list_prompts()
            self.prompts_cache = [p.model_dump() for p in result.prompts]
            self.prompts_cache_timestamp = current_time
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Response: list_prompts() -> {len(self.prompts_cache)} prompts ({format_duration(list_start)})")
            return self.prompts_cache
        except Exception as e:
            print(f"[{get_timestamp()}] [MCP] [{self.display_name}] Error listing prompts: {e}")
            return []

import json
import os

# Simple rule (per lab requirement):
# If these files exist under the repo-root `data/` directory, they must be loaded.
JARVIS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(JARVIS_ROOT, "data")
CONFIG_FILE = os.path.join(DATA_DIR, "mcp_config.json")
SECRETS_FILE = os.path.join(DATA_DIR, "secrets.json")
LLM_CONFIG_FILE = os.path.join(DATA_DIR, "llm_config.json")

class GlobalConnectionManager:
    def __init__(self):
        self.connections: Dict[str, PersistentConnection] = {}
        # Map normalized server key -> display name as provided by the user/config (preserve case)
        self.server_display_names: Dict[str, str] = {}
        self.sampling_callback: Optional[Callable[[Any], Any]] = None
        self.openai_api_key: Optional[str] = None
        self.llm_provider: str = "openai" # openai or ollama
        self.ollama_url: str = "http://10.3.0.7:11434"
        self.ollama_model_name: str = "qwen3:8b"  # Model name for Ollama (e.g., qwen3:8b, gemma3:8b)
        # Lab alignment: controls how aggressively the agent exposes tools and follows untrusted text.
        # - defender: least-privilege tool exposure + safer tool-output framing
        # - naive: intentionally permissive to demonstrate failures
        self.agent_mode: str = "defender"
        self.last_config_mtime = 0
        
    def set_sampling_callback(self, callback: Callable[[Any], Any]):
        self.sampling_callback = callback
        # If we already have connections, we might need to restart them to pick up the callback.
        # But usually this is called at startup before adding servers.

    async def load_config(self):
        print(f"[{get_timestamp()}] [Jarvis] Using DATA_DIR={DATA_DIR}")
        if os.path.exists(CONFIG_FILE):
            try:
                current_mtime = os.path.getmtime(CONFIG_FILE)
                
                with open(CONFIG_FILE, 'r') as f:
                    content = f.read()
                
                # Remove comments (simple approach: lines starting with // or #)
                lines = content.splitlines()
                clean_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("//") or stripped.startswith("#"):
                        continue
                    clean_lines.append(line)
                
                clean_content = "\n".join(clean_lines)
                config = json.loads(clean_content)

                # Debug: show configured MCP servers (redact header values)
                try:
                    servers_cfg = config.get("mcpServers", {}) or {}
                    for srv_name, details in servers_cfg.items():
                        hdrs = details.get("headers") or {}
                        hdr_keys = list(hdrs.keys()) if isinstance(hdrs, dict) else []
                        print(f"[{get_timestamp()}] [Jarvis] Config mcpServers[{srv_name}]: url={details.get('url')} transport={details.get('transport','sse')} headers={hdr_keys}")
                except Exception as e:
                    print(f"[{get_timestamp()}] [Jarvis] Failed to print server config summary: {e}")
                
                # Load Global Settings - MOVED TO LLM_CONFIG_FILE
                # self.llm_provider = config.get("llmProvider", "gemini")
                # self.ollama_url = config.get("ollamaUrl", "http://10.3.0.7:11434")

                # 1. Identify current active servers
                active_servers = set(self.connections.keys())
                
                # 2. Iterate through new config
                new_servers = set()
                for name, details in config.get("mcpServers", {}).items():
                    normalized = (name or "").lower()
                    new_servers.add(normalized)
                    
                    # Check if server is already connected and active
                    existing_conn = self.connections.get(normalized)
                    if existing_conn and existing_conn.session:
                        # Server is already connected and active - skip reconnection
                        # Only reconnect if URL, transport, or headers changed
                        url_changed = existing_conn.url != details["url"]
                        transport_changed = existing_conn.transport != details.get("transport", "sse")
                        new_headers = details.get("headers") or {}
                        headers_changed = existing_conn.headers != new_headers
                        
                        if url_changed or transport_changed or headers_changed:
                            print(f"[{get_timestamp()}] [Jarvis] Server {name} config changed (url={url_changed}, transport={transport_changed}, headers={headers_changed}), reconnecting...")
                            await self.add_server(
                                name, 
                                details["url"], 
                                details.get("headers"), 
                                transport=details.get("transport", "sse"),
                                save=False
                            )
                        else:
                            # Connection is active and config unchanged - skip reconnection
                            pass  # No action needed
                    else:
                        # Server not connected or connection lost - connect it
                    await self.add_server(
                        name, 
                        details["url"], 
                        details.get("headers"), 
                        transport=details.get("transport", "sse"),
                        save=False
                    )
                    
                # 3. Remove servers that are no longer in config
                to_remove = active_servers - new_servers
                for name in to_remove:
                    print(f"Removing server {name} as it was removed from config")
                    if name in self.connections:
                        await self.connections[name].stop()
                        del self.connections[name]
                        self.server_display_names.pop(name, None)
                        
                self.last_config_mtime = current_mtime
                print(f"Loaded config from {CONFIG_FILE}")
            except Exception as e:
                print(f"Failed to load config: {e}")
        else:
            print(f"[{get_timestamp()}] [Jarvis] MCP config not found at {CONFIG_FILE}")

        # Load LLM Config
        if os.path.exists(LLM_CONFIG_FILE):
            try:
                with open(LLM_CONFIG_FILE, 'r') as f:
                    llm_config = json.load(f)
                    self.llm_provider = llm_config.get("llmProvider", "openai")
                    self.ollama_url = llm_config.get("ollamaUrl", "http://10.3.0.7:11434")
                    self.ollama_model_name = llm_config.get("ollamaModelName", "qwen3:8b")
                    self.agent_mode = llm_config.get("agentMode", "defender")
                print(f"Loaded LLM config from {LLM_CONFIG_FILE}")
                print(f"[{get_timestamp()}] [DEBUG] Loaded ollama_model_name: {self.ollama_model_name}")
            except Exception as e:
                print(f"Failed to load LLM config: {e}")
        else:
            print(f"[{get_timestamp()}] [Jarvis] LLM config not found at {LLM_CONFIG_FILE}")


        # Load secrets
        if os.path.exists(SECRETS_FILE):
            try:
                with open(SECRETS_FILE, 'r') as f:
                    secrets = json.load(f)
                    # Prefer openaiApiKey, but keep backwards-compatibility with older "ApiKey"
                    self.openai_api_key = secrets.get("openaiApiKey") or secrets.get("ApiKey")
                print(f"Loaded secrets from {SECRETS_FILE}")
            except Exception as e:
                print(f"Failed to load secrets: {e}")
        else:
            print(f"[{get_timestamp()}] [Jarvis] Secrets not found at {SECRETS_FILE}")

    async def watch_config(self):
        print(f"Starting config watcher for {CONFIG_FILE}")
        while True:
            await asyncio.sleep(2) # Check every 2 seconds
            if os.path.exists(CONFIG_FILE):
                try:
                    mtime = os.path.getmtime(CONFIG_FILE)
                    if mtime > self.last_config_mtime:
                        print("Config file changed, reloading...")
                        await self.load_config()
                except Exception as e:
                    print(f"Error watching config: {e}")

    def save_config(self):
        # Save MCP Servers
        config = {"mcpServers": {}}
        for server_key, conn in self.connections.items():
            # Persist using the original display name (preserve case for labs)
            display_name = conn.display_name or self.server_display_names.get(server_key) or server_key
            config["mcpServers"][display_name] = {
                "url": conn.url,
                "headers": conn.headers,
                "transport": conn.transport
            }
        
        # Save Global Settings - MOVED TO LLM_CONFIG_FILE
        # config["llmProvider"] = self.llm_provider
        # config["ollamaUrl"] = self.ollama_url
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Saved config to {CONFIG_FILE}")
        except Exception as e:
            print(f"Failed to save config: {e}")

        # Save Secrets
        # Preserve existing secrets if the in-memory key is empty/None to avoid "reset on restart".
        existing_openai_key = None
        try:
            if os.path.exists(SECRETS_FILE):
                with open(SECRETS_FILE, "r") as f:
                    existing = json.load(f)
                    existing_openai_key = existing.get("openaiApiKey") or existing.get("ApiKey")
        except Exception as e:
            print(f"Failed to read existing secrets (will not preserve): {e}")

        key_to_write = (self.openai_api_key or "").strip() or (existing_openai_key or "").strip() or None
        secrets = {"openaiApiKey": key_to_write}
        try:
            with open(SECRETS_FILE, 'w') as f:
                json.dump(secrets, f, indent=2)
            print(f"Saved secrets to {SECRETS_FILE}")
        except Exception as e:
            print(f"Failed to save secrets: {e}")

        # Save LLM Config
        llm_config = {
            "llmProvider": self.llm_provider,
            "ollamaUrl": self.ollama_url,
            "ollamaModelName": self.ollama_model_name,
            "agentMode": self.agent_mode
        }
        try:
            with open(LLM_CONFIG_FILE, 'w') as f:
                json.dump(llm_config, f, indent=2)
            print(f"Saved LLM config to {LLM_CONFIG_FILE}")
        except Exception as e:
            print(f"Failed to save LLM config: {e}")

    async def add_server(self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None, transport: str = "sse", save: bool = True):
        # Normalize to lowercase to prevent duplicates
        display_name = server_name
        server_key = (server_name or "").lower()
        
        # Stop existing if any
        if server_key in self.connections:
            print(f"Stopping existing connection for {server_key}...")
            await self.connections[server_key].stop()

        self.server_display_names[server_key] = display_name
        connection = PersistentConnection(server_key, display_name, url, headers, transport, sampling_callback=self.sampling_callback)
        self.connections[server_key] = connection
        connection._task = asyncio.create_task(connection.start())
        if save:
            self.save_config()

    def get_session(self, server_name: str) -> Optional[ClientSession]:
        conn = self.connections.get((server_name or "").lower())
        return conn.session if conn else None
    
    def get_all_sessions(self) -> Dict[str, ClientSession]:
        return {name: conn.session for name, conn in self.connections.items() if conn.session}

    async def list_tools(self, server_name: str = None) -> List[dict]:
        tools = []
        if server_name:
            server_key = (server_name or "").lower() # Normalize
            conn = self.connections.get(server_key)
            if conn:
                tools.extend(await conn.get_tools())
        else:
            for name, conn in self.connections.items():
                tools.extend(await conn.get_tools())
        return tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        server_key = (server_name or "").lower() # Normalize
        conn = self.connections.get(server_key)
        if not conn or not conn.session:
            raise ValueError(f"Server {server_name} not found or not connected")
        
        # Log MCP request
        import json
        args_str = json.dumps(arguments, separators=(',', ':')) if arguments else "{}"
        # Truncate long arguments for logging
        if len(args_str) > 200:
            args_str = args_str[:200] + "... (truncated)"
        call_start = time.time()
        print(f"[{get_timestamp()}] [MCP] [{conn.display_name}] Request: call_tool(tool='{tool_name}', arguments={args_str})")
        
        # Add timeout to prevent indefinite hanging (60 seconds should be enough for most tool calls)
        # If the tool call takes longer, it will raise asyncio.TimeoutError
        tool_call_timeout = 60.0  # 60 seconds
        
        try:
            print(f"[{get_timestamp()}] [MCP] [{conn.display_name}] Waiting for tool response (timeout: {tool_call_timeout}s)...")
            result = await asyncio.wait_for(
                conn.session.call_tool(tool_name, arguments),
                timeout=tool_call_timeout
            )
            
            # Log MCP response
            result_summary = "success"
            if hasattr(result, 'content'):
                content_types = [item.type for item in result.content] if result.content else []
                result_summary = f"success (content: {', '.join(content_types)})"
            elif hasattr(result, 'model_dump'):
                result_summary = "success (structured data)"
            else:
                result_summary = f"success (type: {type(result).__name__})"
            
            print(f"[{get_timestamp()}] [MCP] [{conn.display_name}] Response: call_tool('{tool_name}') -> {result_summary} ({format_duration(call_start)})")
            return result
        except asyncio.TimeoutError:
            elapsed = time.time() - call_start
            error_msg = f"Tool call timed out after {elapsed:.1f}s (timeout: {tool_call_timeout}s). The MCP server '{conn.display_name}' did not respond in time."
            print(f"[{get_timestamp()}] [MCP] [{conn.display_name}] Error: call_tool('{tool_name}') -> TimeoutError: {error_msg} ({format_duration(call_start)})")
            raise TimeoutError(error_msg)
        except Exception as e:
            print(f"[{get_timestamp()}] [MCP] [{conn.display_name}] Error: call_tool('{tool_name}') -> {type(e).__name__}: {str(e)} ({format_duration(call_start)})")
            raise

# Smart Routing Logic
def parse_server_route(message: str) -> Optional[str]:
    """
    Scans for @{server_name} or @server_name in the message.
    Returns the server_name if found, else None.
    """
    # Matches @{name} or @name, preceded by start of string or whitespace
    match = re.search(r'(?:^|\s)@(?:\{([a-zA-Z0-9_-]+)\}|([a-zA-Z0-9_-]+))', message)
    if match:
        name = match.group(1) or match.group(2)
        return name.lower() if name else None
    return None

def parse_all_server_routes(message: str) -> list[str]:
    """
    Scans for ALL @{server_name} or @server_name mentions in the message.
    Returns a list of all server names found (lowercased, deduplicated).
    """
    # Matches all @{name} or @name occurrences
    matches = re.findall(r'(?:^|\s)@(?:\{([a-zA-Z0-9_-]+)\}|([a-zA-Z0-9_-]+))', message)
    servers = []
    for match in matches:
        name = match[0] or match[1]
        if name:
            servers.append(name.lower())
    # Return deduplicated list while preserving order
    seen = set()
    result = []
    for server in servers:
        if server not in seen:
            seen.add(server)
            result.append(server)
    return result

# Singleton instance
connection_manager = GlobalConnectionManager()
