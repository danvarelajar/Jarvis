import React, { useState, useEffect, useRef } from 'react';
import { Send, Server, Terminal, Bot, User, Loader2, ChevronDown, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [servers, setServers] = useState([]);
    const [serverConfigJson, setServerConfigJson] = useState('');
    const [isConfigExpanded, setIsConfigExpanded] = useState(false);
    const [openaiApiKey, setOpenaiApiKey] = useState('');
    const [llmProvider, setLlmProvider] = useState('openai');
    const [ollamaUrl, setOllamaUrl] = useState('http://10.3.0.7:11434');
    const [ollamaModelName, setOllamaModelName] = useState('qwen3:8b');
    const [ollamaModels, setOllamaModels] = useState([]);
    const [isLoadingModels, setIsLoadingModels] = useState(false);
    const [agentMode, setAgentMode] = useState('defender');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Save API Key to backend (debounced or on blur would be better, but for now simple effect)
    // Actually, let's just save it when the user clicks a save button or when they connect servers?
    // The user wants it to be persistent.
    // Let's fetch it on load.

    // Load server config from backend on mount
    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const response = await fetch('/api/config');
                if (response.ok) {
                    const config = await response.json();
                    if (config.openaiApiKey) setOpenaiApiKey(config.openaiApiKey);
                    if (config.llmProvider) setLlmProvider(config.llmProvider);
                    if (config.ollamaUrl) setOllamaUrl(config.ollamaUrl);
                    if (config.ollamaModelName) setOllamaModelName(config.ollamaModelName);
                    if (config.agentMode) setAgentMode(config.agentMode);
                    if (config.mcpServers && Object.keys(config.mcpServers).length > 0) {
                        setServerConfigJson(JSON.stringify(config, null, 2));
                        // Also update the UI list of connected servers
                        setServers(Object.keys(config.mcpServers));
                    }
                }
            } catch (error) {
                console.error("Failed to fetch config:", error);
            }
        };
        fetchConfig();
    }, []);

    // Fetch Ollama models when Ollama is selected or URL changes
    // Use debouncing to avoid fetching on every keystroke
    useEffect(() => {
        // Only fetch if provider is ollama and URL is not empty
        if (llmProvider !== 'ollama' || !ollamaUrl || ollamaUrl.trim() === '') {
            return;
        }

        // Debounce: wait 500ms after last change before fetching
        const timeoutId = setTimeout(() => {
            const fetchModels = async () => {
                setIsLoadingModels(true);
                try {
                    const response = await fetch(`/api/ollama/models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.models) {
                            setOllamaModels(data.models);
                            // If current model is not in the list, keep it but add it
                            const modelExists = data.models.some(m => m.name === ollamaModelName);
                            if (!modelExists && ollamaModelName) {
                                setOllamaModels(prev => [...prev, { name: ollamaModelName, full_name: ollamaModelName }]);
                            }
                        }
                    }
                } catch (error) {
                    console.error("Failed to fetch Ollama models:", error);
                } finally {
                    setIsLoadingModels(false);
                }
            };
            fetchModels();
        }, 500); // Wait 500ms after user stops typing

        // Cleanup: cancel the timeout if URL changes again before 500ms
        return () => clearTimeout(timeoutId);
    }, [llmProvider, ollamaUrl, ollamaModelName]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-openai-api-key': openaiApiKey
                },
                body: JSON.stringify({ messages: [...messages, userMsg] }),
            });

            const data = await response.json();
            setMessages(prev => [...prev, data]);
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error.message}` }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const connectServers = async () => {
        try {
            const config = JSON.parse(serverConfigJson);
            if (!config.mcpServers) {
                throw new Error("Invalid config: missing 'mcpServers' key");
            }

            const promises = Object.entries(config.mcpServers).map(async ([name, details]) => {
                await fetch('/api/connect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        server_name: name,
                        url: details.url,
                        headers: details.headers,
                        transport: details.transport || 'sse'
                    }),
                });
                return name;
            });

            const connectedServerNames = await Promise.all(promises);
            setServers(connectedServerNames);
            alert(`Configuration saved and ${connectedServerNames.length} servers connected`);
        } catch (error) {
            alert(`Failed to connect: ${error.message}`);
        }
    };

    return (
        <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
            {/* Sidebar */}
            <div className="w-64 bg-gray-800 border-r border-gray-700 p-4 flex flex-col">
                <div className="flex items-center gap-2 mb-6 text-xl font-bold text-blue-400">
                    <Bot size={24} />
                    <span>Jarvis MCP</span>
                </div>

                <div className="mb-4">
                    <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2">Connected Servers</h3>
                    <div className="space-y-2">
                        {servers.length === 0 ? (
                            <p className="text-sm text-gray-500 italic">No servers connected</p>
                        ) : (
                            servers.map(server => (
                                <div key={server} className="flex items-center gap-2 text-sm text-green-400">
                                    <Server size={14} />
                                    <span>{server}</span>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                <div className="mt-auto border-t border-gray-700 pt-4">
                    <div className="mb-4">
                        <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2">Agent Mode (Lab)</h3>
                        <select
                            value={agentMode}
                            onChange={async (e) => {
                                const newMode = e.target.value;
                                setAgentMode(newMode);
                                try {
                                    const keyToSend = openaiApiKey.trim() ? openaiApiKey : undefined;
                                    await fetch('/api/config', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            mcpServers: {},
                                            ...(keyToSend !== undefined ? { openaiApiKey: keyToSend } : {}),
                                            llmProvider: llmProvider,
                                            ollamaUrl: ollamaUrl,
                                            ollamaModelName: ollamaModelName,
                                            agentMode: newMode
                                        }),
                                    });
                                } catch (error) {
                                    console.error("Failed to save agent mode:", error);
                                }
                            }}
                            className="w-full bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none"
                        >
                            <option value="defender">Defender (hardened)</option>
                            <option value="naive">Naive (intentionally permissive)</option>
                        </select>
                        <p className="text-[11px] text-gray-500 mt-2 leading-snug">
                            Defender uses least privilege + safer tool handling. Naive loads more tools to demonstrate attacks.
                        </p>
                    </div>

                    <div className="mb-4">
                        <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2">LLM Provider</h3>
                        <select
                            value={llmProvider}
                            onChange={async (e) => {
                                const newProvider = e.target.value;
                                setLlmProvider(newProvider);
                                try {
                                    // Avoid overwriting a saved key with an empty value.
                                    const keyToSend = openaiApiKey.trim() ? openaiApiKey : undefined;
                                    await fetch('/api/config', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            mcpServers: {},
                                            ...(keyToSend !== undefined ? { openaiApiKey: keyToSend } : {}),
                                            llmProvider: newProvider,
                                            ollamaUrl: ollamaUrl,
                                            ollamaModelName: ollamaModelName,
                                            agentMode: agentMode
                                        }),
                                    });
                                } catch (error) {
                                    console.error("Failed to save provider:", error);
                                }
                            }}
                            className="w-full bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none mb-2"
                        >
                            <option value="openai">ChatGPT 4o-mini (OpenAI)</option>
                            <option value="ollama">Ollama (Local)</option>
                        </select>

                        {llmProvider === 'ollama' && (
                            <>
                                <input
                                    type="text"
                                    value={ollamaUrl}
                                    onChange={(e) => setOllamaUrl(e.target.value)}
                                    onBlur={async () => {
                                        try {
                                            const keyToSend = openaiApiKey.trim() ? openaiApiKey : undefined;
                                            await fetch('/api/config', {
                                                method: 'POST',
                                                headers: { 'Content-Type': 'application/json' },
                                                body: JSON.stringify({
                                                    mcpServers: {},
                                                    ...(keyToSend !== undefined ? { openaiApiKey: keyToSend } : {}),
                                                    llmProvider: llmProvider,
                                                    ollamaUrl: ollamaUrl,
                                                    ollamaModelName: ollamaModelName,
                                                    agentMode: agentMode
                                                }),
                                            });
                                        } catch (error) {
                                            console.error("Failed to save Ollama URL:", error);
                                        }
                                    }}
                                    placeholder="http://10.3.0.7:11434"
                                    className="w-full bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none mb-2"
                                />
                                <div className="flex items-center gap-2">
                                    <select
                                        value={ollamaModelName}
                                        onChange={async (e) => {
                                            const newModel = e.target.value;
                                            setOllamaModelName(newModel);
                                            try {
                                                const keyToSend = openaiApiKey.trim() ? openaiApiKey : undefined;
                                                await fetch('/api/config', {
                                                    method: 'POST',
                                                    headers: { 'Content-Type': 'application/json' },
                                                    body: JSON.stringify({
                                                        mcpServers: {},
                                                        ...(keyToSend !== undefined ? { openaiApiKey: keyToSend } : {}),
                                                        llmProvider: llmProvider,
                                                        ollamaUrl: ollamaUrl,
                                                        ollamaModelName: newModel,
                                                        agentMode: agentMode
                                                    }),
                                                });
                                            } catch (error) {
                                                console.error("Failed to save Ollama model:", error);
                                            }
                                        }}
                                        disabled={isLoadingModels}
                                        className="flex-1 bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none"
                                    >
                                        {isLoadingModels ? (
                                            <option>Loading models...</option>
                                        ) : ollamaModels.length === 0 ? (
                                            <option value={ollamaModelName}>{ollamaModelName || 'No models found'}</option>
                                        ) : (
                                            ollamaModels.map((model) => (
                                                <option key={model.name} value={model.name}>
                                                    {model.name}
                                                </option>
                                            ))
                                        )}
                                    </select>
                                    <button
                                        onClick={async () => {
                                            setIsLoadingModels(true);
                                            try {
                                                const response = await fetch(`/api/ollama/models?ollama_url=${encodeURIComponent(ollamaUrl)}`);
                                                if (response.ok) {
                                                    const data = await response.json();
                                                    if (data.models) {
                                                        setOllamaModels(data.models);
                                                    }
                                                }
                                            } catch (error) {
                                                console.error("Failed to refresh models:", error);
                                            } finally {
                                                setIsLoadingModels(false);
                                            }
                                        }}
                                        disabled={isLoadingModels}
                                        className="px-3 py-2 bg-gray-800 hover:bg-gray-700 text-white text-xs rounded border border-gray-700 focus:outline-none disabled:opacity-50"
                                        title="Refresh models"
                                    >
                                        🔄
                                    </button>
                                </div>
                            </>
                        )}
                    </div>

                    {llmProvider === 'openai' && (
                        <div className="mb-4">
                            <h3 className="text-xs uppercase text-gray-500 font-semibold mb-2">OpenAI API Key</h3>
                            <input
                                type="password"
                                value={openaiApiKey}
                                onChange={(e) => setOpenaiApiKey(e.target.value)}
                                onBlur={async () => {
                                    try {
                                        const keyToSend = openaiApiKey.trim() ? openaiApiKey : undefined;
                                        await fetch('/api/config', {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({
                                                mcpServers: {},
                                                ...(keyToSend !== undefined ? { openaiApiKey: keyToSend } : {}),
                                                llmProvider: llmProvider,
                                                ollamaUrl: ollamaUrl,
                                                ollamaModelName: ollamaModelName,
                                                agentMode: agentMode
                                            }),
                                        });
                                    } catch (error) {
                                        console.error("Failed to save OpenAI key:", error);
                                    }
                                }}
                                placeholder="sk-..."
                                className="w-full bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none"
                            />
                        </div>
                    )}

                    <button
                        onClick={() => setIsConfigExpanded(!isConfigExpanded)}
                        className="flex items-center gap-2 w-full text-xs uppercase text-gray-500 font-semibold mb-2 hover:text-gray-300 transition-colors"
                    >
                        {isConfigExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                        Configure Servers (JSON)
                    </button>

                    {isConfigExpanded && (
                        <div className="space-y-2">
                            <textarea
                                placeholder='{ "mcpServers": { ... } }'
                                value={serverConfigJson}
                                onChange={e => setServerConfigJson(e.target.value)}
                                className="w-full bg-gray-900 text-white text-xs rounded p-2 border border-gray-700 focus:border-blue-500 outline-none h-32 font-mono"
                            />
                            <button
                                onClick={connectServers}
                                disabled={!serverConfigJson.trim()}
                                className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded text-xs text-white flex items-center justify-center gap-2"
                            >
                                <Terminal size={14} />
                                Connect / Refresh All
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-gray-500">
                            <Bot size={48} className="mb-4 opacity-50" />
                            <p className="text-lg">How can I help you today?</p>
                            <p className="text-sm">Try asking me to use a tool or route to a specific server with @server_name</p>
                        </div>
                    )}

                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] rounded-lg p-4 ${msg.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-700 text-gray-100 border border-gray-600'
                                }`}>
                                <div className="flex items-center gap-2 mb-1 opacity-75 text-xs">
                                    {msg.role === 'user' ? <User size={12} /> : <Bot size={12} />}
                                    <span className="uppercase font-bold">{msg.role}</span>
                                </div>
                                <div className="prose prose-invert max-w-none text-sm break-words">
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={{
                                            code({ node, inline, className, children, ...props }) {
                                                const match = /language-(\w+)/.exec(className || '')
                                                return !inline && match ? (
                                                    <SyntaxHighlighter
                                                        {...props}
                                                        style={vscDarkPlus}
                                                        language={match[1]}
                                                        PreTag="div"
                                                        wrapLongLines={true}
                                                        className="rounded-md !bg-gray-900 !p-3 !my-2 !whitespace-pre-wrap !break-words max-w-full overflow-x-auto"
                                                    >
                                                        {String(children).replace(/\n$/, '')}
                                                    </SyntaxHighlighter>
                                                ) : (
                                                    <code {...props} className={`${className} bg-black/20 rounded px-1 py-0.5`}>
                                                        {children}
                                                    </code>
                                                )
                                            }
                                        }}
                                    >
                                        {msg.content}
                                    </ReactMarkdown>
                                </div>
                                {msg.tool_result && (
                                    <div className="mt-2 p-2 bg-black/30 rounded text-xs font-mono text-green-300 border-l-2 border-green-500 overflow-x-auto">
                                        <div className="font-bold mb-1">Tool Result:</div>
                                        <pre className="whitespace-pre-wrap break-words">{(() => {
                                            try {
                                                const parsed = JSON.parse(msg.tool_result);
                                                return JSON.stringify(parsed, null, 2);
                                            } catch (e) {
                                                return msg.tool_result;
                                            }
                                        })()}</pre>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 bg-gray-800 border-t border-gray-700">
                    <div className="max-w-4xl mx-auto relative">
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Type your message... (Use @server_name to route)"
                            className="w-full bg-gray-900 text-white rounded-lg pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none h-[52px]"
                        />
                        <button
                            onClick={sendMessage}
                            disabled={isLoading || !input.trim()}
                            className="absolute right-2 top-2 p-2 text-blue-400 hover:text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default App
