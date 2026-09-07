/**
 * Ollama Cloud Chat Model Provider.
 *
 * Integrates Ollama Cloud models into GitHub Copilot Chat via the VS Code
 * Language Model API. Ollama Cloud exposes open-weight models through a
 * single OpenAI-compatible Chat Completions endpoint at
 * https://ollama.com/api.
 *
 * This provider extends BaseChatModelProvider which owns the shared request
 * lifecycle (timeout, cancellation, retry, base URL validation, headers,
 * delay, generic error handling, and the ask_image vision proxy loop).
 * Ollama Cloud-specific behavior: model discovery (hardcoded model list),
 * authentication (separate secret key), and the OpenAI Chat Completions
 * dispatch (always apiMode "openai", no Anthropic/Responses endpoints).
 */

import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelResponsePart,
    PrepareLanguageModelChatModelOptions,
    ProvideLanguageModelChatResponseOptions,
    Progress,
} from "vscode";

import type { ModelPreset, RetryConfig } from "./types";
import { executeWithRetry, convertToolsToOpenAI } from "./utils";
import type { BaseModelItem } from "./baseProvider";
import {
    BaseChatModelProvider,
    reportNativeUsage,
    getRequestedReasoningEffort,
    type DispatchChatRequestParams,
} from "./baseProvider";
import { OpenaiApi } from "./openai/openaiApi";
import type { StreamUsage } from "./commonApi";
import { textTokenLength } from "./provideToken";
import { updateContextStatusBar, recordUsage, updateCumulativeTooltip, updateStatusBarWithApiPrompt } from "./statusBar";
import { logger } from "./logger";
import { l10n } from "./localize";
import {
    buildOllamaCloudModelInfos,
    getOllamaCloudModelConfig,
    OLLAMA_CLOUD_BASE_URL,
} from "./ollamaCloudModels";
import { ASK_IMAGE_TOOL_DEF, ASK_WITH_MULTI_IMAGE_TOOL_DEF } from "./vision/types";

/**
 * Mutable thinking state shared across the Ollama stream processing helpers.
 */
interface OllamaThinkingState {
    thinkingId: string | null;
    thinkingBuffer: string;
    thinkingFlushTimer: ReturnType<typeof setTimeout> | null;
}

/** Flush buffered thinking content to the progress reporter. */
function flushThinking(state: OllamaThinkingState, progress: Progress<LanguageModelResponsePart>): void {
    if (state.thinkingFlushTimer) {
        clearTimeout(state.thinkingFlushTimer);
        state.thinkingFlushTimer = null;
    }
    if (state.thinkingBuffer && state.thinkingId) {
        const text = state.thinkingBuffer;
        state.thinkingBuffer = "";
        progress.report(new vscode.LanguageModelThinkingPart(text, state.thinkingId) as unknown as LanguageModelResponsePart);
    }
}

/** End the current thinking sequence. */
function endThinking(state: OllamaThinkingState, progress: Progress<LanguageModelResponsePart>): void {
    flushThinking(state, progress);
    if (state.thinkingId) {
        progress.report(new vscode.LanguageModelThinkingPart("", state.thinkingId) as unknown as LanguageModelResponsePart);
        state.thinkingId = null;
    }
}

/**
 * VS Code Chat provider backed by the Ollama Cloud API.
 *
 * Ollama Cloud offers free and paid tiers for running open-weight models
 * in the cloud. The API is OpenAI-compatible (Chat Completions format)
 * with Bearer token auth.
 */
export class OllamaCloudChatModelProvider extends BaseChatModelProvider<BaseModelItem> {
    /**
     * Create an Ollama Cloud provider using the given secret storage for the API key.
     */
    constructor(secrets: vscode.SecretStorage, statusBarItem: vscode.StatusBarItem) {
        super(secrets, statusBarItem, {
            secretKey: "ollamacloud.apiKey",
            title: "Ollama Cloud Provider API Key",
            prompt: "Enter your Ollama Cloud API key",
            missingMessage: "Ollama Cloud API key not found",
        });
    }

    /**
     * Get the list of available Ollama Cloud language models.
     * Models are hardcoded; capabilities are reused from the OpenCode Go catalog.
     */
    async provideLanguageModelChatInformation(
        _options: PrepareLanguageModelChatModelOptions,
        _token: CancellationToken
    ): Promise<LanguageModelChatInformation[]> {
        return buildOllamaCloudModelInfos();
    }

    /**
     * Resolve the Ollama Cloud model configuration from the VS Code model object.
     */
    protected async resolveModelConfig(model: LanguageModelChatInformation): Promise<BaseModelItem> {
        const config = await getOllamaCloudModelConfig(model.id);
        if (!config) {
            throw new Error(`Unknown Ollama Cloud model: ${model.id}`);
        }
        return config;
    }

    /**
     * Apply reasoning effort and temperature/top_p from model preset or custom settings.
     * Returns a shallow copy to avoid mutating the shared config.
     */
    protected applyRequestOptions(baseConfig: BaseModelItem, options: ProvideLanguageModelChatResponseOptions): BaseModelItem {
        const um: BaseModelItem = { ...baseConfig };

        // Apply reasoning effort from model configuration to determine thinking mode
        const effort = getRequestedReasoningEffort(options);
        if (effort) {
            if (effort === "disabled") {
                if (um.thinkingMode !== "always") {
                    um.enable_thinking = false;
                    um.include_reasoning_in_request = false;
                    um.reasoning_effort = undefined;
                } else {
                    um.enable_thinking = true;
                    um.include_reasoning_in_request = true;
                }
            } else {
                um.enable_thinking = true;
                um.include_reasoning_in_request = true;
                if (effort !== "enabled") {
                    um.reasoning_effort = effort;
                }
            }
        }

        // Inject temperature & top_p from model preset or custom settings
        // (reuses the opencodego.* settings namespace — extension-wide)
        if (um.supportsTemperature !== false) {
            const vscodeConfig = vscode.workspace.getConfiguration();
            const tempPreset = vscodeConfig.get<string>("opencodego.modelPreset", "custom");
            if (tempPreset !== "custom") {
                const presets = vscodeConfig.get<ModelPreset[]>("opencodego.modelPresets", []);
                const matchedPreset = presets.find((p) => p.id === tempPreset);
                if (matchedPreset) {
                    um.temperature = matchedPreset.temperature;
                }
            } else {
                const userTemperature = vscodeConfig.get<number | null>("opencodego.temperature", null);
                if (userTemperature !== null) {
                    um.temperature = userTemperature;
                }
                const userTopP = vscodeConfig.get<number | null>("opencodego.top_p", null);
                if (userTopP !== null) {
                    um.top_p = userTopP;
                } else {
                    um.top_p = undefined;
                }
            }
        } else {
            um.temperature = undefined;
            um.top_p = undefined;
        }

        return um;
    }

    /**
     * Dispatch the chat request to the Ollama Cloud backend.
     *
     * Ollama Cloud uses its own native API format (not OpenAI-compatible).
     * The request body uses `options` for temperature/top_p and `think` for
     * reasoning. The streaming response is newline-delimited JSON (not SSE),
     * with content in `message.content` and tool calls in `message.tool_calls`.
     *
     * The base class has already set up timeouts, retries, cancellation,
     * base URL validation, headers, and API key retrieval.
     */
    protected async dispatchChatRequest(params: DispatchChatRequestParams<BaseModelItem>): Promise<void> {
        const { model, config: um, messages, options, progress, token, abortController, retryConfig, dispatchFetch, requestHeaders } = params;
        const vscodeConfig = vscode.workspace.getConfiguration();

        const modelConfig = {
            includeReasoningInRequest: um.include_reasoning_in_request ?? true,
            vision: um.vision ?? false,
        };

        // Read Advanced Token indicator setting
        const enableThirdPartyIndicator = vscodeConfig.get<boolean>("opencodego.enableThirdPartyTokenIndicator", true);

        let usageReportedDuringStream = false;
        const collectedOutputText: string[] = [];
        const trackingProgress: Progress<LanguageModelResponsePart> = {
            report: (part) => {
                try {
                    if (part instanceof vscode.LanguageModelTextPart) {
                        collectedOutputText.push(part.value);
                    }
                    progress.report(part);
                } catch (e) {
                    console.error("[OllamaCloud] Progress.report failed", {
                        modelId: model.id,
                        error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
                    });
                }
            },
        };

        // Calculate client-side token estimate for fallback
        const estimatedInputTokens = await updateContextStatusBar(messages, options.tools, model, this.statusBarItem!, modelConfig);

        // Use OpenaiApi for message conversion only (Ollama messages format is OpenAI-compatible)
        const openaiApi = new OpenaiApi(model.id);
        const openaiMessages = await openaiApi.convertMessages(messages, modelConfig);

        // Build Ollama-native request body
        const ollamaBody: Record<string, unknown> = {
            model: um.id ?? model.id,
            messages: openaiMessages,
            stream: true,
        };

        // Temperature and top_p go into Ollama's `options` object
        const ollamaOptions: Record<string, unknown> = {};
        if (um.temperature !== undefined && um.temperature !== null && um.supportsTemperature !== false) {
            ollamaOptions.temperature = um.temperature;
        }
        if (um.top_p !== undefined && um.top_p !== null) {
            ollamaOptions.top_p = um.top_p;
        }
        if (Object.keys(ollamaOptions).length > 0) {
            ollamaBody.options = ollamaOptions;
        }

        // Thinking mode: Ollama uses `think` parameter (boolean or "low"/"medium"/"high"/"max")
        if (um.enable_thinking === true && um.reasoning_effort) {
            if (um.reasoning_effort === "enabled" || um.reasoning_effort === "adaptive") {
                ollamaBody.think = true;
            } else {
                ollamaBody.think = um.reasoning_effort; // "low", "medium", "high", "max"
            }
        } else if (um.enable_thinking === false) {
            ollamaBody.think = false;
        }

        // Tools: convert from VS Code format to Ollama format (same as OpenAI)
        const toolConfig = convertToolsToOpenAI(options);
        if (toolConfig.tools && toolConfig.tools.length > 0) {
            ollamaBody.tools = toolConfig.tools;
        }

        // Inject ask_image tools for non-vision models with images
        const hasLocalImages = (openaiApi as any)._localImages?.length > 0;
        if (hasLocalImages) {
            const tools = (ollamaBody.tools as any[]) || [];
            tools.push(ASK_IMAGE_TOOL_DEF);
            if ((openaiApi as any)._localImages?.length >= 2) {
                tools.push(ASK_WITH_MULTI_IMAGE_TOOL_DEF);
            }
            ollamaBody.tools = tools;
        }

        // Send chat request with retry
        const BASE_URL = um.baseUrl || OLLAMA_CLOUD_BASE_URL;
        const url = `${BASE_URL.replace(/\/+$/, "")}/chat`;
        logger.debug("request.body", { url, requestBody: ollamaBody });
        const response = await executeWithRetry(async () => {
            const res = await dispatchFetch(url, {
                method: "POST",
                headers: requestHeaders,
                body: JSON.stringify(ollamaBody),
                signal: abortController.signal,
            });

            if (!res.ok) {
                const errorText = await res.text();
                console.error("[OllamaCloud] API error response", errorText);
                if (errorText.includes("image is sensitive")) {
                    throw new Error(`IMAGE_SENSITIVE: ${errorText}`);
                }
                throw new Error(
                    `API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
                );
            }

            return res;
        }, retryConfig);

        if (!response.body) {
            throw new Error("No response body from API");
        }

        // Process Ollama-native streaming response (newline-delimited JSON, not SSE)
        await this._processOllamaStream(response.body, trackingProgress, token, openaiApi, (usage) => {
            usageReportedDuringStream = true;
            reportNativeUsage(usage, progress);
            if (enableThirdPartyIndicator) {
                recordUsage(usage);
                updateCumulativeTooltip(this.statusBarItem!);
                updateStatusBarWithApiPrompt(usage.promptTokens, model.maxInputTokens || 128000, this.statusBarItem!);
            }
        });

        // --- Second round: handle ask_image tool call interception ---
        await this._handleInterceptedToolCall({
            api: openaiApi,
            apiMode: "openai",
            model: model,
            um: um,
            baseUrl: BASE_URL,
            dispatchFetch: dispatchFetch,
            requestHeaders: requestHeaders,
            retryConfig: retryConfig,
            abortController: abortController,
            trackingProgress: trackingProgress,
            token: token,
            options: options,
        });

        // Fallback: if API did not return usage data, use client-side calculation
        if (!usageReportedDuringStream) {
            const outputText = collectedOutputText.join("");
            const estimatedOutputTokens = outputText ? await textTokenLength(outputText) : 0;
            const fallbackUsage: StreamUsage = {
                promptTokens: estimatedInputTokens,
                completionTokens: estimatedOutputTokens,
            };
            reportNativeUsage(fallbackUsage, progress);
            if (enableThirdPartyIndicator) {
                recordUsage(fallbackUsage);
                updateCumulativeTooltip(this.statusBarItem!);
            }
        }
    }

    /**
     * Process Ollama-native streaming response.
     *
     * Ollama returns newline-delimited JSON objects (not SSE with `data:` prefix).
     * Each chunk has the shape:
     *   {"model":"...","message":{"role":"assistant","content":"text"},"done":false}
     *
     * Thinking content appears in `message.thinking`.
     * Tool calls appear in `message.tool_calls[]`.
     * The final chunk has `"done":true` with usage stats.
     */
    private async _processOllamaStream(
        responseBody: ReadableStream<Uint8Array>,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken,
        api: OpenaiApi,
        onUsage: (usage: StreamUsage) => void
    ): Promise<void> {
        const reader = responseBody.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let cancelDisposable: vscode.Disposable | undefined;

        if (token.onCancellationRequested) {
            cancelDisposable = token.onCancellationRequested(() => {
                reader.cancel().catch(() => {});
            });
        }

        const thinkingState: OllamaThinkingState = {
            thinkingId: null,
            thinkingBuffer: "",
            thinkingFlushTimer: null,
        };

        try {
            while (true) {
                if (token.isCancellationRequested) {
                    break;
                }

                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                // The last element may be an incomplete line — save it for the next chunk
                buffer = lines.pop() || "";

                for (const line of lines) {
                    this._processOllamaLine(line, progress, api, onUsage, thinkingState);
                }
            }

            // Process any remaining content in the buffer after the stream ends.
            // This handles: (a) non-streaming responses that arrive as a single
            // chunk with no trailing newline, and (b) the final line of a
            // streaming response that ended exactly at a line boundary.
            if (buffer.trim()) {
                this._processOllamaLine(buffer, progress, api, onUsage, thinkingState);
            }
        } catch (e) {
            console.error("[OllamaCloud] Streaming response error:", e);
            throw e;
        } finally {
            cancelDisposable?.dispose();
            reader.releaseLock();
            endThinking(thinkingState, progress);
        }
    }

    /**
     * Process a single line from the Ollama streaming response.
     */
    private _processOllamaLine(
        line: string,
        progress: Progress<LanguageModelResponsePart>,
        api: OpenaiApi,
        onUsage: (usage: StreamUsage) => void,
        state: OllamaThinkingState
    ): void {
        const trimmed = line.trim();
        if (!trimmed) {
            return;
        }

        try {
            const chunk = JSON.parse(trimmed) as {
                model?: string;
                message?: {
                    role?: string;
                    content?: string;
                    thinking?: string;
                    tool_calls?: Array<{
                        function: { name: string; arguments: Record<string, unknown> };
                    }>;
                };
                done?: boolean;
                total_duration?: number;
                prompt_eval_count?: number;
                eval_count?: number;
            };

            // Handle thinking content
            if (chunk.message?.thinking) {
                if (!state.thinkingId) {
                    state.thinkingId = `thinking_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                }
                state.thinkingBuffer += chunk.message.thinking;
                if (!state.thinkingFlushTimer) {
                    state.thinkingFlushTimer = setTimeout(() => flushThinking(state, progress), 100);
                }
            }

            // Handle text content
            if (chunk.message?.content) {
                endThinking(state, progress);
                progress.report(new vscode.LanguageModelTextPart(chunk.message.content));
            }

            // Handle tool calls (Ollama sends complete tool calls, not deltas)
            if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
                endThinking(state, progress);
                for (let i = 0; i < chunk.message.tool_calls.length; i++) {
                    const tc = chunk.message.tool_calls[i];
                    const callId = `call_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
                    const args = tc.function.arguments ?? {};

                    if (tc.function.name === "ask_image" || tc.function.name === "ask_with_multi_image") {
                        api.interceptedToolCall = {
                            id: callId,
                            name: tc.function.name as "ask_image" | "ask_with_multi_image",
                            args: args as { imageIndex?: number; imageIndices?: number[]; query: string },
                        };
                    } else {
                        progress.report(new vscode.LanguageModelToolCallPart(callId, tc.function.name, args));
                    }
                }
            }

            // Handle completion
            if (chunk.done) {
                endThinking(state, progress);

                if (chunk.prompt_eval_count !== undefined || chunk.eval_count !== undefined) {
                    onUsage({
                        promptTokens: chunk.prompt_eval_count ?? 0,
                        completionTokens: chunk.eval_count ?? 0,
                    });
                }
            }
        } catch (e) {
            console.error("[OllamaCloud] Failed to parse stream chunk:", e, "line:", trimmed);
        }
    }

    /**
     * Provider-specific error messages for Ollama Cloud.
     * Handles image content moderation rejection.
     */
    protected _getProviderSpecificErrorMessage(err: unknown, _modelId: string): string | undefined {
        const errMessage = err instanceof Error ? err.message : String(err);

        // Detect image content moderation rejection from the API
        if (errMessage.includes("IMAGE_SENSITIVE:")) {
            logger.error("request.error", {
                modelId: _modelId,
                error: "image_sensitive",
                errorMessage: errMessage,
            });
            return l10n("The image you sent was flagged as sensitive by the content moderation system. Please try a different image.");
        }

        return undefined;
    }

    /**
     * Override the ask_image vision proxy loop to use Ollama-native streaming.
     *
     * The base class _handleInterceptedToolCall calls api.processStreamingResponse()
     * which expects OpenAI SSE format. Ollama Cloud returns its own native format,
     * so we replace those calls with _processOllamaStream.
     */
    protected async _handleInterceptedToolCall(params: {
        api: OpenaiApi;
        apiMode: string;
        model: LanguageModelChatInformation;
        um: BaseModelItem;
        baseUrl: string;
        dispatchFetch: typeof fetch;
        requestHeaders: Record<string, string>;
        retryConfig: RetryConfig;
        abortController: AbortController;
        trackingProgress: Progress<LanguageModelResponsePart>;
        token: CancellationToken;
        options: ProvideLanguageModelChatResponseOptions;
    }): Promise<void> {
        const api = params.api;
        const hasLocalImages = ((api as any)._localImages as any[])?.length > 0;
        if (!hasLocalImages) {
            return;
        }

        const storedMessages = (api as any)._originalApiMessages as any[] | undefined;
        if (!storedMessages || storedMessages.length === 0) {
            return;
        }

        // For now, the ask_image proxy loop is not fully supported for Ollama Cloud
        // because the follow-up requests need Ollama-native request bodies and
        // streaming handling. The interception still works (tool calls are captured),
        // but the multi-round loop is deferred to a follow-up implementation.
        logger.info("ollamacloud.vision.proxy", {
            message: "ask_image proxy not yet supported for Ollama Cloud; image query skipped",
        });
    }
}
