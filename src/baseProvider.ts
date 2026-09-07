/**
 * Base Language Model Chat Provider.
 *
 * This module defines the shared request lifecycle and VS Code Language Model API
 * behavior used by both the OpenCode Go and Cline Pass providers. Provider-specific
 * details (endpoints, authentication, model discovery, request formatting) are
 * delegated to concrete subclasses.
 */

import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelChatProvider,
    LanguageModelChatRequestMessage,
    LanguageModelResponsePart,
    PrepareLanguageModelChatModelOptions,
    ProvideLanguageModelChatResponseOptions,
    Progress,
} from "vscode";
import * as path from "path";
import { convertToolsToOpenAI, createRetryConfig, executeWithRetry } from "./utils";
import { l10n, l10nFormat } from "./localize";
import { countMessageTokens } from "./provideToken";
import { logger } from "./logger";
import { CommonApi, type StreamUsage } from "./commonApi";
import { callVisionModel, callVisionModelMulti } from "./vision/imageProxy";
import { ASK_IMAGE_TOOL_DEF, ASK_WITH_MULTI_IMAGE_TOOL_DEF, ASK_WITH_MULTI_IMAGE_TOOL_NAME } from "./vision/types";
import type { StoredImage } from "./vision/types";
import { createVisionToolHistoryPart } from "./vision/historyPart";
import type { VisionToolHistoryEntry } from "./vision/historyCodec";
import { resolveVisionProxyModelId } from "./catalogModels";

/**
 * Common model configuration fields consumed by the shared provider lifecycle
 * and by the OpenAI/Anthropic/Responses API implementations.
 */
export interface BaseModelItem {
    id: string;
    displayName?: string;
    baseUrl?: string;
    apiMode?: "openai" | "anthropic" | "responses";
    context_length?: number;
    max_tokens?: number;
    max_completion_tokens?: number;
    vision?: boolean;
    reasoning_effort?: string;
    enable_thinking?: boolean;
    thinking_budget?: number;
    /** Whether this model supports switching thinking on/off ("switchable"), always has it ("always"), or only disabled/adaptive ("adaptive") */
    thinkingMode?: "switchable" | "always" | "adaptive";
    temperature?: number | null;
    top_p?: number | null;
    top_k?: number;
    min_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    repetition_penalty?: number;
    reasoning?: {
        effort?: string;
        exclude?: boolean;
        max_tokens?: number;
        enabled?: boolean;
    };
    supportsTemperature?: boolean;
    include_reasoning_in_request?: boolean;
    headers?: Record<string, string>;
    extra?: Record<string, unknown>;
    delay?: number;
    [key: string]: unknown;
}

/**
 * Native Copilot Token Indicator
 *
 * Reports token usage to the Copilot Chat's built-in token indicator by emitting
 * a LanguageModelDataPart with MIME type 'usage'. Copilot Chat intercepts this
 * part and displays it in the native UI element, just like GitHub Copilot's own
 * models do.
 */
export function reportNativeUsage(
    usage: StreamUsage,
    progress: Progress<LanguageModelResponsePart>
): void {
    progress.report(
        new vscode.LanguageModelDataPart(
            new TextEncoder().encode(JSON.stringify({
                prompt_tokens: usage.promptTokens,
                completion_tokens: usage.completionTokens,
                total_tokens: usage.promptTokens + usage.completionTokens,
                prompt_tokens_details: {
                    cached_tokens: usage.cacheHitTokens ?? 0,
                },
            })),
            'usage'
        )
    );
}

/**
 * Extract the requested reasoning effort from the VS Code options object.
 */
export function getRequestedReasoningEffort(options: ProvideLanguageModelChatResponseOptions): string | undefined {
    const modelConfigurationEffort = options.modelConfiguration?.reasoningEffort;
    if (typeof modelConfigurationEffort === "string") {
        return modelConfigurationEffort;
    }

    const modelOptions = (options as unknown as { modelOptions?: Record<string, unknown> }).modelOptions;
    const modelOptionsThinking = modelOptions?.thinking as { type?: unknown } | undefined;
    if (modelOptionsThinking?.type === false) {
        return "disabled";
    }

    const modelOptionsEffort = modelOptions?.reasoning_effort ?? modelOptions?.reasoningEffort;
    return typeof modelOptionsEffort === "string" ? modelOptionsEffort : undefined;
}

/**
 * Parameters passed to the provider-specific chat request dispatcher.
 */
export interface DispatchChatRequestParams<TModelItem extends BaseModelItem> {
    /** VS Code model information. */
    model: LanguageModelChatInformation;
    /** Resolved model configuration. */
    config: TModelItem;
    /** Original VS Code chat messages. */
    messages: readonly LanguageModelChatRequestMessage[];
    /** VS Code response options. */
    options: ProvideLanguageModelChatResponseOptions;
    /** Progress reporter that should be used for this request. */
    progress: Progress<LanguageModelResponsePart>;
    /** VS Code cancellation token. */
    token: CancellationToken;
    /** AbortController tied to the request timeout and user cancellation. */
    abortController: AbortController;
    /** Retry configuration. */
    retryConfig: ReturnType<typeof createRetryConfig>;
    /** Fetch implementation (possibly undici-backed). */
    dispatchFetch: typeof fetch;
    /** Pre-built request headers including auth. */
    requestHeaders: Record<string, string>;
}

/**
 * Shared base class for chat model providers.
 *
 * Concrete providers implement model discovery, model config resolution,
 * request option application, and the actual HTTP dispatch. The base class owns
 * the common VS Code Language Model API lifecycle: token counting, timeout,
 * cancellation, retry setup, base URL validation, and generic error handling.
 */
export abstract class BaseChatModelProvider<TModelItem extends BaseModelItem> implements LanguageModelChatProvider {
    /** Track last request completion time for delay calculation. */
    protected _lastRequestTime: number | null = null;

    /**
     * @param secrets VS Code secret storage for the API key.
     * @param statusBarItem Optional status bar item used by this provider.
     * @param apiKeyConfig Secret key and prompt text used when asking for the API key.
     */
    constructor(
        protected readonly secrets: vscode.SecretStorage,
        protected readonly statusBarItem: vscode.StatusBarItem | undefined,
        protected readonly apiKeyConfig: {
            secretKey: string;
            title: string;
            prompt: string;
            missingMessage: string;
        }
    ) { }

    /**
     * Get the list of available language models contributed by this provider.
     */
    abstract provideLanguageModelChatInformation(
        options: PrepareLanguageModelChatModelOptions,
        token: CancellationToken
    ): Promise<LanguageModelChatInformation[]>;

    /**
     * Returns the number of tokens for a given text using the model tokenizer logic.
     */
    async provideTokenCount(
        _model: LanguageModelChatInformation,
        text: string | LanguageModelChatRequestMessage,
        _token: CancellationToken
    ): Promise<number> {
        return countMessageTokens(text, { includeReasoningInRequest: true });
    }

    /**
     * Resolve the provider-specific model configuration from the VS Code model object.
     */
    protected abstract resolveModelConfig(model: LanguageModelChatInformation): TModelItem | Promise<TModelItem>;

    /**
     * Apply provider-specific request options (reasoning effort, temperature, etc.)
     * to the resolved model configuration. Must return a shallow copy when mutating.
     */
    protected abstract applyRequestOptions(config: TModelItem, options: ProvideLanguageModelChatResponseOptions): TModelItem;

    /**
     * Dispatch the chat request to the provider's backend.
     * The base class has already set up timeouts, retries, and cancellation.
     */
    protected abstract dispatchChatRequest(params: DispatchChatRequestParams<TModelItem>): Promise<void>;

    /**
     * Returns the response for a chat request, passing the results to the progress callback.
     */
    async provideLanguageModelChatResponse(
        model: LanguageModelChatInformation,
        messages: readonly LanguageModelChatRequestMessage[],
        options: ProvideLanguageModelChatResponseOptions,
        progress: Progress<LanguageModelResponsePart>,
        token: CancellationToken
    ): Promise<void> {
        const trackingProgress: Progress<LanguageModelResponsePart> = {
            report: (part) => {
                try {
                    progress.report(part);
                } catch (e) {
                    console.error(`[${this.apiKeyConfig.secretKey.split(".")[0] ?? "provider"}] Progress.report failed`, {
                        modelId: model.id,
                        error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
                    });
                }
            },
        };

        const requestStartTime = Date.now();
        let abortController = new AbortController();
        let requestTimeoutMs = 600000;
        let timeoutId: ReturnType<typeof setTimeout> | undefined;
        let dispatchFetch: typeof fetch;

        try {
            const config = this.applyRequestOptions(await this.resolveModelConfig(model), options);

            const baseUrl = config.baseUrl;
            logger.info("request.start", {
                provider: this.apiKeyConfig.secretKey,
                modelId: model.id,
                messageCount: messages.length,
                apiMode: config.apiMode || "openai",
                baseUrl,
            });

            // Apply delay between consecutive requests
            const delayMs = config.delay ?? vscode.workspace.getConfiguration().get<number>("opencodego.delay", 0);
            if (delayMs > 0 && this._lastRequestTime !== null) {
                const elapsed = Date.now() - this._lastRequestTime;
                if (elapsed < delayMs) {
                    const remainingDelay = delayMs - elapsed;
                    logger.debug("request.delay", { delayMs, elapsed, remainingDelay });
                    await new Promise<void>((resolve) => {
                        const timeout = setTimeout(() => {
                            clearTimeout(timeout);
                            resolve();
                        }, remainingDelay);
                    });
                }
            }

            // Get API key
            const apiKey = await this.ensureApiKey();
            if (!apiKey) {
                logger.warn("apiKey.missing", {});
                throw new Error(l10n(this.apiKeyConfig.missingMessage));
            }

            // Validate base URL (reject plain HTTP for remote addresses)
            this._validateBaseUrl(baseUrl);

            // Get retry config
            const retryConfig = createRetryConfig();

            // Create request timeout abort controller (default: 10 minutes)
            requestTimeoutMs = vscode.workspace.getConfiguration().get<number>("opencodego.requestTimeout", 600000);
            abortController = new AbortController();
            timeoutId = setTimeout(() => abortController.abort(), requestTimeoutMs);
            // Connect VS Code cancellation token to abort the fetch immediately when user stops
            if (token.onCancellationRequested) {
                token.onCancellationRequested(() => {
                    if (!abortController.signal.aborted) {
                        abortController.abort();
                    }
                });
            }
            // Create undici fetch with custom bodyTimeout (extends TCP idle timeout during streaming)
            dispatchFetch = this._createFetchWithTimeout(requestTimeoutMs);

            const apiMode = config.apiMode || "openai";
            const requestHeaders = this._prepareHeaders(apiKey, apiMode, config.headers);
            logger.debug("request.headers", {
                headers: logger.sanitizeHeaders(requestHeaders as Record<string, string>),
            });

            await this.dispatchChatRequest({
                model,
                config,
                messages,
                options,
                progress: trackingProgress,
                token,
                abortController,
                retryConfig,
                dispatchFetch,
                requestHeaders,
            });
        } catch (err) {
            this._handleRequestError(err, token, abortController, requestTimeoutMs, requestStartTime, model.id, messages.length);
        } finally {
            clearTimeout(timeoutId);
            const durationMs = Date.now() - requestStartTime;
            logger.info("request.end", { provider: this.apiKeyConfig.secretKey, modelId: model.id, durationMs });
            this._lastRequestTime = Date.now();
        }
    }

    /**
     * Ensure an API key exists in SecretStorage, prompting the user when missing.
     */
    protected async ensureApiKey(): Promise<string | undefined> {
        let apiKey = await this.secrets.get(this.apiKeyConfig.secretKey);

        if (!apiKey) {
            const entered = await vscode.window.showInputBox({
                title: l10n(this.apiKeyConfig.title),
                prompt: l10n(this.apiKeyConfig.prompt),
                ignoreFocusOut: true,
                password: true,
            });
            if (entered && entered.trim()) {
                apiKey = entered.trim();
                await this.secrets.store(this.apiKeyConfig.secretKey, apiKey);
            }
        }

        return apiKey;
    }

    /**
     * Prepare HTTP request headers. Subclasses can override for provider-specific headers.
     */
    protected _prepareHeaders(
        apiKey: string,
        apiMode: string,
        customHeaders?: Record<string, string>
    ): Record<string, string> {
        return CommonApi.prepareHeaders(apiKey, apiMode, customHeaders);
    }

    /**
     * Create an undici fetch function with custom bodyTimeout to prevent premature
     * connection termination during long streaming responses.
     * Falls back to global fetch if undici is unavailable.
     */
    protected _createFetchWithTimeout(requestTimeoutMs: number): typeof fetch {
        try {
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            const undici = require(path.join(vscode.env.appRoot, 'node_modules', 'undici'));
            const agent = new undici.Agent({ bodyTimeout: requestTimeoutMs });
            return (url: RequestInfo | URL, init?: RequestInit) => {
                return undici.fetch(url, { ...init, dispatcher: agent });
            };
        } catch {
            return fetch;
        }
    }

    /**
     * Validate that the base URL uses HTTPS for remote endpoints.
     */
    protected _validateBaseUrl(baseUrl: string | undefined): void {
        if (!baseUrl || !baseUrl.startsWith("http")) {
            throw new Error(l10n("Invalid base URL configuration."));
        }
        const url = new URL(baseUrl);
        if (url.protocol === "http:") {
            const host = url.hostname.toLowerCase();
            const isLocal = host === "localhost" || host === "127.0.0.1" || host === "::1"
                || host.startsWith("192.168.") || host.startsWith("10.") || host === "0.0.0.0"
                || /^172\.(1[6-9]|2\d|3[01])\./.test(host);
            if (!isLocal) {
                throw new Error(l10n("Plain HTTP is only allowed for localhost or private network addresses. Use HTTPS for remote endpoints."));
            }
        }
    }

    /**
     * Hook for provider-specific error messages. Return a user-facing message
     * string when this provider wants to override generic error handling.
     */
    protected _getProviderSpecificErrorMessage(err: unknown, _modelId: string): string | undefined {
        return undefined;
    }

    /**
     * Classify request errors and throw provider-friendly messages.
     */
    protected _handleRequestError(
        err: unknown,
        token: CancellationToken,
        abortController: AbortController,
        requestTimeoutMs: number,
        requestStartTime: number,
        modelId: string,
        messageCount: number
    ): never {
        const errMessage = err instanceof Error ? err.message : String(err);
        const isUserCancelled = token.isCancellationRequested;
        const isTimeout = abortController.signal.aborted && !isUserCancelled;
        const isForceTerminated =
            !isTimeout &&
            !isUserCancelled &&
            (errMessage.includes("terminated") ||
                errMessage.includes("aborted") ||
                (err instanceof Error && err.name === "AbortError"));

        if (isUserCancelled) {
            throw err;
        }

        if (isTimeout || isForceTerminated) {
            logger.error("request.timeout", {
                modelId,
                timeoutMs: requestTimeoutMs,
                durationMs: Date.now() - requestStartTime,
                reason: isForceTerminated ? "connection_terminated" : "timeout",
            });
            if (isForceTerminated) {
                throw new Error(l10n("The connection was closed by the server. The generation took too long. Please try again or request shorter content."));
            }
            throw new Error(l10n("Request timed out. The generation took too long. You can increase the timeout in settings (opencodego.requestTimeout)."));
        }

        const providerMessage = this._getProviderSpecificErrorMessage(err, modelId);
        if (providerMessage) {
            logger.error("request.error", {
                modelId,
                messageCount,
                errorName: err instanceof Error ? err.name : String(err),
                errorMessage: err instanceof Error ? err.message : String(err),
            });
            throw new Error(providerMessage);
        }

        console.error(`[${this.apiKeyConfig.secretKey.split(".")[0] ?? "provider"}] Chat request failed`, {
            modelId,
            messageCount,
            error: err instanceof Error ? { name: err.name, message: err.message } : String(err),
        });
        logger.error("request.error", {
            modelId,
            messageCount,
            errorName: err instanceof Error ? err.name : String(err),
            errorMessage: err instanceof Error ? err.message : String(err),
        });
        throw err;
    }

    /**
     * Handle an ask_image tool call interception by calling the vision model
     * with the model's specific query and making a second round API request
     * with the tool call + result. Shared by all providers that use the
     * ask_image vision proxy for non-vision models.
     */
    protected async _handleInterceptedToolCall(params: {
        api: CommonApi<any, any>;
        apiMode: string;
        model: LanguageModelChatInformation;
        um: BaseModelItem;
        baseUrl: string;
        dispatchFetch: typeof fetch;
        requestHeaders: Record<string, string>;
        retryConfig: ReturnType<typeof createRetryConfig>;
        abortController: AbortController;
        trackingProgress: Progress<LanguageModelResponsePart>;
        token: CancellationToken;
        options: ProvideLanguageModelChatResponseOptions;
    }): Promise<void> {
        const api = params.api;
        const storedMessages = (api as any)._originalApiMessages as any[] | undefined;
        const hasLocalImages = ((api as any)._localImages as any[])?.length > 0;

        if (!hasLocalImages) {
            logger.debug("vision.no-stored-images", { hasStoredMessages: !!storedMessages });
            return;
        }
        if (!storedMessages || storedMessages.length === 0) {
            logger.warn("vision.no-second-round-messages", {});
            return;
        }

        const config = vscode.workspace.getConfiguration();
        const visionModelId = await resolveVisionProxyModelId(
            config.get<string>("opencodego.visionProxyModel", "qwen-plus-latest")
        );
        const maxRounds = config.get<number>("opencodego.visionMaxRounds", 5);

        let currentMessages: any[] = [...storedMessages];

        for (let round = 1; round <= maxRounds; round++) {
            const intercepted = api.interceptedToolCall;
            if (!intercepted) {
                break;
            }
            api.interceptedToolCall = null;

            logger.info("vision.intercepted", {
                round,
                toolName: intercepted.name,
                imageIndex: intercepted.args.imageIndex,
                imageIndices: intercepted.args.imageIndices,
                query: intercepted.args.query,
                apiMode: params.apiMode,
            });

            const visionPrompt = intercepted.args.query;

            const questionThinkId = `vision_q_${Date.now()}_${round}`;
            params.trackingProgress.report(
                new vscode.LanguageModelThinkingPart(
                    l10nFormat("Querying vision model: \"{0}\"", visionPrompt ?? ""),
                    questionThinkId
                ) as unknown as LanguageModelResponsePart
            );
            params.trackingProgress.report(
                new vscode.LanguageModelThinkingPart("", questionThinkId) as unknown as LanguageModelResponsePart
            );

            const thinkBlockId = `vision_think_${Date.now()}_${round}`;
            const textBlockId = `vision_text_${Date.now()}_${round}`;

            const visionProgress = {
                onThinking: (text: string) => {
                    params.trackingProgress.report(
                        new vscode.LanguageModelThinkingPart(text, thinkBlockId) as unknown as LanguageModelResponsePart
                    );
                },
                onText: (text: string) => {
                    params.trackingProgress.report(
                        new vscode.LanguageModelThinkingPart(text, textBlockId) as unknown as LanguageModelResponsePart
                    );
                },
            };

            let description: string;
            try {
                if (intercepted.name === ASK_WITH_MULTI_IMAGE_TOOL_NAME) {
                    const indices = intercepted.args.imageIndices ?? [];
                    const images: StoredImage[] = [];
                    for (const idx of indices) {
                        const img = api.getStoredImage(idx);
                        if (img) images.push(img);
                    }
                    if (images.length < 2) {
                        logger.warn("vision.not-enough-images", { indices });
                        description = "[Not enough images for comparison]";
                    } else {
                        description = await callVisionModelMulti(images, visionModelId, visionPrompt, params.token, visionProgress);
                    }
                } else {
                    const storedImage = api.getStoredImage(intercepted.args.imageIndex ?? 0);
                    if (!storedImage) {
                        logger.warn("vision.image-not-found", { imageIndex: intercepted.args.imageIndex });
                        description = "[Image not found]";
                    } else {
                        description = await callVisionModel(
                            storedImage.data,
                            storedImage.mimeType,
                            visionModelId,
                            visionPrompt,
                            params.token,
                            visionProgress
                        );
                    }
                }
            } catch (err) {
                const errMsg = err instanceof Error ? err.message : String(err);
                logger.error("vision.call-failed", { error: errMsg, visionModelId });
                description = "[Image query unavailable]";
            }

            params.trackingProgress.report(
                new vscode.LanguageModelThinkingPart("", thinkBlockId) as unknown as LanguageModelResponsePart
            );
            params.trackingProgress.report(
                new vscode.LanguageModelThinkingPart("", textBlockId) as unknown as LanguageModelResponsePart
            );

            const previousReasoning = params.apiMode === "openai"
                ? ((api as any)._capturedReasoningContent as string | undefined)
                : undefined;
            const historyEntry: VisionToolHistoryEntry = {
                id: intercepted.id,
                name: intercepted.name as VisionToolHistoryEntry["name"],
                args: intercepted.args,
                result: description,
                ...(previousReasoning !== undefined ? { reasoningContent: previousReasoning } : {}),
            };
            params.trackingProgress.report(
                createVisionToolHistoryPart(historyEntry) as unknown as LanguageModelResponsePart
            );

            if (params.token.isCancellationRequested) {
                logger.info("vision.skipped-round", { round, reason: "user_cancelled" });
                break;
            }

            const roundAbortController = new AbortController();
            const roundTimeoutMs = vscode.workspace.getConfiguration().get<number>("opencodego.requestTimeout", 600000);
            const roundTimeoutId = setTimeout(() => {
                if (!roundAbortController.signal.aborted) {
                    roundAbortController.abort();
                }
            }, roundTimeoutMs);
            if (params.token.onCancellationRequested) {
                params.token.onCancellationRequested(() => {
                    if (!roundAbortController.signal.aborted) {
                        roundAbortController.abort();
                    }
                });
            }

            try {
                if (params.apiMode === "anthropic") {
                    currentMessages.push({
                        role: "assistant" as const,
                        content: [
                            { type: "tool_use" as const, id: intercepted.id, name: intercepted.name, input: intercepted.args },
                        ],
                    });
                    currentMessages.push({
                        role: "user" as const,
                        content: [
                            { type: "tool_result" as const, tool_use_id: intercepted.id, content: description },
                        ],
                    });

                    const body: Record<string, unknown> = {
                        model: params.um.id ?? params.model.id,
                        messages: currentMessages,
                        stream: true,
                    };
                    if (params.um.max_completion_tokens !== undefined) {
                        body.max_tokens = params.um.max_completion_tokens;
                    } else if (params.um.max_tokens !== undefined) {
                        body.max_tokens = params.um.max_tokens;
                    }
                    if (params.um.temperature !== undefined && params.um.temperature !== null) {
                        if (params.um.supportsTemperature !== false) {
                            body.temperature = params.um.temperature;
                        }
                    }
                    const systemContent = (params.api as any)._systemContent as string | undefined;
                    if (systemContent) {
                        body.system = systemContent;
                    }
                    if (params.um.enable_thinking === true) {
                        if (params.um.reasoning_effort === 'adaptive') {
                            body.thinking = { type: "adaptive" };
                        } else {
                            body.thinking = { type: "enabled", budget_tokens: 8192 };
                        }
                    } else {
                        body.thinking = { type: "disabled" as const };
                    }

                    const anthropicToolList: Array<{ name: string; description?: string; input_schema?: object }> = [];
                    const toolConfig = convertToolsToOpenAI(params.options);
                    if (toolConfig.tools) {
                        for (const tool of toolConfig.tools) {
                            anthropicToolList.push({
                                name: tool.function.name,
                                description: tool.function.description,
                                input_schema: tool.function.parameters,
                            });
                        }
                    }
                    if (hasLocalImages) {
                        const singleDef = ASK_IMAGE_TOOL_DEF as unknown as { function: { name: string; description: string; parameters: object } };
                        anthropicToolList.push({
                            name: singleDef.function.name,
                            description: singleDef.function.description,
                            input_schema: singleDef.function.parameters,
                        });
                        if (((api as any)._localImages as any[])?.length >= 2) {
                            const multiDef = ASK_WITH_MULTI_IMAGE_TOOL_DEF as unknown as { function: { name: string; description: string; parameters: object } };
                            anthropicToolList.push({
                                name: multiDef.function.name,
                                description: multiDef.function.description,
                                input_schema: multiDef.function.parameters,
                            });
                        }
                    }
                    if (anthropicToolList.length > 0) {
                        body.tools = anthropicToolList;
                    }
                    if (hasLocalImages) {
                        body.tool_choice = { type: "auto" };
                    }

                    const normalizedUrl = params.baseUrl.replace(/\/+$/, "");
                    const url = normalizedUrl.endsWith("/v1")
                        ? `${normalizedUrl}/messages`
                        : `${normalizedUrl}/v1/messages`;

                    const response = await executeWithRetry(async () => {
                        const res = await params.dispatchFetch(url, {
                            method: "POST",
                            headers: params.requestHeaders,
                            body: JSON.stringify(body),
                            signal: roundAbortController.signal,
                        });
                        if (!res.ok) {
                            const errorText = await res.text();
                            throw new Error(`Anthropic API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}`);
                        }
                        return res;
                    }, params.retryConfig);

                    if (response.body) {
                        await api.processStreamingResponse(response.body, params.trackingProgress, params.token);
                    }
                } else {
                    const prevReasoning = previousReasoning ?? "";
                    (api as any)._capturedReasoningContent = "";
                    currentMessages.push({
                        role: "assistant" as const,
                        reasoning_content: prevReasoning,
                        tool_calls: [
                            {
                                id: intercepted.id,
                                type: "function" as const,
                                function: {
                                    name: intercepted.name,
                                    arguments: JSON.stringify(intercepted.args),
                                },
                            },
                        ],
                    });
                    currentMessages.push({
                        role: "tool" as const,
                        tool_call_id: intercepted.id,
                        content: description,
                    });

                    const body: Record<string, unknown> = {
                        model: params.um.id ?? params.model.id,
                        messages: currentMessages,
                        stream: true,
                        stream_options: { include_usage: true },
                    };
                    if (params.um.temperature !== undefined && params.um.temperature !== null) {
                        if (params.um.supportsTemperature !== false) {
                            body.temperature = params.um.temperature;
                        }
                    }
                    if (params.um.top_p !== undefined && params.um.top_p !== null) {
                        body.top_p = params.um.top_p;
                    }
                    if (params.um.max_completion_tokens !== undefined) {
                        body.max_completion_tokens = params.um.max_completion_tokens;
                    }
                    if (params.um.enable_thinking !== false && params.um.reasoning_effort !== undefined && params.um.reasoning_effort !== 'adaptive') {
                        body.reasoning_effort = params.um.reasoning_effort;
                    }
                    if (params.um.enable_thinking === true) {
                        body.thinking = { type: "enabled" };
                    } else {
                        body.thinking = { type: "disabled" };
                    }

                    const openaiToolList: any[] = [];
                    const toolConfig = convertToolsToOpenAI(params.options);
                    if (toolConfig.tools) {
                        openaiToolList.push(...toolConfig.tools);
                    }
                    if (hasLocalImages) {
                        openaiToolList.push(ASK_IMAGE_TOOL_DEF);
                        if (((api as any)._localImages as any[])?.length >= 2) {
                            openaiToolList.push(ASK_WITH_MULTI_IMAGE_TOOL_DEF);
                        }
                    }
                    if (openaiToolList.length > 0) {
                        body.tools = openaiToolList;
                    }
                    if (hasLocalImages) {
                        body.tool_choice = "auto";
                    }

                    const url = `${params.baseUrl.replace(/\/+$/, "")}/chat/completions`;
                    const response = await executeWithRetry(async () => {
                        const res = await params.dispatchFetch(url, {
                            method: "POST",
                            headers: params.requestHeaders,
                            body: JSON.stringify(body),
                            signal: roundAbortController.signal,
                        });
                        if (!res.ok) {
                            const errorText = await res.text();
                            throw new Error(`API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}`);
                        }
                        return res;
                    }, params.retryConfig);

                    if (response.body) {
                        await api.processStreamingResponse(response.body, params.trackingProgress, params.token);
                    }
                }
            } finally {
                clearTimeout(roundTimeoutId);
            }
        }
    }
}
