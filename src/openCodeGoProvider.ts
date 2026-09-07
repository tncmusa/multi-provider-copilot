import * as vscode from "vscode";
import {
    CancellationToken,
    LanguageModelChatInformation,
    LanguageModelResponsePart,
    PrepareLanguageModelChatModelOptions,
    ProvideLanguageModelChatResponseOptions,
    Progress,
} from "vscode";

import type { ModelPreset, OpenCodeGoModelItem } from "./types";

import { executeWithRetry } from "./utils";
import { getCatalogProviderBaseUrl } from "./modelsDev";

import { prepareLanguageModelChatInformation } from "./openCodeGoModels";
import { getCatalogModelConfig, resolveProviderForModelId } from "./catalogModels";
import { l10nFormat, l10n } from "./localize";
import { textTokenLength } from "./provideToken";
import { updateContextStatusBar, recordUsage, updateCumulativeTooltip, updateStatusBarWithApiPrompt } from "./statusBar";
import { OpenaiApi } from "./openai/openaiApi";
import { AnthropicApi } from "./anthropic/anthropicApi";
import { ResponsesApi } from "./responses/responsesApi";
import type { ResponsesRequestBody } from "./responses/responsesTypes";
import type { AnthropicRequestBody } from "./anthropic/anthropicTypes";
import type { StreamUsage } from "./commonApi";
import { logger } from "./logger";
import {
    BaseChatModelProvider,
    reportNativeUsage,
    getRequestedReasoningEffort,
    type DispatchChatRequestParams,
} from "./baseProvider";

/**
 * VS Code Chat provider backed by OpenCode Go API.
 *
 * Extends BaseChatModelProvider which owns the shared request lifecycle
 * (timeout, cancellation, retry, base URL validation, headers, delay,
 * generic error handling). This class implements OpenCode Go-specific
 * behavior: model discovery, reasoning/temperature option application,
 * the 3-way apiMode dispatch (OpenAI/Anthropic/Responses), status bar
 * integration, the ask_image vision proxy loop, and provider-specific
 * error messages (Zen 401, IMAGE_SENSITIVE).
 */
export class OpenCodeGoChatModelProvider extends BaseChatModelProvider<OpenCodeGoModelItem> {
    /**
     * Create a provider using the given secret storage for the API key.
     */
    constructor(
        secrets: vscode.SecretStorage,
        statusBarItem: vscode.StatusBarItem
    ) {
        super(secrets, statusBarItem, {
            secretKey: "opencodego.apiKey",
            title: "OpenCode Go Provider API Key",
            prompt: "Enter your OpenCode Go API key",
            missingMessage: "OpenCode Go API key not found",
        });
    }

    /**
     * Get the list of available language models contributed by this provider.
     */
    async provideLanguageModelChatInformation(
        options: PrepareLanguageModelChatModelOptions,
        _token: CancellationToken
    ): Promise<LanguageModelChatInformation[]> {
        return prepareLanguageModelChatInformation(options, _token, this.secrets);
    }

    /**
     * Resolve the OpenCode Go model configuration from the VS Code model object.
     */
    protected resolveModelConfig(model: LanguageModelChatInformation): OpenCodeGoModelItem {
        return getCatalogModelConfig(model.id);
    }

    /**
     * Apply reasoning effort and temperature/top_p from model preset or custom settings.
     * Returns a shallow copy to avoid mutating the shared resolved config.
     */
    protected applyRequestOptions(baseConfig: OpenCodeGoModelItem, options: ProvideLanguageModelChatResponseOptions): OpenCodeGoModelItem {
        const um: OpenCodeGoModelItem = { ...baseConfig };

        // Apply reasoning effort from model configuration to determine thinking mode
        const effort = getRequestedReasoningEffort(options);
        if (effort) {
            if (effort === "disabled") {
                if (um.thinkingMode !== "always") {
                    um.enable_thinking = false;
                    um.include_reasoning_in_request = false;
                    um.reasoning_effort = undefined;
                } else {
                    // Grok 4.5 requires thinking; never send a disabled
                    // reasoning setting even if an older client requests it.
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
                    // Keep top_p undefined so the model uses its default
                    um.top_p = undefined;
                }
            }
        } else {
            // Model does not support temperature; ensure it's not sent
            um.temperature = undefined;
            um.top_p = undefined;
        }

        return um;
    }

    /**
     * Dispatch the chat request to the OpenCode Go backend.
     * The base class has already set up timeouts, retries, cancellation,
     * base URL validation, headers, and API key retrieval.
     */
    protected async dispatchChatRequest(params: DispatchChatRequestParams<OpenCodeGoModelItem>): Promise<void> {
        const { model, config: um, messages, options, progress, token, abortController, retryConfig, dispatchFetch, requestHeaders } = params;
        const vscodeConfig = vscode.workspace.getConfiguration();

        // Prepare model configuration
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
                    console.error("[OpenCodeGo] Progress.report failed", {
                        modelId: model.id,
                        error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
                    });
                }
            },
        };

        // Calculate client-side token estimate for fallback (also updates Advanced Token indicator if enabled)
        const estimatedInputTokens = await updateContextStatusBar(messages, options.tools, model, this.statusBarItem!, modelConfig);

        const apiMode = um.apiMode || "openai";
        const BASE_URL = um.baseUrl || getCatalogProviderBaseUrl("opencode-go", "https://opencode.ai/zen/go/v1/");

        if (apiMode === "anthropic") {
            // Anthropic API mode
            const anthropicApi = new AnthropicApi(model.id);
            anthropicApi.onUsage = (usage) => {
                usageReportedDuringStream = true;
                // Always report to native Copilot indicator (use original progress, not trackingProgress wrapper)
                reportNativeUsage(usage, progress);
                // Conditionally update Advanced Token indicator
                if (enableThirdPartyIndicator) {
                    recordUsage(usage);
                    updateCumulativeTooltip(this.statusBarItem!);
                    updateStatusBarWithApiPrompt(usage.promptTokens, model.maxInputTokens || 128000, this.statusBarItem!);
                }
            };
            const anthropicMessages = await anthropicApi.convertMessages(messages, modelConfig);

            // requestBody
            let requestBody: AnthropicRequestBody = {
                model: um.id ?? model.id,
                messages: anthropicMessages,
                stream: true,
            };
            requestBody = anthropicApi.prepareRequestBody(requestBody, um, options);

            // Build Anthropic messages endpoint URL
            const normalizedBaseUrl = BASE_URL.replace(/\/+$/, "");
            const url = normalizedBaseUrl.endsWith("/v1")
                ? `${normalizedBaseUrl}/messages`
                : `${normalizedBaseUrl}/v1/messages`;
            logger.debug("request.body", { url, requestBody });
            const response = await executeWithRetry(async () => {
                const res = await dispatchFetch(url, {
                    method: "POST",
                    headers: requestHeaders,
                    body: JSON.stringify(requestBody),
                    signal: abortController.signal,
                });

                if (!res.ok) {
                    const errorText = await res.text();
                    console.error("[Anthropic Provider] Anthropic API error response", errorText);
                    // Detect content moderation rejection for images — skip retries, this won't recover
                    if (errorText.includes("image is sensitive")) {
                        throw new Error(`IMAGE_SENSITIVE: ${errorText}`);
                    }
                    throw new Error(
                        `Anthropic API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`
                    );
                }

                return res;
            }, retryConfig);

            if (!response.body) {
                throw new Error("No response body from Anthropic API");
            }
            await anthropicApi.processStreamingResponse(response.body, trackingProgress, token);

            // --- Second round: handle ask_image tool call interception ---
            await this._handleInterceptedToolCall({
                api: anthropicApi,
                apiMode: "anthropic",
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
        } else if (apiMode === "responses") {
            const responsesApi = new ResponsesApi(model.id);
            responsesApi.onUsage = (usage) => {
                usageReportedDuringStream = true;
                reportNativeUsage(usage, progress);
                if (enableThirdPartyIndicator) {
                    recordUsage(usage);
                    updateCumulativeTooltip(this.statusBarItem!);
                    updateStatusBarWithApiPrompt(usage.promptTokens, model.maxInputTokens || 128000, this.statusBarItem!);
                }
            };
            const input = await responsesApi.convertMessages(messages, modelConfig);
            let requestBody: ResponsesRequestBody = {
                model: um.id ?? model.id,
                input,
                // OpenCode Go's Responses endpoint accepts the JSON response
                // form used by the working PowerShell example.
                stream: false,
            };
            requestBody = responsesApi.prepareRequestBody(requestBody, um, options);
            const url = `${BASE_URL.replace(/\/+$/, "")}/responses`;
            logger.debug("request.body", { url, requestBody });
            const response = await executeWithRetry(async () => {
                const res = await dispatchFetch(url, {
                    method: "POST",
                    headers: requestHeaders,
                    body: JSON.stringify(requestBody),
                    signal: abortController.signal,
                });
                if (!res.ok) {
                    const errorText = await res.text();
                    throw new Error(`Responses API error: [${res.status}] ${res.statusText}${errorText ? `\n${errorText}` : ""}\nURL: ${url}`);
                }
                return res;
            }, retryConfig);
            await responsesApi.processResponse(response, trackingProgress);
            await this._handleInterceptedToolCall({ api: responsesApi, apiMode: "responses", model, um, baseUrl: BASE_URL, dispatchFetch, requestHeaders, retryConfig, abortController, trackingProgress, token, options });
        } else {
            // OpenAI Chat Completions API mode
            const openaiApi = new OpenaiApi(model.id);
            openaiApi.onUsage = (usage) => {
                usageReportedDuringStream = true;
                // Always report to native Copilot indicator (use original progress, not trackingProgress wrapper)
                reportNativeUsage(usage, progress);
                // Conditionally update Advanced Token indicator
                if (enableThirdPartyIndicator) {
                    recordUsage(usage);
                    updateCumulativeTooltip(this.statusBarItem!);
                    updateStatusBarWithApiPrompt(usage.promptTokens, model.maxInputTokens || 128000, this.statusBarItem!);
                }
            };
            const openaiMessages = await openaiApi.convertMessages(messages, modelConfig);

            // requestBody
            let requestBody: Record<string, unknown> = {
                model: um.id ?? model.id,
                messages: openaiMessages,
                stream: true,
                stream_options: { include_usage: true },
            };

            requestBody = openaiApi.prepareRequestBody(requestBody, um, options);

            // Send chat request with retry
            const url = `${BASE_URL.replace(/\/+$/, "")}/chat/completions`;
            logger.debug("request.body", { url, requestBody });
            const response = await executeWithRetry(async () => {
                const res = await dispatchFetch(url, {
                    method: "POST",
                    headers: requestHeaders,
                    body: JSON.stringify(requestBody),
                    signal: abortController.signal,
                });

                if (!res.ok) {
                    const errorText = await res.text();
                    console.error("[OpenCodeGo] API error response", errorText);
                    // Detect content moderation rejection for images — skip retries, this won't recover
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

            await openaiApi.processStreamingResponse(response.body, trackingProgress, token);

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
        }

        // Fallback: if API did not return usage data, use client-side calculation for native indicator
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
     * Provider-specific error messages for OpenCode Go.
     * Handles Zen free model expiration (401) and image content moderation rejection.
     */
    protected _getProviderSpecificErrorMessage(err: unknown, modelId: string): string | undefined {
        const errMessage = err instanceof Error ? err.message : String(err);

        // Detect Zen free model expiration: a 401 from a Zen free model
        // means the free promotion has ended (error text may vary - don't match on it)
        if (errMessage.includes("[401]") && resolveProviderForModelId(modelId) === "opencode") {
            const zenConfig = getCatalogModelConfig(modelId);
            const zenModelName = zenConfig.displayName ?? modelId;
            logger.error("request.error", {
                modelId,
                error: "zen_free_model_expired",
                errorMessage: errMessage,
            });
            return l10nFormat("{0} is no longer available as a free model. Please use a different model.", zenModelName);
        }

        // Detect image content moderation rejection from the API
        if (errMessage.includes("IMAGE_SENSITIVE:")) {
            logger.error("request.error", {
                modelId,
                error: "image_sensitive",
                errorMessage: errMessage,
            });
            return l10n("The image you sent was flagged as sensitive by the content moderation system. Please try a different image.");
        }

        return undefined;
    }
}
