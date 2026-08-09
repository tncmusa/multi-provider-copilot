/**
 * Cline Pass Chat Model Provider.
 *
 * Integrates Cline Pass models into GitHub Copilot Chat via the VS Code
 * Language Model API. Cline Pass exposes 11 open-weight models through a
 * single OpenAI-compatible Chat Completions endpoint at
 * https://api.cline.bot/api/v1.
 *
 * This provider extends BaseChatModelProvider which owns the shared request
 * lifecycle (timeout, cancellation, retry, base URL validation, headers,
 * delay, generic error handling, and the ask_image vision proxy loop).
 * Cline Pass-specific behavior: model discovery (hardcoded 11 models),
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

import type { ModelPreset } from "./types";
import { executeWithRetry } from "./utils";
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
import { logger } from "./logger";
import { l10n } from "./localize";
import {
    buildClinePassModelInfos,
    getClinePassModelConfig,
} from "./clinePassModels";

/**
 * VS Code Chat provider backed by the Cline Pass API.
 *
 * Cline Pass is a flat $9.99/month subscription that offers 2-5x the usage
 * on popular open coding models compared to standard API rate. The API is
 * OpenAI-compatible (Chat Completions format) with Bearer token auth.
 */
export class ClinePassChatModelProvider extends BaseChatModelProvider<BaseModelItem> {
    /**
     * Create a Cline Pass provider using the given secret storage for the API key.
     */
    constructor(secrets: vscode.SecretStorage) {
        super(secrets, undefined, {
            secretKey: "clinepass.apiKey",
            title: "Cline Pass Provider API Key",
            prompt: "Enter your Cline Pass API key",
            missingMessage: "Cline Pass API key not found",
        });
    }

    /**
     * Get the list of available Cline Pass language models.
     * The 11 models are hardcoded; capabilities are reused from the OpenCode Go catalog.
     */
    async provideLanguageModelChatInformation(
        _options: PrepareLanguageModelChatModelOptions,
        _token: CancellationToken
    ): Promise<LanguageModelChatInformation[]> {
        return buildClinePassModelInfos();
    }

    /**
     * Resolve the Cline Pass model configuration from the VS Code model object.
     */
    protected async resolveModelConfig(model: LanguageModelChatInformation): Promise<BaseModelItem> {
        const config = await getClinePassModelConfig(model.id);
        if (!config) {
            throw new Error(`Unknown Cline Pass model: ${model.id}`);
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
     * Dispatch the chat request to the Cline Pass backend.
     * Cline Pass is always OpenAI Chat Completions (no Anthropic/Responses endpoints).
     * The base class has already set up timeouts, retries, cancellation,
     * base URL validation, headers, and API key retrieval.
     */
    protected async dispatchChatRequest(params: DispatchChatRequestParams<BaseModelItem>): Promise<void> {
        const { model, config: um, messages, options, progress, token, abortController, retryConfig, dispatchFetch, requestHeaders } = params;

        const modelConfig = {
            includeReasoningInRequest: um.include_reasoning_in_request ?? true,
            vision: um.vision ?? false,
        };

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
                    console.error("[ClinePass] Progress.report failed", {
                        modelId: model.id,
                        error: e instanceof Error ? { name: e.name, message: e.message } : String(e),
                    });
                }
            },
        };

        // OpenAI Chat Completions API mode (Cline Pass is always OpenAI-compatible)
        const openaiApi = new OpenaiApi(model.id);
        openaiApi.onUsage = (usage) => {
            usageReportedDuringStream = true;
            // Always report to native Copilot indicator (use original progress, not trackingProgress wrapper)
            reportNativeUsage(usage, progress);
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
        const BASE_URL = um.baseUrl || "https://api.cline.bot/api/v1";
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
                console.error("[ClinePass] API error response", errorText);
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
            throw new Error("No response body from Cline Pass API");
        }

        await openaiApi.processStreamingResponse(response.body, trackingProgress, token);

        // --- Second round: handle ask_image tool call interception ---
        // (vision proxy for non-vision models — shared logic in base class)
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

        // Fallback: if API did not return usage data, use client-side calculation for native indicator
        if (!usageReportedDuringStream) {
            const outputText = collectedOutputText.join("");
            const estimatedOutputTokens = outputText ? await textTokenLength(outputText) : 0;
            const fallbackUsage: StreamUsage = {
                promptTokens: 0,
                completionTokens: estimatedOutputTokens,
            };
            reportNativeUsage(fallbackUsage, progress);
        }
    }

    /**
     * Provider-specific error messages for Cline Pass.
     * Handles image content moderation rejection.
     */
    protected _getProviderSpecificErrorMessage(err: unknown, modelId: string): string | undefined {
        const errMessage = err instanceof Error ? err.message : String(err);

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