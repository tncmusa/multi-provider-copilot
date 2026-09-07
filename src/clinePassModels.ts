/**
 * Cline Pass model discovery and configuration.
 *
 * ClinePass exposes 11 open-weight models via a single OpenAI-compatible
 * Chat Completions endpoint at https://api.cline.bot/api/v1. The model IDs
 * use a `cline-pass/` prefix (e.g. `cline-pass/glm-5.2`).
 *
 * The underlying models overlap with the OpenCode Go catalog (same model
 * families from Z.ai, Moonshot, DeepSeek, MiniMax, MiMo, Qwen). Capabilities
 * (vision, reasoning, context length, thinking mode) are reused from the
 * OpenCode Go catalog entries for the equivalent underlying model.
 *
 * Key differences from OpenCode Go:
 * - Base URL: https://api.cline.bot/api/v1 (vs opencode.ai/zen/go/v1/)
 * - API mode: always "openai" (ClinePass has no Anthropic or Responses endpoints)
 * - Model ID sent to API: the full `cline-pass/...` slug
 * - No `extra` request body parameters (e.g. `reasoning_split` is Anthropic-only)
 * - No `/models` availability filter endpoint (model list is hardcoded)
 */

import type { LanguageModelChatInformation } from "vscode";
import type { BaseModelItem } from "./baseProvider";
import { buildCatalogModelInfo, resolveModelMeta } from "./catalogModels";
import { ensureModelsDevLoaded } from "./modelsDev";

/** Cline Pass API base URL. */
export const CLINE_PASS_BASE_URL = "https://api.cline.bot/api/v1";

/**
 * Mapping of ClinePass model IDs to their underlying OpenCode Go catalog model IDs.
 * Each ClinePass model shares the same underlying model as its OpenCode Go counterpart;
 * capabilities are reused from the catalog entry for the underlying ID.
 */
const CLINE_PASS_MODELS: Record<string, string> = {
    "cline-pass/glm-5.2": "glm-5.2",
    "cline-pass/glm-5.3": "glm-5.3",
    "cline-pass/glm-5.3-flash": "glm-5.3-flash",
    "cline-pass/kimi-k3": "kimi-k3",
    "cline-pass/kimi-k2.7-code": "kimi-k2.7-code",
    "cline-pass/kimi-k2.6": "kimi-k2.6",
    "cline-pass/deepseek-v4-pro": "deepseek-v4-pro",
    "cline-pass/deepseek-v4-flash": "deepseek-v4-flash",
    "cline-pass/mimo-v2.5": "mimo-v2.5",
    "cline-pass/mimo-v2.5-pro": "mimo-v2.5-pro",
    "cline-pass/minimax-m3": "minimax-m3",
    "cline-pass/qwen3.7-max": "qwen3.7-max",
    "cline-pass/qwen3.7-plus": "qwen3.7-plus",
};

/**
 * Get all ClinePass model IDs.
 */
export function getClinePassModelIds(): string[] {
    return Object.keys(CLINE_PASS_MODELS);
}

/**
 * Check whether a model ID belongs to the ClinePass provider.
 */
export function isClinePassModel(modelId: string): boolean {
    return modelId in CLINE_PASS_MODELS;
}

/**
 * Resolve the underlying OpenCode Go catalog model ID for a ClinePass model ID.
 */
export function resolveClinePassUnderlyingId(modelId: string): string | undefined {
    return CLINE_PASS_MODELS[modelId];
}

/**
 * Build a LanguageModelChatInformation entry for a ClinePass model.
 *
 * Capabilities (reasoning enum, vision, context limits, tool calling) are
 * reused from the OpenCode Go catalog entry for the underlying model.
 * The display fields (family, detail, tooltip) are overridden for ClinePass.
 */
export function buildClinePassModelInfo(modelId: string): LanguageModelChatInformation | undefined {
    const underlyingId = CLINE_PASS_MODELS[modelId];
    if (!underlyingId) {
        return undefined;
    }

    // Build from the OpenCode Go catalog to reuse capabilities and reasoning enum
    const info = buildCatalogModelInfo("opencode-go", underlyingId);

    // Override ClinePass-specific display fields
    return {
        ...info,
        id: modelId,
        family: "ClinePass",
        detail: "Cline Pass",
        tooltip: "Cline Pass",
    };
}

/**
 * Build the BaseModelItem request config for a ClinePass model.
 *
 * Capabilities are reused from the OpenCode Go catalog, but:
 * - `apiMode` is always "openai" (ClinePass is OpenAI-compatible only)
 * - `baseUrl` is always the Cline Pass endpoint
 * - `id` is the full `cline-pass/...` slug (as expected by the Cline API)
 * - No `extra` parameters (e.g. `reasoning_split` is Anthropic-only and
 *   does not apply to ClinePass's OpenAI Chat Completions endpoint)
 */
export async function getClinePassModelConfig(modelId: string): Promise<BaseModelItem | undefined> {
    const underlyingId = CLINE_PASS_MODELS[modelId];
    if (!underlyingId) {
        return undefined;
    }

    // Ensure the catalog is loaded so resolveModelMeta returns real metadata
    // rather than conservative defaults. Degrades gracefully if unavailable.
    await ensureModelsDevLoaded();

    const meta = resolveModelMeta("opencode-go", underlyingId);

    const config: BaseModelItem = {
        id: modelId,
        displayName: meta.displayName,
        baseUrl: CLINE_PASS_BASE_URL,
        apiMode: "openai",
        context_length: meta.contextLength,
        max_completion_tokens: meta.maxOutputTokens,
        vision: meta.vision,
        enable_thinking: true,
        include_reasoning_in_request: true,
        supportsTemperature: meta.supportsTemperature,
        thinkingMode: meta.thinkingMode,
    };

    // Only send an explicit effort when it is a real effort value
    // ("enabled"/"adaptive" are handled via the thinking flags instead).
    if (meta.defaultReasoningEffort && meta.defaultReasoningEffort !== "enabled" && meta.defaultReasoningEffort !== "adaptive") {
        config.reasoning_effort = meta.defaultReasoningEffort;
    }
    if (meta.thinkingBudget?.max !== undefined) {
        config.thinking_budget = meta.thinkingBudget.max;
    }

    return config;
}

/**
 * Get all ClinePass model entries for the model picker.
 */
export async function buildClinePassModelInfos(): Promise<LanguageModelChatInformation[]> {
    const ids = getClinePassModelIds();
    const infos: LanguageModelChatInformation[] = [];
    for (const id of ids) {
        const info = buildClinePassModelInfo(id);
        if (info) {
            infos.push(info);
        }
    }
    return infos;
}