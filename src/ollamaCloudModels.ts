/**
 * Ollama Cloud model discovery and configuration.
 *
 * Ollama Cloud exposes open-weight models via an OpenAI-compatible
 * Chat Completions endpoint at https://ollama.com/api. The model IDs
 * use an `ollama-cloud/` prefix (e.g. `ollama-cloud/gpt-oss:120b`).
 *
 * The underlying models overlap with the OpenCode Go catalog for many
 * models (GLM, DeepSeek, Kimi, MiniMax, Qwen). Capabilities (vision,
 * reasoning, context length, thinking mode) are reused from the
 * OpenCode Go catalog entries for the equivalent underlying model.
 * For models unique to Ollama Cloud (GPT-OSS, Gemma, Nemotron, Mistral),
 * capabilities are resolved from the global models.dev catalog.
 *
 * Key differences from OpenCode Go:
 * - Base URL: https://ollama.com/api (vs opencode.ai/zen/go/v1/)
 * - API mode: always "openai" (Ollama Cloud is OpenAI-compatible only)
 * - Model ID sent to API: stripped of `ollama-cloud/` prefix
 * - No `extra` request body parameters
 * - No `/models` availability filter endpoint (model list is hardcoded)
 */

import type { LanguageModelChatInformation } from "vscode";
import type { BaseModelItem } from "./baseProvider";
import { buildCatalogModelInfo, resolveModelMeta } from "./catalogModels";
import { ensureModelsDevLoaded } from "./modelsDev";

/** Ollama Cloud API base URL. */
export const OLLAMA_CLOUD_BASE_URL = "https://ollama.com/api";

/**
 * Mapping of Ollama Cloud model IDs to their capability source.
 *
 * Each entry maps an `ollama-cloud/` prefixed ID to:
 *   - provider:  the models.dev provider to resolve capabilities from
 *   - sourceId:  the model ID within that provider (or global catalog)
 *
 * For models that overlap with OpenCode Go, capabilities are reused from
 * the "opencode-go" provider. For Ollama-unique models, the global
 * models.dev catalog is used as the capability source.
 */
const OLLAMA_CLOUD_MODELS: Record<string, { provider: "opencode-go"; sourceId: string }> = {
    // ── Models overlapping with OpenCode Go ──
    "ollama-cloud/deepseek-v4-pro":    { provider: "opencode-go", sourceId: "deepseek-v4-pro" },
    "ollama-cloud/deepseek-v4-flash":  { provider: "opencode-go", sourceId: "deepseek-v4-flash" },
    "ollama-cloud/glm-5.2":            { provider: "opencode-go", sourceId: "glm-5.2" },
    "ollama-cloud/glm-5.1":            { provider: "opencode-go", sourceId: "glm-5.1" },
    "ollama-cloud/kimi-k3":            { provider: "opencode-go", sourceId: "kimi-k3" },
    "ollama-cloud/kimi-k2.7-code":     { provider: "opencode-go", sourceId: "kimi-k2.7-code" },
    "ollama-cloud/kimi-k2.6":          { provider: "opencode-go", sourceId: "kimi-k2.6" },
    "ollama-cloud/minimax-m3":         { provider: "opencode-go", sourceId: "minimax-m3" },
    "ollama-cloud/minimax-m2.7":       { provider: "opencode-go", sourceId: "minimax-m2.7" },
    "ollama-cloud/qwen3.5":            { provider: "opencode-go", sourceId: "qwen3.5-plus" },
    "ollama-cloud/nemotron-3-super":   { provider: "opencode-go", sourceId: "nemotron-3-super" },

    // ── Ollama Cloud exclusives (resolved from global models.dev catalog) ──
    // These models exist in the global catalog but not in the opencode-go provider.
    // resolveModelMeta falls back to global lookup when the provider entry is absent.
    "ollama-cloud/gpt-oss:120b":       { provider: "opencode-go", sourceId: "gpt-oss" },
    "ollama-cloud/gemma4:31b":         { provider: "opencode-go", sourceId: "gemma4" },
    "ollama-cloud/nemotron-3-ultra":   { provider: "opencode-go", sourceId: "nemotron-3-ultra" },
    "ollama-cloud/mistral-large-3":    { provider: "opencode-go", sourceId: "mistral-large-3" },
};

/**
 * Strip the `ollama-cloud/` prefix to get the API model ID sent to Ollama Cloud.
 */
export function ollamaCloudApiModelId(modelId: string): string {
    return modelId.startsWith("ollama-cloud/") ? modelId.slice("ollama-cloud/".length) : modelId;
}

/**
 * Get all Ollama Cloud model IDs.
 */
export function getOllamaCloudModelIds(): string[] {
    return Object.keys(OLLAMA_CLOUD_MODELS);
}

/**
 * Check whether a model ID belongs to the Ollama Cloud provider.
 */
export function isOllamaCloudModel(modelId: string): boolean {
    return modelId in OLLAMA_CLOUD_MODELS;
}

/**
 * Build a LanguageModelChatInformation entry for an Ollama Cloud model.
 *
 * Capabilities (reasoning enum, vision, context limits, tool calling) are
 * reused from the OpenCode Go catalog entry for the underlying model.
 * The display fields (family, detail, tooltip) are overridden for Ollama Cloud.
 */
export function buildOllamaCloudModelInfo(modelId: string): LanguageModelChatInformation | undefined {
    const mapping = OLLAMA_CLOUD_MODELS[modelId];
    if (!mapping) {
        return undefined;
    }

    // Build from the catalog to reuse capabilities and reasoning enum
    const info = buildCatalogModelInfo(mapping.provider, mapping.sourceId);

    // Override Ollama Cloud-specific display fields
    return {
        ...info,
        id: modelId,
        family: "OllamaCloud",
        detail: "Ollama Cloud",
        tooltip: "Ollama Cloud",
    };
}

/**
 * Build the BaseModelItem request config for an Ollama Cloud model.
 *
 * Capabilities are reused from the catalog, but:
 * - `apiMode` is always "openai" (Ollama Cloud is OpenAI-compatible only)
 * - `baseUrl` is always the Ollama Cloud endpoint
 * - `id` is the API model ID (without `ollama-cloud/` prefix)
 * - No `extra` parameters
 */
export async function getOllamaCloudModelConfig(modelId: string): Promise<BaseModelItem | undefined> {
    const mapping = OLLAMA_CLOUD_MODELS[modelId];
    if (!mapping) {
        return undefined;
    }

    // Ensure the catalog is loaded so resolveModelMeta returns real metadata
    // rather than conservative defaults. Degrades gracefully if unavailable.
    await ensureModelsDevLoaded();

    const meta = resolveModelMeta(mapping.provider, mapping.sourceId);

    const config: BaseModelItem = {
        id: ollamaCloudApiModelId(modelId),
        displayName: meta.displayName,
        baseUrl: OLLAMA_CLOUD_BASE_URL,
        apiMode: "openai",
        context_length: meta.contextLength,
        max_completion_tokens: meta.maxOutputTokens,
        vision: meta.vision,
        enable_thinking: true,
        include_reasoning_in_request: true,
        supportsTemperature: meta.supportsTemperature,
        thinkingMode: meta.thinkingMode,
    };

    return config;
}

/**
 * Build all Ollama Cloud model LanguageModelChatInformation entries.
 */
export async function buildOllamaCloudModelInfos(): Promise<LanguageModelChatInformation[]> {
    const infos: LanguageModelChatInformation[] = [];
    for (const modelId of getOllamaCloudModelIds()) {
        const info = buildOllamaCloudModelInfo(modelId);
        if (info) {
            infos.push(info);
        }
    }
    return infos;
}
