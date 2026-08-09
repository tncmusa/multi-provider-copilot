/**
 * NanoGPT Subscription model discovery and configuration.
 *
 * NanoGPT Subscription exposes models via an OpenAI-compatible Chat Completions
 * endpoint at https://nano-gpt.com/api/subscription/v1. The model IDs use a
 * `nanogpt/` prefix (e.g. `nanogpt/google/gemma-4-12b-it`).
 *
 * Model IDs follow the `provider/model` format (e.g. `google/gemma-4-12b-it`,
 * `openai/gpt-5.2`). Thinking/reasoning variants use a `:thinking` suffix
 * (e.g. `anthropic/claude-opus-4.6:thinking`).
 *
 * Capabilities (vision, reasoning, context length, thinking mode) are resolved
 * from the models.dev catalog when a matching entry exists, falling back to
 * conservative defaults for unmatched models.
 *
 * Key differences from OpenCode Go:
 * - Base URL: https://nano-gpt.com/api/subscription/v1 (vs opencode.ai/zen/go/v1/)
 * - API mode: always "openai" (NanoGPT is OpenAI-compatible only)
 * - Model ID sent to API: stripped of `nanogpt/` prefix
 * - Model list: fetched dynamically from GET /api/subscription/v1/models
 * - No `extra` request body parameters
 */

import type { LanguageModelChatInformation } from "vscode";
import type { BaseModelItem } from "./baseProvider";
import { buildCatalogModelInfo, resolveModelMeta, type ProviderId } from "./catalogModels";
import { ensureModelsDevLoaded, lookupModelDevEntry, type ModelsDevEntry } from "./modelsDev";
import { logger } from "./logger";

/** NanoGPT Subscription API base URL. */
export const NANOGPT_BASE_URL = "https://nano-gpt.com/api/subscription/v1";

/** Prefix used for all NanoGPT model IDs in the VS Code model picker. */
const NANOGPT_PREFIX = "nanogpt/";

/** Cache TTL for the model list (1 minute). */
const CACHE_TTL_MS = 60 * 1000;

// ── Module-level cache ──
let cachedModelIds: string[] | null = null;
let cacheTimestamp = 0;

/**
 * Strip the `nanogpt/` prefix from a model ID to get the API-level model ID.
 */
export function stripNanoGptPrefix(modelId: string): string {
    if (modelId.startsWith(NANOGPT_PREFIX)) {
        return modelId.slice(NANOGPT_PREFIX.length);
    }
    return modelId;
}

/**
 * Add the `nanogpt/` prefix to an API-level model ID.
 */
function addNanoGptPrefix(apiModelId: string): string {
    return NANOGPT_PREFIX + apiModelId;
}

/**
 * Check whether a model ID belongs to the NanoGPT provider.
 */
export function isNanoGptModel(modelId: string): boolean {
    return modelId.startsWith(NANOGPT_PREFIX);
}

/**
 * Fetch the list of subscription-included model IDs from the NanoGPT API.
 * Returns an empty array on failure (silent degradation).
 */
async function fetchNanoGptModelIds(): Promise<string[]> {
    const url = `${NANOGPT_BASE_URL}/models`;
    try {
        const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
        if (!response.ok) {
            throw new Error(`NanoGPT model list error: [${response.status}] ${response.statusText}`);
        }
        const body = (await response.json()) as { data?: Array<{ id: string }> };
        return (body.data ?? []).map((m) => m.id);
    } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
            logger.warn("nanogpt.models.fetch.timeout", { url });
        } else {
            logger.warn("nanogpt.models.fetch.failed", {
                url,
                error: err instanceof Error ? err.message : String(err),
            });
        }
        return [];
    }
}

/**
 * Get the list of NanoGPT subscription model IDs with 1-minute caching.
 */
async function getNanoGptModelIds(): Promise<string[]> {
    const now = Date.now();
    if (cachedModelIds !== null && now - cacheTimestamp < CACHE_TTL_MS) {
        return cachedModelIds;
    }

    const ids = await fetchNanoGptModelIds();
    if (ids.length > 0) {
        cachedModelIds = ids;
        cacheTimestamp = now;
    }
    // On failure, return stale cache if available, otherwise empty
    return ids.length > 0 ? ids : (cachedModelIds ?? []);
}

/**
 * Clear the cached model list (for forced refresh).
 */
export function clearNanoGptModelCache(): void {
    cachedModelIds = null;
    cacheTimestamp = 0;
}

/**
 * Try to find a matching models.dev catalog entry for a NanoGPT model ID.
 *
 * NanoGPT uses `provider/model` format (e.g. `google/gemma-4-12b-it`).
 * models.dev uses the same format for global entries. We try:
 * 1. Exact match in the global catalog
 * 2. Strip `:thinking` suffix and try again
 */
function findCatalogEntry(apiModelId: string): { entry: ModelsDevEntry | undefined; baseId: string } {
    // Try exact match first
    const exact = lookupModelDevEntry(apiModelId);
    if (exact) {
        return { entry: exact, baseId: apiModelId };
    }

    // Try without :thinking suffix
    if (apiModelId.endsWith(":thinking")) {
        const baseId = apiModelId.slice(0, -":thinking".length);
        const base = lookupModelDevEntry(baseId);
        if (base) {
            return { entry: base, baseId };
        }
    }

    return { entry: undefined, baseId: apiModelId };
}

/**
 * Conservative default capabilities for models not found in the catalog.
 */
const CONSERVATIVE_DEFAULTS = {
    contextLength: 128000,
    maxOutputTokens: 4096,
    vision: false,
    toolCalling: true,
    supportsTemperature: true,
};

/**
 * Build a LanguageModelChatInformation entry for a NanoGPT model.
 *
 * Capabilities are resolved from the models.dev catalog when a matching
 * entry exists, falling back to conservative defaults.
 */
function buildNanoGptModelInfo(apiModelId: string): LanguageModelChatInformation | undefined {
    const prefixedId = addNanoGptPrefix(apiModelId);
    const { entry, baseId } = findCatalogEntry(apiModelId);

    if (entry) {
        // Found in catalog — reuse the full capability resolution
        // Determine which provider to use for catalog lookup
        const info = buildCatalogModelInfo("opencode-go", baseId);
        if (info) {
            return {
                ...info,
                id: prefixedId,
                family: "NanoGPT",
                detail: "NanoGPT",
                tooltip: "NanoGPT Subscription",
            };
        }
    }

    // Not in catalog — build with conservative defaults
    const isThinking = apiModelId.endsWith(":thinking");
    const displayBase = apiModelId.replace(/:thinking$/, "");
    const displayName = displayBase
        .split("/")
        .pop()
        ?.replace(/-/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase()) ?? apiModelId;

    return {
        id: prefixedId,
        name: `${displayName}${isThinking ? " (Thinking)" : ""}`,
        family: "NanoGPT",
        version: "1.0",
        detail: "NanoGPT",
        tooltip: "NanoGPT Subscription",
        maxInputTokens: CONSERVATIVE_DEFAULTS.contextLength,
        maxOutputTokens: CONSERVATIVE_DEFAULTS.maxOutputTokens,
        capabilities: {
            toolCalling: CONSERVATIVE_DEFAULTS.toolCalling,
            vision: CONSERVATIVE_DEFAULTS.vision,
        },
        reasoningEffort: {
            enumValues: isThinking ? ["disabled", "enabled"] : ["disabled"],
            enumItemLabels: isThinking ? ["Disabled", "Thinking"] : ["Disabled"],
            enumDescriptions: isThinking
                ? ["Do not enable thinking", "Enable thinking"]
                : ["Do not enable thinking"],
            defaultEffort: isThinking ? "enabled" : "disabled",
        },
    } as LanguageModelChatInformation;
}

/**
 * Build the BaseModelItem request config for a NanoGPT model.
 *
 * Capabilities are resolved from the models.dev catalog when available,
 * falling back to conservative defaults.
 */
export async function getNanoGptModelConfig(modelId: string): Promise<BaseModelItem | undefined> {
    if (!isNanoGptModel(modelId)) {
        return undefined;
    }

    const apiModelId = stripNanoGptPrefix(modelId);
    const { entry, baseId } = findCatalogEntry(apiModelId);

    await ensureModelsDevLoaded();

    if (entry) {
        // Use catalog metadata via the merge chain
        const meta = resolveModelMeta("opencode-go", baseId);

        const isThinking = apiModelId.endsWith(":thinking");
        const config: BaseModelItem = {
            id: apiModelId, // API-level ID (without nanogpt/ prefix)
            displayName: meta.displayName,
            baseUrl: NANOGPT_BASE_URL,
            apiMode: "openai",
            context_length: meta.contextLength,
            max_completion_tokens: meta.maxOutputTokens,
            vision: meta.vision,
            enable_thinking: isThinking,
            include_reasoning_in_request: isThinking,
            supportsTemperature: meta.supportsTemperature,
            thinkingMode: isThinking ? meta.thinkingMode : "switchable",
        };

        if (isThinking && meta.defaultReasoningEffort && meta.defaultReasoningEffort !== "enabled" && meta.defaultReasoningEffort !== "adaptive") {
            config.reasoning_effort = meta.defaultReasoningEffort;
        }
        if (meta.thinkingBudget?.max !== undefined) {
            config.thinking_budget = meta.thinkingBudget.max;
        }

        return config;
    }

    // Conservative defaults for unmatched models
    const isThinking = apiModelId.endsWith(":thinking");
    const config: BaseModelItem = {
        id: apiModelId,
        displayName: apiModelId,
        baseUrl: NANOGPT_BASE_URL,
        apiMode: "openai",
        context_length: CONSERVATIVE_DEFAULTS.contextLength,
        max_completion_tokens: CONSERVATIVE_DEFAULTS.maxOutputTokens,
        vision: CONSERVATIVE_DEFAULTS.vision,
        enable_thinking: isThinking,
        include_reasoning_in_request: isThinking,
        supportsTemperature: CONSERVATIVE_DEFAULTS.supportsTemperature,
        thinkingMode: isThinking ? "switchable" : "switchable",
    };

    return config;
}

/**
 * Get all NanoGPT model entries for the model picker.
 * Fetches the model list from the API with 1-minute caching.
 */
export async function buildNanoGptModelInfos(): Promise<LanguageModelChatInformation[]> {
    const apiModelIds = await getNanoGptModelIds();
    if (apiModelIds.length === 0) {
        logger.warn("nanogpt.models.empty", {});
        return [];
    }

    const infos: LanguageModelChatInformation[] = [];
    for (const apiId of apiModelIds) {
        const info = buildNanoGptModelInfo(apiId);
        if (info) {
            infos.push(info);
        }
    }

    logger.info("nanogpt.models.discovery", {
        action: "loaded",
        count: infos.length,
        ids: infos.map((i) => i.id).join(", "),
    });

    return infos;
}
