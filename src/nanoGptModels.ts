/**
 * NanoGPT Subscription model discovery and configuration.
 *
 * NanoGPT Subscription exposes models via an OpenAI-compatible Chat Completions
 * endpoint at https://nano-gpt.com/api/subscription/v1. The model IDs use a
 * `nanogpt/` prefix (e.g. `nanogpt/google/gemma-4-12b-it`).
 *
 * Model IDs follow the `provider/model` format (e.g. `google/gemma-4-12b-it`,
 * `openai/gpt-5.2`). Thinking/reasoning variants use a `:thinking` suffix
 * (e.g. `deepseek/deepseek-v4-pro:thinking`).
 *
 * ALL model metadata (capabilities, context length, reasoning efforts, vision,
 * tool calling) comes exclusively from NanoGPT's own API:
 *   GET /api/subscription/v1/models?detailed=true
 *
 * This is the single source of truth — no models.dev catalog fallback.
 *
 * Key differences from OpenCode Go:
 * - Base URL: https://nano-gpt.com/api/subscription/v1 (vs opencode.ai/zen/go/v1/)
 * - API mode: always "openai" (NanoGPT is OpenAI-compatible only)
 * - Model ID sent to API: stripped of `nanogpt/` prefix
 * - Model list + capabilities: fetched from NanoGPT API directly
 * - No `extra` request body parameters
 */

import type { LanguageModelChatInformation } from "vscode";
import type { BaseModelItem } from "./baseProvider";
import { logger } from "./logger";

/** NanoGPT Subscription API base URL. */
export const NANOGPT_BASE_URL = "https://nano-gpt.com/api/subscription/v1";

/** Prefix used for all NanoGPT model IDs in the VS Code model picker. */
const NANOGPT_PREFIX = "nanogpt/";

/** Cache TTL for the model list (1 minute). */
const CACHE_TTL_MS = 60 * 1000;

// ── Types for the NanoGPT detailed models API ──

/** Capabilities block from the NanoGPT detailed models response. */
interface NanoGptCapabilities {
    vision: boolean;
    reasoning: boolean;
    tool_calling: boolean;
    parallel_tool_calls: boolean;
    structured_output: boolean;
    pdf_upload: boolean;
}

/** A single model entry from NanoGPT's ?detailed=true response. */
interface NanoGptModelEntry {
    id: string;
    object: string;
    created: number;
    owned_by: string;
    name?: string;
    description?: string;
    context_length: number;
    max_output_tokens: number | null;
    capabilities: NanoGptCapabilities;
    /** Optional reasoning effort levels (e.g. ["none","low","medium","high","xhigh"]) */
    reasoning_efforts?: string[];
    pricing?: {
        prompt: number;
        completion: number;
        currency: string;
        unit: string;
    };
}

/** Top-level response from NanoGPT's /models?detailed=true endpoint. */
interface NanoGptModelsResponse {
    object: string;
    data: NanoGptModelEntry[];
}

// ── Module-level cache ──
let cachedEntries: NanoGptModelEntry[] | null = null;
let cacheTimestamp = 0;

// ── Model ID helpers ──

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

// ── Model ID filtering ──

/** Model IDs to exclude from the picker (internal/helper models). */
const EXCLUDED_MODEL_IDS = new Set([
    "nano-gpt-help",
    "auto-model",
    "auto-model-basic",
    "auto-model-standard",
    "auto-model-premium",
]);

/**
 * Fetch the detailed model list from the NanoGPT subscription API.
 * Returns an empty array on failure (silent degradation).
 */
async function fetchNanoGptModels(): Promise<NanoGptModelEntry[]> {
    const url = `${NANOGPT_BASE_URL}/models?detailed=true`;
    try {
        const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
        if (!response.ok) {
            throw new Error(`NanoGPT model list error: [${response.status}] ${response.statusText}`);
        }
        const body = (await response.json()) as NanoGptModelsResponse;
        const entries = (body.data ?? []).filter((m) => !EXCLUDED_MODEL_IDS.has(m.id));
        logger.info("nanogpt.models.fetch", {
            url,
            totalCount: body.data?.length ?? 0,
            filteredCount: entries.length,
        });
        return entries;
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
 * Get the detailed NanoGPT model list with 1-minute caching.
 */
async function getNanoGptModels(): Promise<NanoGptModelEntry[]> {
    const now = Date.now();
    if (cachedEntries !== null && now - cacheTimestamp < CACHE_TTL_MS) {
        return cachedEntries;
    }

    const entries = await fetchNanoGptModels();
    if (entries.length > 0) {
        cachedEntries = entries;
        cacheTimestamp = now;
    }
    return entries.length > 0 ? entries : (cachedEntries ?? []);
}

/**
 * Clear the cached model list (for forced refresh).
 */
export function clearNanoGptModelCache(): void {
    cachedEntries = null;
    cacheTimestamp = 0;
}

// ── Reasoning effort label helpers ──

/** Map reasoning effort values to display labels. */
function reasoningEffortLabel(effort: string): string {
    switch (effort) {
        case "none": return "Disabled";
        case "minimal": return "Minimal";
        case "low": return "Low";
        case "medium": return "Medium";
        case "high": return "High";
        case "xhigh": return "Extra High";
        case "max": return "Maximum";
        default: return effort.charAt(0).toUpperCase() + effort.slice(1);
    }
}

/** Map reasoning effort values to descriptions. */
function reasoningEffortDescription(effort: string): string {
    switch (effort) {
        case "none": return "Do not enable thinking";
        case "minimal": return "Minimal reasoning depth";
        case "low": return "Reduce thinking, faster response";
        case "medium": return "Balance thinking and speed";
        case "high": return "Deeper thinking, slower response";
        case "xhigh": return "Very deep thinking, slower response";
        case "max": return "Maximum thinking depth, slowest response";
        default: return effort;
    }
}

/**
 * Build the reasoning effort enum for a model from its NanoGPT capabilities.
 *
 * Rules:
 * - If the model has `reasoning_efforts` array, use those values directly.
 * - If the model has `capabilities.reasoning: true` but no explicit efforts,
 *   provide a simple "disabled"/"enabled" toggle.
 * - If the model has `capabilities.reasoning: false`, only "disabled" is available.
 * - `:thinking` suffix models default to thinking enabled.
 */
function buildReasoningEnum(entry: NanoGptModelEntry): {
    enumValues: string[];
    enumItemLabels: string[];
    enumDescriptions: string[];
    defaultEffort: string;
} {
    const isThinkingSuffix = entry.id.endsWith(":thinking");
    const hasReasoning = entry.capabilities.reasoning;

    // If the API provides explicit reasoning_efforts, use them
    if (entry.reasoning_efforts && entry.reasoning_efforts.length > 0) {
        const efforts = entry.reasoning_efforts;
        const hasNone = efforts.includes("none");

        // Build enum: "disabled" maps to "none" if present, otherwise add "disabled" at front
        const enumValues = hasNone
            ? efforts.map((e) => e === "none" ? "disabled" : e)
            : ["disabled", ...efforts];

        const defaultEffort = isThinkingSuffix
            ? (efforts.find((e) => e !== "none") ?? efforts[efforts.length - 1])
            : "disabled";

        return {
            enumValues,
            enumItemLabels: enumValues.map(reasoningEffortLabel),
            enumDescriptions: enumValues.map(reasoningEffortDescription),
            defaultEffort,
        };
    }

    // No explicit efforts — use simple toggle based on reasoning capability
    if (hasReasoning) {
        return {
            enumValues: ["disabled", "enabled"],
            enumItemLabels: ["Disabled", "Thinking"],
            enumDescriptions: ["Do not enable thinking", "Enable thinking"],
            defaultEffort: isThinkingSuffix ? "enabled" : "disabled",
        };
    }

    // No reasoning at all
    return {
        enumValues: ["disabled"],
        enumItemLabels: ["Disabled"],
        enumDescriptions: ["Do not enable thinking"],
        defaultEffort: "disabled",
    };
}

// ── Model info / config builders ──

/**
 * Build a LanguageModelChatInformation entry from a NanoGPT model entry.
 */
function buildNanoGptModelInfo(entry: NanoGptModelEntry): LanguageModelChatInformation {
    const prefixedId = addNanoGptPrefix(entry.id);
    const reasoningEnum = buildReasoningEnum(entry);
    const displayName = entry.name ?? entry.id;

    return {
        id: prefixedId,
        name: displayName,
        family: "NanoGPT",
        version: "1.0",
        detail: "NanoGPT",
        tooltip: "NanoGPT Subscription",
        maxInputTokens: entry.context_length || 128000,
        maxOutputTokens: entry.max_output_tokens || 4096,
        isUserSelectable: true,
        capabilities: {
            toolCalling: entry.capabilities.tool_calling,
            // Always declare imageInput=true so VS Code passes image data through.
            // Non-vision models handle images via the ask_image tool proxy internally.
            imageInput: true,
        },
        configurationSchema: {
            properties: {
                reasoningEffort: {
                    type: "string",
                    title: "Reasoning Effort",
                    enum: reasoningEnum.enumValues,
                    enumItemLabels: reasoningEnum.enumItemLabels,
                    enumDescriptions: reasoningEnum.enumDescriptions,
                    default: reasoningEnum.defaultEffort,
                    group: "navigation",
                },
            },
        },
    } satisfies LanguageModelChatInformation;
}

/**
 * Build the BaseModelItem request config from a NanoGPT model entry.
 */
function buildNanoGptModelConfig(entry: NanoGptModelEntry): BaseModelItem {
    const isThinkingSuffix = entry.id.endsWith(":thinking");
    const hasReasoning = entry.capabilities.reasoning;

    // Determine thinking mode:
    // - "switchable" if the model has reasoning_efforts including "none"
    //   (user can turn it off via the picker)
    // - "always" only if reasoning is on AND there's no "none" option
    //   (e.g. a :thinking model that cannot be disabled)
    let thinkingMode: BaseModelItem["thinkingMode"] = "switchable";
    const canDisable = entry.reasoning_efforts?.includes("none") ?? false;
    if (hasReasoning && isThinkingSuffix && !canDisable) {
        thinkingMode = "always";
    }

    // Determine default reasoning effort:
    // - :thinking models default to the first non-"none" effort
    // - non-:thinking models default to "none" (disabled)
    let defaultEffort: string | undefined;
    if (hasReasoning && entry.reasoning_efforts && entry.reasoning_efforts.length > 0) {
        if (isThinkingSuffix) {
            const nonNone = entry.reasoning_efforts.find((e) => e !== "none");
            if (nonNone) {
                defaultEffort = nonNone;
            }
        } else {
            // Non-:thinking model with reasoning capability: default to "none"
            defaultEffort = "none";
        }
    }

    const config: BaseModelItem = {
        id: entry.id, // API-level ID (without nanogpt/ prefix)
        displayName: entry.name ?? entry.id,
        baseUrl: NANOGPT_BASE_URL,
        apiMode: "openai",
        context_length: entry.context_length || 128000,
        max_completion_tokens: entry.max_output_tokens || 4096,
        vision: entry.capabilities.vision,
        enable_thinking: isThinkingSuffix,
        include_reasoning_in_request: isThinkingSuffix,
        supportsTemperature: true,
        thinkingMode,
    };

    if (defaultEffort) {
        config.reasoning_effort = defaultEffort;
    }

    return config;
}

/**
 * Build the BaseModelItem request config for a NanoGPT model by its VS Code model ID.
 */
export async function getNanoGptModelConfig(modelId: string): Promise<BaseModelItem | undefined> {
    if (!isNanoGptModel(modelId)) {
        return undefined;
    }

    const apiModelId = stripNanoGptPrefix(modelId);
    const entries = await getNanoGptModels();
    const entry = entries.find((e) => e.id === apiModelId);
    if (!entry) {
        logger.warn("nanogpt.model.not-found", { modelId, apiModelId });
        return undefined;
    }

    return buildNanoGptModelConfig(entry);
}

/**
 * Get all NanoGPT model entries for the model picker.
 * Fetches the detailed model list from the NanoGPT API with 1-minute caching.
 */
export async function buildNanoGptModelInfos(): Promise<LanguageModelChatInformation[]> {
    const entries = await getNanoGptModels();
    if (entries.length === 0) {
        logger.warn("nanogpt.models.empty", {});
        return [];
    }

    const infos = entries.map(buildNanoGptModelInfo);

    logger.info("nanogpt.models.discovery", {
        action: "loaded",
        count: infos.length,
    });

    return infos;
}
