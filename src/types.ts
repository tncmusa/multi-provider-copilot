import type { BaseModelItem } from "./baseProvider";

/**
 * A single model entry for OpenCode Go.
 */
export interface OpenCodeGoModelItem extends BaseModelItem {
    object?: string;
    created?: number;
    owned_by: string;
    configId?: string;
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
    /**
     * Whether this model can be used for Git commit message generation.
     */
    useForCommitGeneration?: boolean;
    /** Additional fields may be present in provider-specific entries */
    [key: string]: unknown;
}

/**
 * Response from the models endpoint.
 */
export interface ModelsResponse {
    object: string;
    data: ModelItem[];
}

export interface ModelItem {
    id: string;
    object?: string;
    created?: number;
    owned_by?: string;
}

/**
 * A model preset for temperature and top_p configuration.
 */
export interface ModelPreset {
    id: string;
    label: string;
    temperature: number;
    top_p: number;
}

/**
 * Retry configuration.
 */
export interface RetryConfig {
    enabled: boolean;
    maxAttempts: number;
    intervalMs: number;
    backoffFactor: number;
    maxIntervalMs: number;
    statusCodes: number[];
}
