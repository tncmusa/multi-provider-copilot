/** OpenAI Responses API input and output shapes used by the provider. */
export interface ResponsesInputText { type: "input_text"; text: string }
export interface ResponsesInputImage { type: "input_image"; image_url: string; detail?: "auto" | "low" | "high" }
export type ResponsesInputContent = ResponsesInputText | ResponsesInputImage;

export interface ResponsesInputMessage {
    role: "user" | "assistant";
    content: ResponsesInputContent[];
}

export interface ResponsesFunctionCall {
    type: "function_call";
    call_id?: string;
    name: string;
    arguments: string;
}

export interface ResponsesFunctionCallOutput {
    type: "function_call_output";
    call_id: string;
    output: string;
}

export type ResponsesInputItem = ResponsesInputMessage | ResponsesFunctionCall | ResponsesFunctionCallOutput;

export interface ResponsesRequestBody {
    model: string;
    input: ResponsesInputItem[];
    instructions?: string;
    stream?: boolean;
    max_output_tokens?: number;
    temperature?: number;
    top_p?: number;
    tools?: Array<Record<string, unknown>>;
    tool_choice?: unknown;
    reasoning?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface ResponsesResponse {
    output_text?: string;
    output?: Array<{ type?: string; content?: Array<{ type?: string; text?: string }>; call_id?: string; name?: string; arguments?: string }>;
    usage?: { input_tokens?: number; output_tokens?: number; input_tokens_details?: { cached_tokens?: number } };
}
