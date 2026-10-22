// Groq chat models (the main pipeline model and the smaller utility model)
// and reading the text out of their replies.

import { ChatGroq } from "@langchain/groq";
import { BaseMessage } from "@langchain/core/messages";
import { requireEnv, runtimeConfig } from "./config";

const DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b";
// Smaller, faster model for bulk utility work (context headers at upload)
const DEFAULT_UTILITY_MODEL = "openai/gpt-oss-20b";

export function messageText(content: BaseMessage["content"]): string {
  const text =
    typeof content === "string"
      ? content
      : content.map((part: any) => ("text" in part ? part.text : "")).join("");
  // Reasoning models such as Qwen3 may put their thinking inline
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

export function getUtilityModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: process.env.GROQ_UTILITY_MODEL || DEFAULT_UTILITY_MODEL,
    temperature: 0,
    maxTokens: 2048,
    maxRetries: runtimeConfig.maxRetries,
    cache: runtimeConfig.llmCache,
  });
}

export function getChatModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: process.env.GROQ_MODEL || DEFAULT_CHAT_MODEL,
    temperature: 0.1,
    // gpt-oss is a reasoning model: its hidden reasoning tokens count against this limit
    maxTokens: 4096,
    maxRetries: runtimeConfig.maxRetries,
    cache: runtimeConfig.llmCache,
  });
}
