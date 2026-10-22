// Follow-up questions: caps the conversation history and condenses a
// follow-up into a standalone question before retrieval.

import { ChatGroq } from "@langchain/groq";
import { HumanMessage } from "@langchain/core/messages";
import { handleStageFailure } from "./config";
import { getUtilityModel, messageText } from "./llm";
import type { ChatTurn } from "./types";

// Conversation history used to resolve follow-up questions. Enforced here as
// well as in the client, since the request body can't be trusted to be small.
const MAX_HISTORY_TURNS = 6; // last 3 question/answer pairs
const MAX_HISTORY_TURN_CHARS = 1000; // long answers are cut; their opening carries the gist

// Keeps only the most recent, well-formed turns and cuts long ones, so a
// follow-up costs a bounded number of tokens however long the chat gets.
export function capHistory(history: unknown): ChatTurn[] {
  if (!Array.isArray(history)) return [];
  return history
    .filter(
      (turn): turn is ChatTurn =>
        (turn?.role === "user" || turn?.role === "assistant") &&
        typeof turn?.content === "string" &&
        turn.content.trim().length > 0
    )
    .slice(-MAX_HISTORY_TURNS)
    .map(({ role, content }) => {
      const text = content.replace(/\s+/g, " ").trim();
      return {
        role,
        content:
          text.length > MAX_HISTORY_TURN_CHARS ? `${text.slice(0, MAX_HISTORY_TURN_CHARS)}…` : text,
      };
    });
}

// Follow-ups like "what about sales staff?" can't be searched on their own:
// neither the embedding nor BM25 knows what "what about" refers to. One LLM
// call rewrites the follow-up into a standalone question using the
// conversation, and everything downstream (retrieval, generation, grading)
// uses that. The history is only used here, to resolve references; it is
// never shown to the answer generator, so earlier answers can't become a
// source of facts.
export async function condenseQuestion(
  question: string,
  history: ChatTurn[],
  llm?: ChatGroq
): Promise<string> {
  const turns = capHistory(history);
  if (turns.length === 0) return question;
  // The small model is enough for a rewrite and keeps follow-ups fast
  const model = llm || getUtilityModel();

  const conversation = turns
    .map((turn) => `${turn.role === "user" ? "User" : "Assistant"}: ${turn.content}`)
    .join("\n");

  const prompt = `Given a conversation and a follow-up question, rewrite the follow-up as a standalone question that can be understood without the conversation.

- Replace pronouns and vague references ("it", "they", "that policy", "what about X?") with what they refer to in the conversation.
- Otherwise keep the user's intent and wording. Do not answer the question or add facts.
- If the follow-up is already standalone or changes topic, return it unchanged.

<conversation>
${conversation}
</conversation>

Follow-up question: "${question}"

Output only the standalone question.`;

  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const rewritten = messageText(response.content)
      .replace(/^\s*standalone question:\s*/i, "")
      .replace(/^["'“]+|["'”]+$/g, "")
      .replace(/\s+/g, " ")
      .trim();
    // An empty or rambling reply is worse than the raw follow-up
    if (!rewritten || rewritten.length > Math.max(300, question.length * 4)) return question;
    return rewritten;
  } catch (error) {
    handleStageFailure("Question condensing", error);
    return question;
  }
}
