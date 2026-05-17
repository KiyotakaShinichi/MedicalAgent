import { describe, it, expect } from "vitest";
import {
  normaliseMessages,
  parseSavedActions,
  parseCitations,
} from "../../src/components/ui/chat-utils";

describe("normaliseMessages", () => {
  it("returns [] for non-array input (undefined, null, object, string)", () => {
    expect(normaliseMessages(undefined)).toEqual([]);
    expect(normaliseMessages(null)).toEqual([]);
    expect(normaliseMessages({})).toEqual([]);
    expect(normaliseMessages("hello")).toEqual([]);
  });

  it("filters out entries without a recognised role", () => {
    const input = [
      { role: "user", message: "hi" },
      { role: "system", message: "skip me" },
      { message: "no role" },
      null,
      "not an object",
      { role: "assistant", message: "reply" },
    ];
    const out = normaliseMessages(input);
    expect(out).toHaveLength(2);
    expect(out.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(out.map((m) => m.content)).toEqual(["hi", "reply"]);
  });

  it("accepts either `message` or `content` field for the body", () => {
    const out = normaliseMessages([
      { role: "user",      message: "from message field" },
      { role: "assistant", content: "from content field" },
      { role: "user" }, // missing body — should become empty string, not crash
    ]);
    expect(out.map((m) => m.content)).toEqual([
      "from message field",
      "from content field",
      "",
    ]);
  });

  it("synthesises stable ids when the message has none", () => {
    const out = normaliseMessages([
      { role: "user", message: "a" },
      { role: "assistant", message: "b" },
    ]);
    expect(out[0].id).toBe("msg_user_0");
    expect(out[1].id).toBe("msg_assistant_1");
  });

  it("preserves a server-supplied id (string or number)", () => {
    const out = normaliseMessages([
      { id: "srv-42", role: "user",      message: "a" },
      { id: 99,        role: "assistant", message: "b" },
    ]);
    expect(out[0].id).toBe("srv-42");
    expect(out[1].id).toBe("99");
  });
});

describe("parseSavedActions", () => {
  it("returns [] for missing / empty input", () => {
    expect(parseSavedActions(undefined)).toEqual([]);
    expect(parseSavedActions(null)).toEqual([]);
    expect(parseSavedActions("")).toEqual([]);
  });

  it("parses a JSON-string array", () => {
    const out = parseSavedActions('[{"type":"saved_symptom","data":{}}]');
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe("saved_symptom");
  });

  it("unwraps the {saved_actions: [...]} object shape", () => {
    const out = parseSavedActions({
      saved_actions: [{ type: "saved_labs" }],
      tool_plan: "ignored",
    });
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe("saved_labs");
  });

  it("drops items without a string `type` field", () => {
    const out = parseSavedActions([
      { type: "saved_symptom" },
      { type: 42 },
      null,
      { data: "no type" },
    ]);
    expect(out).toHaveLength(1);
    expect(out[0].type).toBe("saved_symptom");
  });

  it("returns [] for unparseable JSON instead of throwing", () => {
    expect(parseSavedActions("{not json")).toEqual([]);
  });
});

describe("parseCitations", () => {
  it("returns [] for non-array input", () => {
    expect(parseCitations(undefined)).toEqual([]);
    expect(parseCitations({ id: "x" })).toEqual([]);
  });

  it("keeps strings verbatim and pulls .id from objects", () => {
    const out = parseCitations([
      "kb_source_1",
      { id: "kb_source_2" },
      { id: 42 },
      {},
      "",
    ]);
    expect(out).toEqual(["kb_source_1", "kb_source_2", "42"]);
  });
});
