import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useSpeech } from "./useSpeech";
import type { SignResult } from "./useSignSocket";

beforeEach(() => {
  // @ts-expect-error test override
  window.speechSynthesis = { speak: vi.fn() };
  // @ts-expect-error test override
  global.SpeechSynthesisUtterance = vi.fn().mockImplementation(function (text: string) {
    return { text };
  });
});

describe("useSpeech", () => {
  it("speaks once per newly recognized label", () => {
    const { rerender } = renderHook(({ result }: { result: SignResult }) => useSpeech(result), {
      initialProps: { result: { status: "recognized", label: "A", confidence: 0.9 } },
    });

    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1);

    rerender({ result: { status: "recognized", label: "A", confidence: 0.95 } });
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(1);

    rerender({ result: { status: "recognized", label: "B", confidence: 0.9 } });
    expect(window.speechSynthesis.speak).toHaveBeenCalledTimes(2);
  });
});
