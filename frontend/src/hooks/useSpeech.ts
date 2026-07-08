import { useEffect, useRef } from "react";
import type { SignResult } from "./useSignSocket";

export function useSpeech(result: SignResult) {
  const lastSpokenRef = useRef<string | null>(null);

  useEffect(() => {
    if (result.status !== "recognized") {
      // Any non-recognized status (hand lost, still tracking, unsure)
      // re-arms the guard so the next recognized sign — even a repeat
      // of the same label — gets announced again.
      lastSpokenRef.current = null;
      return;
    }
    if (result.label === lastSpokenRef.current) return;
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    lastSpokenRef.current = result.label;
    const utterance = new SpeechSynthesisUtterance(result.label);
    window.speechSynthesis.speak(utterance);
  }, [result]);
}
