import { useEffect, useRef } from "react";
import type { SignResult } from "./useSignSocket";

export function useSpeech(result: SignResult) {
  const lastSpokenRef = useRef<string | null>(null);

  useEffect(() => {
    if (result.status !== "recognized") return;
    if (result.label === lastSpokenRef.current) return;
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    lastSpokenRef.current = result.label;
    const utterance = new SpeechSynthesisUtterance(result.label);
    window.speechSynthesis.speak(utterance);
  }, [result]);
}
