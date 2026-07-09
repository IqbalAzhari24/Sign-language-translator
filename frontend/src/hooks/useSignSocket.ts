import { useEffect, useRef, useState, useCallback } from "react";

export type SignResult =
  | { status: "no_hand" | "tracking" | "unsure" }
  | { status: "recognized"; label: string; confidence: number };

export function useSignSocket(url: string) {
  const [connected, setConnected] = useState(false);
  const [result, setResult] = useState<SignResult>({ status: "no_hand" });
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryDelay = 500;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    function connect() {
      const socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        retryDelay = 500;
        setConnected(true);
      };
      socket.onmessage = (event) => {
        setResult(JSON.parse(event.data));
      };
      socket.onclose = () => {
        setConnected(false);
        setResult({ status: "no_hand" });
        if (!cancelled) {
          retryTimer = setTimeout(connect, retryDelay);
          retryDelay = Math.min(retryDelay * 2, 8000);
        }
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      socketRef.current?.close();
    };
  }, [url]);

  const sendFrame = useCallback((blob: Blob) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(blob);
    }
  }, []);

  return { connected, result, sendFrame };
}
