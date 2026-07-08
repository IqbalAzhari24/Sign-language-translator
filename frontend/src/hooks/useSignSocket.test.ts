import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { useSignSocket } from "./useSignSocket";

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: unknown[] = [];
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    setTimeout(() => this.onopen?.(), 0);
  }

  send(data: unknown) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  // @ts-expect-error test override of global WebSocket
  global.WebSocket = FakeWebSocket;
});

describe("useSignSocket", () => {
  it("connects and updates result on incoming message", async () => {
    const { result } = renderHook(() => useSignSocket("ws://test"));

    await waitFor(() => expect(result.current.connected).toBe(true));

    const socket = FakeWebSocket.instances[0];
    act(() => {
      socket.onmessage?.({
        data: JSON.stringify({ status: "recognized", label: "A", confidence: 0.9 }),
      });
    });

    expect(result.current.result).toEqual({ status: "recognized", label: "A", confidence: 0.9 });
  });

  it("sends frames only while connected", async () => {
    const { result } = renderHook(() => useSignSocket("ws://test"));
    await waitFor(() => expect(result.current.connected).toBe(true));

    const blob = new Blob(["frame"]);
    act(() => {
      result.current.sendFrame(blob);
    });

    expect(FakeWebSocket.instances[0].sent).toContain(blob);
  });
});
