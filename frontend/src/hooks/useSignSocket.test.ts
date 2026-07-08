import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, beforeEach, vi } from "vitest";
import { useSignSocket } from "./useSignSocket";

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = FakeWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: unknown[] = [];
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    setTimeout(() => {
      this.readyState = FakeWebSocket.OPEN;
      this.onopen?.();
    }, 0);
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

  it("does not send frames before the socket has opened", () => {
    const { result } = renderHook(() => useSignSocket("ws://test"));

    // Called immediately, before the fake onopen timeout has fired —
    // the underlying socket is still CONNECTING at this point.
    const blob = new Blob(["frame"]);
    act(() => {
      result.current.sendFrame(blob);
    });

    expect(result.current.connected).toBe(false);
    expect(FakeWebSocket.instances[0].sent).toEqual([]);
  });

  it("does not open a new socket if unmounted during the reconnect backoff window", () => {
    vi.useFakeTimers();
    try {
      const { result, unmount } = renderHook(() => useSignSocket("ws://test"));

      act(() => {
        vi.advanceTimersByTime(0); // fire the fake onopen timeout
      });
      expect(result.current.connected).toBe(true);
      expect(FakeWebSocket.instances).toHaveLength(1);

      act(() => {
        FakeWebSocket.instances[0].close(); // triggers onclose -> schedules reconnect at 500ms
      });
      expect(result.current.connected).toBe(false);

      unmount(); // must clear the pending reconnect timer

      act(() => {
        vi.advanceTimersByTime(1000); // past the 500ms retry delay
      });

      // No second WebSocket should have been constructed after unmount.
      expect(FakeWebSocket.instances).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("reconnects with exponential backoff after repeated closes", () => {
    vi.useFakeTimers();
    try {
      renderHook(() => useSignSocket("ws://test"));

      expect(FakeWebSocket.instances).toHaveLength(1);
      act(() => {
        vi.advanceTimersByTime(0); // let the first socket's fake onopen fire
      });

      act(() => {
        FakeWebSocket.instances[0].close(); // schedules reconnect at 500ms, delay -> 1000ms
      });
      act(() => {
        vi.advanceTimersByTime(499);
      });
      expect(FakeWebSocket.instances).toHaveLength(1); // not yet — still within 500ms delay
      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(FakeWebSocket.instances).toHaveLength(2); // reconnected at 500ms

      act(() => {
        // Close the second socket immediately, before its own fake onopen
        // timeout fires, so retryDelay (already doubled to 1000ms) is not
        // reset back to 500ms by an intervening open event.
        FakeWebSocket.instances[1].close();
      });
      act(() => {
        vi.advanceTimersByTime(999);
      });
      expect(FakeWebSocket.instances).toHaveLength(2); // not yet — this delay doubled to 1000ms
      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(FakeWebSocket.instances).toHaveLength(3); // reconnected at 1000ms, confirming backoff doubled
    } finally {
      vi.useRealTimers();
    }
  });
});
