import { useEffect } from "react";
import { useWebcam } from "./hooks/useWebcam";
import { useSignSocket } from "./hooks/useSignSocket";
import { useSpeech } from "./hooks/useSpeech";
import { captureFrameBlob } from "./lib/captureFrame";
import { SubtitleDisplay } from "./components/SubtitleDisplay";
import { StatusIndicator } from "./components/StatusIndicator";

const SEND_INTERVAL_MS = 80; // ~12.5 fps
const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/sign-stream";

export default function App() {
  const { videoRef, ready, error } = useWebcam();
  const { connected, result, sendFrame } = useSignSocket(WS_URL);
  useSpeech(result);

  useEffect(() => {
    if (!ready) return;
    const interval = setInterval(async () => {
      const video = videoRef.current;
      // videoWidth is 0 until loadedmetadata; a zero-area canvas makes
      // toBlob throw, so skip ticks until the stream reports dimensions.
      if (!video || !video.videoWidth) return;
      const blob = await captureFrameBlob(video);
      if (blob) sendFrame(blob);
    }, SEND_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [ready, videoRef, sendFrame]);

  return (
    <div className="relative h-screen w-screen bg-black">
      <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />
      {error && (
        <div className="absolute left-1/2 top-6 -translate-x-1/2 rounded-lg bg-red-600/90 px-4 py-2 text-sm font-medium text-white">
          Camera unavailable: {error}
        </div>
      )}
      <div className="absolute right-4 top-4">
        <StatusIndicator status={connected ? result.status : "no_hand"} />
      </div>
      <SubtitleDisplay result={result} />
    </div>
  );
}
