import { useEffect, useRef, useState } from "react";

export function useWebcam() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((s) => {
        if (cancelled) {
          // Unmounted before getUserMedia resolved — stop the granted
          // stream immediately instead of leaking it (camera indicator
          // would otherwise stay lit until the page fully closes).
          s.getTracks().forEach((track) => track.stop());
          return;
        }
        stream = s;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          setReady(true);
        }
      })
      .catch((err: Error) => setError(err.message));

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  return { videoRef, ready, error };
}
