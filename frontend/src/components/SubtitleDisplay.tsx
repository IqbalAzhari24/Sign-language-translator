import type { SignResult } from "../hooks/useSignSocket";

export function SubtitleDisplay({ result }: { result: SignResult }) {
  const text = result.status === "recognized" ? result.label : "";

  return (
    <div
      aria-live="polite"
      role="status"
      className="absolute bottom-6 left-1/2 min-h-[3rem] min-w-[8rem] -translate-x-1/2 rounded-lg bg-black/70 px-6 py-3 text-center text-2xl font-semibold text-white"
    >
      {text}
    </div>
  );
}
