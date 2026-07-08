import type { SignResult } from "../hooks/useSignSocket";

const LABELS: Record<SignResult["status"], string> = {
  no_hand: "No hand detected",
  tracking: "Recognizing…",
  unsure: "Unsure",
  recognized: "Recognized",
};

const COLORS: Record<SignResult["status"], string> = {
  no_hand: "bg-gray-400",
  tracking: "bg-yellow-400",
  unsure: "bg-orange-400",
  recognized: "bg-green-500",
};

export function StatusIndicator({ status }: { status: SignResult["status"] }) {
  return (
    <div className="flex items-center gap-2 rounded-full bg-white/90 px-3 py-1 text-sm shadow">
      <span className={`h-2 w-2 rounded-full ${COLORS[status]}`} />
      {LABELS[status]}
    </div>
  );
}
