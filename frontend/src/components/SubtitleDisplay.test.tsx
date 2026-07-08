import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SubtitleDisplay } from "./SubtitleDisplay";

describe("SubtitleDisplay", () => {
  it("shows the recognized label", () => {
    render(<SubtitleDisplay result={{ status: "recognized", label: "SATU", confidence: 0.9 }} />);
    expect(screen.getByText("SATU")).toBeInTheDocument();
  });

  it("shows nothing when not recognized", () => {
    render(<SubtitleDisplay result={{ status: "tracking" }} />);
    expect(screen.queryByText("SATU")).not.toBeInTheDocument();
  });
});
