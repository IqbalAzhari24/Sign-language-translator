import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusIndicator } from "./StatusIndicator";

describe("StatusIndicator", () => {
  it("shows the label for the given status", () => {
    render(<StatusIndicator status="no_hand" />);
    expect(screen.getByText("No hand detected")).toBeInTheDocument();
  });
});
