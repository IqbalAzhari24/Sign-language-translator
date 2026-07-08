import { describe, expect, it, vi } from "vitest";
import { captureFrameBlob } from "./captureFrame";

describe("captureFrameBlob", () => {
  it("draws the video frame onto a canvas and requests a jpeg blob", async () => {
    const drawImage = vi.fn();
    const toBlob = vi.fn((cb: BlobCallback) => cb(new Blob(["x"])));
    const getContext = vi.fn(() => ({ drawImage }));

    vi.spyOn(document, "createElement").mockReturnValue({
      width: 0,
      height: 0,
      getContext,
      toBlob,
    } as unknown as HTMLCanvasElement);

    const video = { videoWidth: 320, videoHeight: 240 } as HTMLVideoElement;
    const blob = await captureFrameBlob(video);

    expect(getContext).toHaveBeenCalledWith("2d");
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 320, 240);
    expect(blob).toBeInstanceOf(Blob);
  });
});
