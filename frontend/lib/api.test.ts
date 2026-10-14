import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, getDatasets, getHealth, postInference } from "./api";

describe("API client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests the health payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: "ok", api_version: "0.1.0" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getHealth()).resolves.toEqual({ status: "ok", api_version: "0.1.0" });
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/health", undefined);
  });

  it("reads catalog categories", async () => {
    const payload = {
      id: "mvtec_ad",
      name: "MVTec AD",
      license: "CC BY-NC-SA 4.0",
      license_url: "https://example.test",
      commercial_use: false,
      storage_root: "data/mvtec-ad",
      categories: [],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 })),
    );

    await expect(getDatasets()).resolves.toEqual(payload);
  });

  it("surfaces the API detail when inference is not implemented", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(
        () =>
          new Response(JSON.stringify({ detail: "No trained artifact is available for this model yet." }), {
            status: 501,
          }),
      ),
    );

    const form = new FormData();
    await expect(postInference(form)).rejects.toBeInstanceOf(ApiError);
    await expect(postInference(form)).rejects.toThrow(/artifact/);
  });
});
