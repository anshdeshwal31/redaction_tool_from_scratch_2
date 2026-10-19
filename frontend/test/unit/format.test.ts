import { describe, expect, it } from "vitest";

import { formatDuration } from "@/lib/format";

describe("durations", () => {
  it.each([
    [null, "–"],
    [420, "420 ms"],
    [999.6, "1.0 s"],
    [9_400, "9.4 s"],
    [9_960, "10 s"],
    [59_400, "59 s"],
    [59_600, "1 min 00 s"],
    [119_700, "2 min 00 s"],
    [125_000, "2 min 05 s"],
    [3_599_400, "59 min 59 s"],
    [3_599_700, "1 h 00 min"],
    [5_430_000, "1 h 31 min"],
  ])("%s ms reads %s", (ms, text) => {
    expect(formatDuration(ms)).toBe(text);
  });
});
