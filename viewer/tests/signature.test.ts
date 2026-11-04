import { describe, it, expect } from "vitest";
import { annText, defaultText } from "../src/lib/signature.ts";

describe("defaultText", () => {
  it("returns plain string defaults unchanged", () => {
    expect(defaultText("1")).toBe("1");
    expect(defaultText("'hi'")).toBe("'hi'");
  });

  it("returns null for Empty/missing defaults", () => {
    expect(defaultText(null)).toBe(null);
    expect(defaultText({ __type: "Empty", __tag: 4031 })).toBe(null);
  });
});

describe("annText", () => {
  it("returns string annotations and null for Empty/missing", () => {
    expect(annText("int")).toBe("int");
    expect(annText(null)).toBe(null);
    expect(annText({ __type: "Empty", __tag: 4031 })).toBe(null);
  });
});
