import { afterEach, describe, expect, it } from "vitest";

import { authOptions, googleAllowed } from "@/lib/auth";
import { hashPassword, parseUsers, verifyPassword } from "@/lib/server/passwords";

afterEach(() => {
  delete process.env.MLREDACT_UI_USERS;
  delete process.env.MLREDACT_UI_GOOGLE_ALLOWED_EMAILS;
  delete process.env.MLREDACT_UI_GOOGLE_ALLOWED_DOMAINS;
});

describe("password hashes", () => {
  it("verify the right password only", async () => {
    const hash = await hashPassword("correct horse battery");
    expect(hash).toMatch(/^scrypt\.[0-9a-f]{32}\.[0-9a-f]{64}$/);
    expect(await verifyPassword("correct horse battery", hash)).toBe(true);
    expect(await verifyPassword("correct horse batterY", hash)).toBe(false);
  });

  it("are salted", async () => {
    expect(await hashPassword("same")).not.toBe(await hashPassword("same"));
  });

  it("never verify an unknown user or a malformed hash", async () => {
    expect(await verifyPassword("anything", undefined)).toBe(false);
    expect(await verifyPassword("", undefined)).toBe(false);
    expect(await verifyPassword("x", "bcrypt.00.11")).toBe(false);
    expect(await verifyPassword("x", "scrypt.00.11")).toBe(false);
  });

  it("parse MLREDACT_UI_USERS, ignoring malformed entries", () => {
    const users = parseUsers(` Alice@Example.com=scrypt.aa.bb , broken, bob@example.com=md5.x, =scrypt.1.2`);
    expect([...users]).toEqual([["alice@example.com", "scrypt.aa.bb"]]);
  });
});

describe("email and password sign-in", () => {
  type Authorize = (c: Record<string, string>, req: object) => Promise<{ email: string } | null>;
  const authorize = (authOptions.providers[0] as unknown as { options: { authorize: Authorize } }).options.authorize;

  it("accepts a listed user with the right password, case-insensitively", async () => {
    process.env.MLREDACT_UI_USERS = `demo@example.com=${await hashPassword("a long demo password")}`;
    expect(await authorize({ email: "Demo@Example.com ", password: "a long demo password" }, {})).toMatchObject({
      email: "demo@example.com",
    });
    expect(await authorize({ email: "demo@example.com", password: "wrong" }, {})).toBeNull();
    expect(await authorize({ email: "other@example.com", password: "a long demo password" }, {})).toBeNull();
  });
});

describe("Google sign-in allowlist", () => {
  it("lets nobody in without an allowlist", () => {
    expect(googleAllowed("someone@gmail.com", true)).toBe(false);
  });

  it("accepts listed addresses and domains, only when Google verified the address", () => {
    process.env.MLREDACT_UI_GOOGLE_ALLOWED_EMAILS = "joel@gmail.com";
    process.env.MLREDACT_UI_GOOGLE_ALLOWED_DOMAINS = "lawfirm.com.au";
    expect(googleAllowed("Joel@Gmail.com", true)).toBe(true);
    expect(googleAllowed("anyone@lawfirm.com.au", true)).toBe(true);
    expect(googleAllowed("anyone@lawfirm.com.au", false)).toBe(false);
    expect(googleAllowed("anyone@evil-lawfirm.com.au", true)).toBe(false);
    expect(googleAllowed("someone@gmail.com", true)).toBe(false);
    expect(googleAllowed(null, true)).toBe(false);
  });

  it("is applied in the sign-in callback, and credentials users are not affected", async () => {
    const signIn = authOptions.callbacks!.signIn! as (p: object) => Promise<boolean>;
    expect(await signIn({ account: { provider: "google" }, profile: { email: "x@gmail.com", email_verified: true } })).toBe(
      false,
    );
    expect(await signIn({ account: { provider: "credentials" }, profile: undefined })).toBe(true);
  });
});
