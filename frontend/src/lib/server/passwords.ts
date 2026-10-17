import "server-only";

import { randomBytes, scrypt as scryptCb, timingSafeEqual, type BinaryLike, type ScryptOptions } from "node:crypto";

/*
 * Email-and-password users come from MLREDACT_UI_USERS: comma-separated `email=hash` entries, where
 * the hash is `scrypt.<salt hex>.<key hex>` from `npm run hash-password`.  Hex and dots only, so
 * the value survives .env files and dashboards (no `$` for variable expansion to eat).
 */

const PARAMS: ScryptOptions = { N: 16384, r: 8, p: 1, maxmem: 64 * 1024 * 1024 };
const KEY_BYTES = 32;
// Whole bytes of hex only: Buffer.from(…, "hex") silently stops at the first bad character.
const HEX = /^(?:[0-9a-f]{2})+$/i;

function scrypt(password: BinaryLike, salt: BinaryLike): Promise<Buffer> {
  return new Promise((resolve, reject) =>
    scryptCb(password, salt, KEY_BYTES, PARAMS, (err, key) => (err ? reject(err) : resolve(key))),
  );
}

export async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16);
  return `scrypt.${salt.toString("hex")}.${(await scrypt(password, salt)).toString("hex")}`;
}

// Checked when the email is unknown, so a wrong email costs the same time as a wrong password.
const DUMMY = `scrypt.${"00".repeat(16)}.${"00".repeat(KEY_BYTES)}`;

export async function verifyPassword(password: string, stored: string | undefined): Promise<boolean> {
  const [scheme, saltHex, keyHex] = (stored ?? DUMMY).split(".");
  if (scheme !== "scrypt" || !saltHex || !keyHex || !HEX.test(saltHex) || !HEX.test(keyHex)) return false;
  if (keyHex.length !== KEY_BYTES * 2) return false;
  const actual = await scrypt(password, Buffer.from(saltHex, "hex"));
  return timingSafeEqual(actual, Buffer.from(keyHex, "hex")) && stored !== undefined;
}

export function parseUsers(raw: string | undefined): Map<string, string> {
  const users = new Map<string, string>();
  for (const entry of (raw ?? "").split(",")) {
    const at = entry.lastIndexOf("=");
    if (at <= 0) continue;
    const email = entry.slice(0, at).trim().toLowerCase();
    const hash = entry.slice(at + 1).trim();
    if (email.includes("@") && hash.startsWith("scrypt.")) users.set(email, hash);
  }
  return users;
}
