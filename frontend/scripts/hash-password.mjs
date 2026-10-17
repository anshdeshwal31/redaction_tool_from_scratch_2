// Prints a password hash for MLREDACT_UI_USERS:  npm run hash-password
// The password is read from the terminal, so it never lands in shell history.
import { randomBytes, scryptSync } from "node:crypto";
import { createInterface } from "node:readline";

const rl = createInterface({ input: process.stdin, output: process.stderr, terminal: Boolean(process.stdin.isTTY) });
rl._writeToOutput = (s) => {
  // Echo the prompt, mask the typing.
  if (s.includes("Password")) process.stderr.write(s);
};
rl.question("Password: ", (password) => {
  rl.close();
  process.stderr.write("\n");
  if (password.length < 12) {
    process.stderr.write("Use at least 12 characters.\n");
    process.exit(1);
  }
  const salt = randomBytes(16);
  const key = scryptSync(password, salt, 32, { N: 16384, r: 8, p: 1, maxmem: 64 * 1024 * 1024 });
  process.stdout.write(`scrypt.${salt.toString("hex")}.${key.toString("hex")}\n`);
});
