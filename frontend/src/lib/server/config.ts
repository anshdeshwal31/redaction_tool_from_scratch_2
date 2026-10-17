import "server-only";

/**
 * Frontend server settings, from the environment (see .env.example).  The API token never reaches
 * the browser: only this server talks to the backend's token-protected routes.
 */
export interface ServerConfig {
  /** Backend base URL as this server reaches it, e.g. http://localhost:8100 or https://api.example.com. */
  apiUrl: string;
  /** Backend base URL as browsers reach it, for direct uploads and downloads (defaults to apiUrl). */
  publicApiUrl: string;
  apiToken: string;
}

const trimSlash = (u: string) => u.trim().replace(/\/+$/, "");

export function serverConfig(): ServerConfig {
  const apiUrl = trimSlash(process.env.MLREDACT_API_URL ?? "http://127.0.0.1:8100");
  return {
    apiUrl,
    publicApiUrl: trimSlash(process.env.MLREDACT_API_PUBLIC_URL || apiUrl),
    apiToken: process.env.MLREDACT_API_TOKEN ?? "",
  };
}
