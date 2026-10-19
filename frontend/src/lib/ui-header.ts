/**
 * The header every change the UI makes carries.  Browsers cannot send it cross-site without a CORS
 * preflight this server never grants, so lib/server/http.ts refuses a change without it.
 */
export const UI_HEADER = "x-mlredact-ui";
export const MUTATION_HEADERS = { [UI_HEADER]: "1" };
