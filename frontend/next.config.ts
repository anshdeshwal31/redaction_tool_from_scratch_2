import type { NextConfig } from "next";

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "no-referrer" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), interest-cohort=()" },
];

const nextConfig: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  // No next/image optimisation: it would load sharp (libvips, LGPL-3.0), which this UI never needs.
  images: { unoptimized: true },
  // Don't write AGENTS.md into the project on `next dev`.
  agentRules: false,
  async headers() {
    return [
      { source: "/:path*", headers: securityHeaders },
      // Job data is document-derived: never let a browser or proxy cache it.
      { source: "/api/:path*", headers: [{ key: "Cache-Control", value: "no-store" }] },
    ];
  },
};

export default nextConfig;
