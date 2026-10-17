import type { Metadata, Viewport } from "next";
import { getServerSession } from "next-auth";

import { AppHeader } from "@/components/app-header";
import { RevealObserver } from "@/components/motion";
import { SiteFooter } from "@/components/site-footer";
import { THEME_BOOTSTRAP } from "@/components/theme-toggle";
import { authOptions } from "@/lib/auth";

// Bundled with the app (no font CDN): the interface makes no external requests.
import "@fontsource/instrument-serif/400.css";
import "@fontsource/instrument-serif/400-italic.css";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "mlredact", template: "%s · mlredact" },
  description: "Local, deterministic PII redaction for Australian medico-legal documents.",
  robots: { index: false, follow: false },
  referrer: "no-referrer",
};

export const viewport: Viewport = {
  themeColor: "#0e1214",
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await getServerSession(authOptions);
  return (
    <html lang="en-AU" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body className="flex min-h-dvh flex-col">
        <AppHeader user={session?.user?.email ?? null} />
        <main className="mx-auto w-full max-w-352 flex-1 px-4 sm:px-6 lg:px-8">{children}</main>
        <SiteFooter />
        <RevealObserver />
      </body>
    </html>
  );
}
