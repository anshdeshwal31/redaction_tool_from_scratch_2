"use client";

import { useEffect, useState } from "react";

import { cx } from "./ui";

type Theme = "light" | "dark";
const STORAGE_KEY = "mlredact-theme";
const THEMES: Theme[] = ["light", "dark"];

/**
 * Runs before first paint (inlined in the layout) so the stored theme never flashes.  Dark is the
 * default.  It also marks <html> with `js`, which lets scroll reveals hide content only when
 * scripts actually run.
 */
export const THEME_BOOTSTRAP = `document.documentElement.classList.add("js");try{if(localStorage.getItem("${STORAGE_KEY}")==="light")document.documentElement.dataset.theme="light"}catch(e){}`;

function apply(theme: Theme) {
  if (theme === "dark") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  try {
    if (theme === "dark") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* storage unavailable: the choice lasts for this page only */
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    if (document.documentElement.dataset.theme === "light") setTheme("light");
  }, []);

  return (
    <div role="radiogroup" aria-label="Colour theme" className="relative inline-flex border border-line bg-surface p-1">
      <span
        aria-hidden
        className="absolute top-1 bottom-1 left-1 w-16 bg-surface-3 transition-transform duration-300 ease-out"
        style={{ transform: `translateX(${THEMES.indexOf(theme) * 100}%)` }}
      />
      {THEMES.map((t) => (
        <button
          key={t}
          type="button"
          role="radio"
          aria-checked={theme === t}
          onClick={() => {
            apply(t);
            setTheme(t);
          }}
          className={cx(
            "relative h-7 w-16 text-xs capitalize transition-colors",
            theme === t ? "text-ink" : "text-muted hover:text-ink",
          )}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
