"use client";

import { FileImage, FileText, UploadCloud, X } from "lucide-react";
import { useId, useRef, useState } from "react";

import { formatBytes } from "@/lib/format";

import { cx } from "./ui";

export const ACCEPTED = ".pdf,.tif,.tiff,.png,.jpg,.jpeg";
const ACCEPTED_EXT = /\.(pdf|tiff?|png|jpe?g)$/i;

export function FileDrop({
  file,
  onChange,
  maxBytes,
  disabled,
}: {
  file: File | null;
  onChange: (file: File | null, error?: string) => void;
  maxBytes: number;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const hintId = useId();

  const pick = (f: File | undefined) => {
    if (!f) return;
    if (!ACCEPTED_EXT.test(f.name)) return onChange(null, "Use a PDF, TIFF, PNG or JPEG file.");
    if (f.size === 0) return onChange(null, "That file is empty.");
    if (f.size > maxBytes) return onChange(null, `That file is larger than the ${formatBytes(maxBytes)} limit.`);
    onChange(f);
  };

  if (file) {
    const isPdf = /\.pdf$/i.test(file.name);
    const Icon = isPdf ? FileText : FileImage;
    return (
      <div className="fade-up flex items-center gap-4 bg-surface-2 px-4 py-4">
        <div className="pop flex size-11 shrink-0 items-center justify-center bg-ink text-page">
          <Icon className="size-5" aria-hidden />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-ink" title={file.name}>
            {file.name}
          </div>
          <div className="mt-0.5 text-xs text-muted">
            {formatBytes(file.size)} · {isPdf ? "PDF" : "Image scan"}
          </div>
        </div>
        <button
          type="button"
          onClick={() => onChange(null)}
          disabled={disabled}
          className="inline-flex size-9 items-center justify-center text-muted transition-colors hover:bg-surface-3 hover:text-ink"
          aria-label="Remove file"
        >
          <X className="size-4" aria-hidden />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) pick(e.dataTransfer.files[0]);
      }}
      className={cx(
        "group relative flex flex-col items-center justify-center gap-3 border border-dashed px-6 py-14 text-center transition-colors duration-300",
        dragging ? "border-ink bg-surface-3" : "border-line-strong bg-surface-2/60 hover:bg-surface-2",
      )}
    >
      <UploadCloud
        className={cx(
          "size-8 transition-[translate,scale,color] duration-300 ease-out",
          dragging ? "-translate-y-1.5 scale-110 text-ink" : "text-muted group-hover:-translate-y-0.5",
        )}
        strokeWidth={1.5}
        aria-hidden
      />
      <div className="text-[0.9375rem] text-ink-2">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="font-medium text-ink underline underline-offset-4 hover:decoration-2"
          aria-describedby={hintId}
        >
          Choose a document
        </button>{" "}
        or drop it here
      </div>
      <p id={hintId} className="text-xs text-muted">
        PDF, TIFF, PNG or JPEG · up to {formatBytes(maxBytes)}
      </p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          pick(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
    </div>
  );
}
