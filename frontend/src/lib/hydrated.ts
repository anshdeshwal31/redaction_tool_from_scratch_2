"use client";

import { useSyncExternalStore } from "react";

const subscribe = () => () => {};

/** False on the server and while React hydrates; true once the component's handlers are attached. */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
