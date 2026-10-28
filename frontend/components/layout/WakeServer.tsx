"use client";

import { useEffect } from "react";
import { wakeServer } from "@/lib/api";

/** Renders nothing; starts the backend booting as soon as any page loads. */
export function WakeServer() {
  useEffect(() => {
    wakeServer();
  }, []);
  return null;
}
