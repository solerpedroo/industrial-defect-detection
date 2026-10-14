"use client";

import { useEffect, useState } from "react";

import { request } from "@/lib/api";

type ResourceState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
};

export function useApiGet<T>(path: string): ResourceState<T> {
  const [state, setState] = useState<ResourceState<T> & { path: string }>({
    data: null,
    error: null,
    loading: true,
    path,
  });

  if (state.path !== path) {
    setState({ data: null, error: null, loading: true, path });
  }

  useEffect(() => {
    let cancelled = false;
    request<T>(path)
      .then((data) => {
        if (!cancelled) {
          setState({ data, error: null, loading: false, path });
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          const message = reason instanceof Error ? reason.message : "Request failed.";
          setState({ data: null, error: message, loading: false, path });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data: state.data, error: state.error, loading: state.loading };
}
