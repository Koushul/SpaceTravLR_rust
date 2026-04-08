import { apiUrl } from "./apiClient";
import type { Meta } from "./viewerTypes";

export async function withMetaProgressPoll<T>(
  applyMetaProgressToUi: (m: Meta) => void,
  work: Promise<T>,
  clearProgressBar?: () => void,
): Promise<T> {
  let pollInFlight = false;
  const id = window.setInterval(() => {
    if (pollInFlight) return;
    pollInFlight = true;
    void (async () => {
      try {
        const mr = await fetch(apiUrl("/api/meta"));
        if (!mr.ok) return;
        const m = (await mr.json()) as Meta;
        applyMetaProgressToUi(m);
      } catch {
        /* ignore */
      } finally {
        pollInFlight = false;
      }
    })();
  }, 150);
  try {
    return await work;
  } finally {
    clearInterval(id);
    try {
      const mr = await fetch(apiUrl("/api/meta"));
      if (mr.ok) {
        const m = (await mr.json()) as Meta;
        applyMetaProgressToUi(m);
        const pm = m.perturb_progress_permille;
        const pct = m.perturb_progress_percent;
        if (
          (pm != null && Number.isFinite(pm)) ||
          (pct != null && Number.isFinite(pct))
        ) {
          await new Promise((r) => setTimeout(r, 200));
        }
      }
    } catch {
      /* ignore */
    }
    clearProgressBar?.();
  }
}
