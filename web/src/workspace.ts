/* The one piece of per-tab UI state: which run the user last opened, so the side nav's
   "Current run" link has somewhere to go. sessionStorage, not localStorage -- it is scoped to
   this tab and means nothing in another one. Every access is guarded: storage throws in a private
   window, and this is a convenience, never a source of truth. */

const LAST_RUN = "wmx.peptide.lastRun";

export const lastRun = {
  get(): string | null {
    try {
      return sessionStorage.getItem(LAST_RUN);
    } catch {
      return null;
    }
  },
  set(jobId: string): void {
    try {
      sessionStorage.setItem(LAST_RUN, jobId);
    } catch {
      /* private mode: the link is simply disabled */
    }
  },
};
