/* The one piece of per-tab UI state: which run the user last opened, so the side nav's
   "Current run" link has somewhere to go. sessionStorage, not localStorage -- it is scoped to
   this tab and means nothing in another one. Every access is guarded: storage throws in a private
   window, and this is a convenience, never a source of truth.

   CLEARED ON EVERY SIGN-IN AND SIGN-OUT. It is per TAB, not per account, so without that a second
   account signing in to the same tab inherits the first one's "Current run" link -- and can read
   the previous account's job id straight out of the href. Opening it is refused by the proxy's
   ownership check, so no run data leaks, but the id should not be there to read. Found by signing
   in as a second account during the walkthrough. */

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
  clear(): void {
    try {
      sessionStorage.removeItem(LAST_RUN);
    } catch {
      /* nothing to clear if storage is unavailable */
    }
  },
};
