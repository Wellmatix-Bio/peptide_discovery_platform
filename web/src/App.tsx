import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./Layout";
import { useAuth } from "./auth/AuthContext";
import { Recover, RecoveryCodeShown, Register, SignIn } from "./auth/pages";
import { Account } from "./account/pages";
import { ModelsAndHealth } from "./model/pages";
import { History } from "./runs/History";
import { RunView } from "./runs/RunView";
import { SubmitRun } from "./runs/SubmitRun";
import { Loading } from "./components/ui";

export default function App() {
  const { user, ready } = useAuth();

  /* Nothing is decided until the remembered session has been confirmed, or a signed-in user
     would be bounced to the sign-in page on every reload. */
  if (!ready) return <Loading>Checking your session…</Loading>;

  if (!user) {
    return (
      <Routes>
        <Route path="/register" element={<Register />} />
        <Route path="/register/recovery-code" element={<RecoveryCodeShown />} />
        <Route path="/recover" element={<Recover />} />
        <Route path="*" element={<SignIn />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<SubmitRun />} />
        <Route path="/runs" element={<History />} />
        <Route path="/runs/:jobId/*" element={<RunView />} />
        <Route path="/models" element={<ModelsAndHealth />} />
        <Route path="/account" element={<Account />} />
        <Route path="/register/recovery-code" element={<RecoveryCodeShown />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
