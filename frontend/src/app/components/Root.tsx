import { Outlet, useLocation } from "react-router";
import { Navbar } from "./Navbar";

export function Root() {
  const location = useLocation();
  const isLandingPage = location.pathname === "/";

  return (
    <div className="min-h-screen flex flex-col">
      {!isLandingPage && <Navbar />}
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  );
}