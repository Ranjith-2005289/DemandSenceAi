import { useEffect, useState } from "react";

// Tiny hash router: #/dashboard, #/new, ... Keeps browser back/forward and
// deep links working without adding a routing dependency.
export const ROUTES = ["dashboard", "new", "forecast", "models", "products", "report", "settings"];

function current() {
  const id = window.location.hash.replace(/^#\/?/, "").split(/[/?]/)[0];
  return ROUTES.includes(id) ? id : "dashboard";
}

export function navigate(id) {
  if (current() !== id || !window.location.hash) window.location.hash = `/${id}`;
}

export function useRoute() {
  const [route, setRoute] = useState(current);
  useEffect(() => {
    const onChange = () => setRoute(current());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
