/*
 * LM Arena service worker.
 *
 * The arena runs on the phone (or any box) that serves this page, so "offline"
 * here means: the app shell keeps working while the model server restarts, and
 * the home-screen app opens instantly. Generated text is never cached -- every
 * /api/* and /health request goes to the network, always.
 */

const CACHE = "lm-arena-v1";
const SHELL = [
  "/",
  "/leaderboard",
  "/static/style.css",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting())
      .catch(() => self.skipWaiting())   // a missing shell file must not block install
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // Never interfere with generation, votes or status: those must be live.
  if (request.method !== "GET" || url.origin !== self.location.origin
      || url.pathname.startsWith("/api/") || url.pathname === "/health") {
    return;
  }

  // Navigations: network first so a rebuild is picked up, cache as fallback.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() =>
        caches.match(request).then((hit) => hit || caches.match("/")))
    );
    return;
  }

  // Static assets: cache first, refresh in the background.
  event.respondWith(
    caches.match(request).then((hit) => {
      const network = fetch(request).then((response) => {
        if (response && response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      }).catch(() => hit);
      return hit || network;
    })
  );
});
