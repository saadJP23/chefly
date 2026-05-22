const http = require("http");
const https = require("https");
const { URL } = require("url");

const port = process.env.PORT || 3000;
const apiBaseUrl = (process.env.API_BASE_URL || "").replace(/\/$/, "");

/** Map legacy .html URLs to Flask routes. */
const FLASK_PATH_MAP = {
  "/": "/",
  "/index.html": "/",
  "/home.html": "/",
  "/recipes.html": "/famous-dishes",
  "/famous_dishes.html": "/famous-dishes",
  "/generate.html": "/generate",
  "/submit-recipe.html": "/submit-recipe",
  "/submit_recipe.html": "/submit-recipe",
  "/upload.html": "/upload",
  "/trained-dishes.html": "/trained_dishes",
  "/trained_dishes.html": "/trained_dishes",
  "/about.html": "/about",
  "/contact.html": "/contact",
  "/login.html": "/login",
  "/signup.html": "/signup",
  "/search.html": "/search",
  "/search_results.html": "/search",
  "/forgot-password.html": "/forgot-password",
  "/forgot_password.html": "/forgot-password",
  "/reset_password.html": "/reset-password",
  "/profile.html": "/profile",
  "/admin_analytics.html": "/admin/analytics",
  "/recipe.html": "/search"
};

function resolveFlaskPath(urlPath) {
  const [pathname, search] = urlPath.split("?");
  if (pathname.startsWith("/static/") || pathname.startsWith("/uploads/")) {
    return urlPath;
  }
  const mapped = FLASK_PATH_MAP[pathname];
  const flaskPath = mapped || pathname.replace(/\.html$/, "") || "/";
  return search ? `${flaskPath}?${search}` : flaskPath;
}

function proxyToFlask(req, res) {
  const target = new URL(resolveFlaskPath(req.url), apiBaseUrl + "/");
  const client = target.protocol === "https:" ? https : http;

  const headers = { ...req.headers, host: target.host };
  delete headers.connection;

  const proxyReq = client.request(
    target,
    { method: req.method, headers },
    (proxyRes) => {
      const responseHeaders = { ...proxyRes.headers };
      delete responseHeaders.connection;

      // Rewrite Location headers so redirects come back through the proxy
      if (responseHeaders.location) {
        try {
          const loc = new URL(responseHeaders.location);
          if (loc.host === target.host) {
            const frontendHost = req.headers.host || `localhost:${port}`;
            loc.host = frontendHost;
            loc.protocol = "http:";
            responseHeaders.location = loc.toString();
          }
        } catch (_) {}
      }

      res.writeHead(proxyRes.statusCode || 502, responseHeaders);
      proxyRes.pipe(res);
    }
  );

  proxyReq.on("error", function () {
    res.writeHead(502, { "Content-Type": "text/html; charset=utf-8" });
    res.end("<!DOCTYPE html><html><body style='font-family:sans-serif;padding:2rem'><h1>Backend unavailable</h1><p>Flask backend at <code>" + apiBaseUrl + "</code> is not responding.</p></body></html>");
  });

  req.pipe(proxyReq);
}

http
  .createServer((req, res) => {
    if (req.url === "/config.js") {
      res.writeHead(200, { "Content-Type": "application/javascript; charset=utf-8" });
      return res.end(`window.CHEFLY_CONFIG = ${JSON.stringify({ apiBaseUrl })};`);
    }

    if (!apiBaseUrl) {
      res.writeHead(503, { "Content-Type": "text/plain" });
      return res.end("API_BASE_URL is not set.");
    }

    proxyToFlask(req, res);
  })
  .listen(port, () => {
    console.log(`Chefly frontend listening on ${port}, proxying all requests to Flask: ${apiBaseUrl}`);
  });
