const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const publicDir = path.join(__dirname, "public");
const port = process.env.PORT || 3000;
const apiBaseUrl = (process.env.API_BASE_URL || "").replace(/\/$/, "");

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

/** Map static .html URLs to Flask routes (Jinja-rendered pages). */
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

function send(res, status, body, type = "text/plain; charset=utf-8", extraHeaders = {}) {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": type.includes("text/html") ? "no-store" : "public, max-age=31536000, immutable",
    ...extraHeaders
  });
  res.end(body);
}

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  return path.join(publicDir, normalized === "/" ? "index.html" : normalized);
}

function resolveFlaskPath(urlPath) {
  const [pathname, search] = urlPath.split("?");
  const mapped = FLASK_PATH_MAP[pathname];
  const flaskPath = mapped || pathname.replace(/\.html$/, "") || "/";
  return search ? `${flaskPath}?${search}` : flaskPath;
}

function shouldProxyToFlask(urlPath) {
  if (!apiBaseUrl) return false;
  if (urlPath.startsWith("/static/")) return false;
  if (urlPath === "/config.js") return false;
  if (urlPath.startsWith("/api/")) return false;
  const pathname = urlPath.split("?")[0];
  if (pathname.includes(".")) {
    const ext = path.extname(pathname).toLowerCase();
    return ext === ".html" || ext === "";
  }
  return true;
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
      res.writeHead(proxyRes.statusCode || 502, responseHeaders);
      proxyRes.pipe(res);
    }
  );

  proxyReq.on("error", function () {
    send(
      res,
      502,
      "<!DOCTYPE html><html><body style='font-family:sans-serif;padding:2rem'><h1>Backend unavailable</h1><p>Set <code>API_BASE_URL</code> to your Flask server URL.</p></body></html>",
      "text/html; charset=utf-8"
    );
  });

  req.pipe(proxyReq);
}

http
  .createServer((req, res) => {
    if (req.url === "/config.js") {
      const body = `window.CHEFLY_CONFIG = ${JSON.stringify({ apiBaseUrl })};`;
      return send(res, 200, body, "application/javascript; charset=utf-8");
    }

    if (shouldProxyToFlask(req.url)) {
      return proxyToFlask(req, res);
    }

    let filePath = safePath(req.url);
    if (!path.extname(filePath)) filePath += ".html";

    fs.readFile(filePath, (err, data) => {
      if (err) {
        return fs.readFile(path.join(publicDir, "404.html"), (notFoundErr, notFound) => {
          send(res, 404, notFoundErr ? "Not found" : notFound, "text/html; charset=utf-8");
        });
      }

      const type = contentTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream";
      send(res, 200, data, type);
    });
  })
  .listen(port, () => {
    console.log(`Chefly frontend listening on ${port}`);
    if (apiBaseUrl) {
      console.log(`HTML pages proxied to Flask (Jinja): ${apiBaseUrl}`);
    } else {
      console.warn("API_BASE_URL not set — only /static and index.html are served locally.");
    }
  });
