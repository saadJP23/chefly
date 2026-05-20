const http = require("http");
const fs = require("fs");
const path = require("path");

const publicDir = path.join(__dirname, "public");
const port = process.env.PORT || 3000;
const apiBaseUrl = process.env.API_BASE_URL || "";

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

function send(res, status, body, type = "text/plain; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": type.includes("text/html") ? "no-store" : "public, max-age=31536000, immutable"
  });
  res.end(body);
}

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split("?")[0]);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  return path.join(publicDir, normalized === "/" ? "index.html" : normalized);
}

http.createServer((req, res) => {
  if (req.url === "/config.js") {
    const body = `window.CHEFLY_CONFIG = ${JSON.stringify({ apiBaseUrl })};`;
    return send(res, 200, body, "application/javascript; charset=utf-8");
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
}).listen(port, () => {
  console.log(`Chefly frontend listening on ${port}`);
});
