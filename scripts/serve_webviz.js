#!/usr/bin/env node
/**
 * Minimal zero-dependency static file server for the FlashEdges web viewer.
 *
 *   node scripts/serve_webviz.js          # serves repo root -> http://localhost:8000/webviz/
 *   node scripts/serve_webviz.js 9000     # custom port
 *
 * Works with plain Node (no npm install). Falls back to the same logic as
 * `python3 -m http.server` — any GET, with correct MIME types for the assets
 * the viewer needs (.html, .js, .mjs, .css, .webp, .png, .json).
 */
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = +process.argv[2] || 8000;
const ROOT = path.resolve(__dirname, "..");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js":   "application/javascript; charset=utf-8",
  ".mjs":  "application/javascript; charset=utf-8",
  ".css":  "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".webp": "image/webp",
  ".png":  "image/png",
  ".jpg":  "image/jpeg",
  ".svg":  "image/svg+xml",
  ".ico":  "image/x-icon",
  ".map":  "application/json; charset=utf-8",
};

const server = http.createServer((req, res) => {
  // strip query string, prevent path traversal
  const decoded = decodeURIComponent(req.url.split("?")[0]);
  const safe = path.normalize(decoded).replace(/^(\.\.[/\\])+/, "");
  let filePath = path.join(ROOT, safe);

  fs.stat(filePath, (err, stat) => {
    if (err || !stat) {
      // directory listing if path is a folder
      if (!err && stat === undefined) {} // unreachable
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("404 Not Found: " + decoded);
      return;
    }
    if (stat.isDirectory()) {
      filePath = path.join(filePath, "index.html");
      fs.stat(filePath, (e2, s2) => {
        if (e2 || !s2.isFile()) {
          res.writeHead(404, { "Content-Type": "text/plain" });
          res.end("404 Not Found: " + decoded);
          return;
        }
        serve(filePath, res);
      });
      return;
    }
    serve(filePath, res);
  });

  function serve(f, res) {
    const ext = path.extname(f).toLowerCase();
    res.writeHead(200, {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control": "no-cache",
      // MapLibre sets crossOrigin="anonymous" on images used as WebGL
      // textures, so the server must send CORS headers even for same-origin.
      "Access-Control-Allow-Origin": "*",
    });
    fs.createReadStream(f).on("error", () => {
      res.writeHead(500, { "Content-Type": "text/plain" });
      res.end("500 Internal Server Error");
    }).pipe(res);
  }
});

server.listen(PORT, () => {
  const url = `http://localhost:${PORT}/webviz/`;
  console.log(`FlashEdges web viewer served from:\n  ${ROOT}\n\nOpen:\n  ${url}\n\nCtrl-C to stop.`);
});
