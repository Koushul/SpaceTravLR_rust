import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import WebSocket from "ws";

const root = fileURLToPath(new URL("..", import.meta.url));
const port = Number(process.env.SMOKE_PORT || 8791);
const cdpPort = Number(process.env.CDP_PORT || 9222);
const mime = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".wasm": "application/wasm",
  ".css": "text/css",
};

const server = createServer((req, res) => {
  const url = new URL(req.url || "/", `http://127.0.0.1:${port}`);
  let rel = decodeURIComponent(url.pathname);
  if (rel === "/") rel = "/webgpu_browser_smoke.html";
  const path = normalize(join(root, rel));
  if (!path.startsWith(root) || !existsSync(path)) {
    res.writeHead(404);
    res.end("not found");
    return;
  }
  const body = readFileSync(path);
  res.writeHead(200, {
    "content-type": mime[extname(path)] || "application/octet-stream",
    "cache-control": "no-store",
  });
  res.end(body);
});

await new Promise((r) => server.listen(port, "127.0.0.1", r));
const pageUrl = `http://127.0.0.1:${port}/webgpu_browser_smoke.html`;
console.log("serving", pageUrl);

const chrome =
  process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const userData = `/tmp/spacetravlr_chrome_webgpu_smoke_${process.pid}`;
const child = spawn(
  chrome,
  [
    `--user-data-dir=${userData}`,
    "--headless=new",
    "--disable-gpu-sandbox",
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    `--remote-debugging-port=${cdpPort}`,
    pageUrl,
  ],
  { stdio: ["ignore", "pipe", "pipe"] },
);
let stderr = "";
child.stderr.on("data", (d) => {
  stderr += d.toString();
});

async function waitForSmoke(timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  let wsUrl;
  while (Date.now() < deadline) {
    try {
      const tabs = await fetch(`http://127.0.0.1:${cdpPort}/json/list`).then((r) => r.json());
      const page = tabs.find(
        (t) => t.type === "page" && t.url.includes("webgpu_browser_smoke"),
      );
      if (page?.webSocketDebuggerUrl) {
        wsUrl = page.webSocketDebuggerUrl;
        break;
      }
    } catch {
      // chrome not ready
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  if (!wsUrl) throw new Error(`no CDP target\n${stderr.slice(-2000)}`);

  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => {
    ws.once("open", res);
    ws.once("error", rej);
  });

  let id = 0;
  const pending = new Map();
  ws.on("message", (raw) => {
    const msg = JSON.parse(raw.toString());
    if (msg.method === "Runtime.consoleAPICalled") {
      console.log(
        "CONSOLE",
        msg.params.type,
        msg.params.args?.map((a) => a.value ?? a.description).join(" "),
      );
    }
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
    }
  });
  function send(method, params = {}) {
    const mid = ++id;
    return new Promise((resolve, reject) => {
      pending.set(mid, { resolve, reject });
      ws.send(JSON.stringify({ id: mid, method, params }));
    });
  }

  await send("Runtime.enable");
  while (Date.now() < deadline) {
    const ev = await send("Runtime.evaluate", {
      expression: "window.__SMOKE__ ? JSON.stringify(window.__SMOKE__) : null",
      returnByValue: true,
    });
    const v = ev.result?.value;
    if (v) {
      ws.close();
      return JSON.parse(v);
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  ws.close();
  throw new Error("timeout waiting for __SMOKE__");
}

try {
  const result = await waitForSmoke();
  console.log("BROWSER_SMOKE", JSON.stringify(result, null, 2));
  if (!result.ok) process.exitCode = 1;
  else if (result.backend !== "webgpu") {
    console.warn("NOTE: backend is not webgpu — headless WebGPU may be unavailable");
    process.exitCode = 2;
  }
} catch (e) {
  console.error("browser smoke failed:", e);
  process.exitCode = 1;
} finally {
  child.kill("SIGTERM");
  server.close();
  setTimeout(() => process.exit(process.exitCode || 0), 400);
}
