// Runs a lesson demo or project script with the local Python and streams its output as NDJSON:
//   {"type":"out","text":"..."}  {"type":"img","name":"loss.png","data":"data:image/png;base64,..."}  {"type":"exit","code":0}
// Only .py files inside curriculum/ or projects/ can run, and only for requests from this machine.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ROOT, safeInside } from "@/lib/content";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ALLOWED_DIRS = [path.join(ROOT, "curriculum"), path.join(ROOT, "projects")];
const LOCAL_HOSTS = /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/;

function pythonCommand(): string {
  return process.env.ML_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

export async function POST(req: Request) {
  if (!LOCAL_HOSTS.test(req.headers.get("host") ?? "")) {
    return new Response("Scripts can only be run from localhost.", { status: 403 });
  }
  const { script, args = [] } = (await req.json()) as { script: string; args?: string[] };
  const full = path.resolve(ROOT, script ?? "");
  const allowed = ALLOWED_DIRS.some((d) => safeInside(d, full)) && full.endsWith(".py") && fs.existsSync(full);
  if (!allowed || !Array.isArray(args) || args.some((a) => typeof a !== "string" || a.length > 200)) {
    return new Response("Not a runnable course script.", { status: 400 });
  }

  const outDir = fs.mkdtempSync(path.join(os.tmpdir(), "ml-run-"));
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      const send = (obj: object) => controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
      const child = spawn(pythonCommand(), ["-u", full, ...args], {
        cwd: path.dirname(full),
        env: { ...process.env, ML_OUTPUT_DIR: outDir, MPLBACKEND: "Agg", PYTHONIOENCODING: "utf-8" },
      });
      const timer = setTimeout(() => child.kill(), 15 * 60 * 1000);
      child.stdout.on("data", (d) => send({ type: "out", text: d.toString() }));
      child.stderr.on("data", (d) => send({ type: "err", text: d.toString() }));
      child.on("error", (e) => {
        send({ type: "err", text: `Could not start Python (${pythonCommand()}): ${e.message}\nSet ML_PYTHON to your interpreter.` });
      });
      child.on("close", (code) => {
        clearTimeout(timer);
        for (const name of fs.readdirSync(outDir).filter((f) => f.endsWith(".png")).sort()) {
          const data = fs.readFileSync(path.join(outDir, name)).toString("base64");
          send({ type: "img", name, data: `data:image/png;base64,${data}` });
        }
        fs.rmSync(outDir, { recursive: true, force: true });
        send({ type: "exit", code });
        controller.close();
      });
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store" } });
}
