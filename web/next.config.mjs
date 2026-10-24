import { fileURLToPath } from "node:url";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // The app reads lessons and source code from the parent folder at request time.
  // fileURLToPath (not URL.pathname) so Windows gets "E:\..." rather than "/E:/...".
  outputFileTracingRoot: fileURLToPath(new URL("..", import.meta.url)),
};

export default nextConfig;
