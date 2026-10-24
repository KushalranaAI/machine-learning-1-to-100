/** @type {import('next').NextConfig} */
const nextConfig = {
  // The app reads lessons and source code from the parent folder at request time.
  outputFileTracingRoot: new URL("..", import.meta.url).pathname,
};

export default nextConfig;
