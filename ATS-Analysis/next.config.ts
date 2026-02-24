import type { NextConfig } from "next";

// Proxy all `/api` requests from the frontend to the backend service. This
// removes the need for CORS and lets the UI simply call `/api/...` regardless
// of where the backend is listening. During development you can override the
// target with the `NEXT_PUBLIC_API_PROXY` env var (e.g. `http://localhost:5000`).
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_PROXY || "http://localhost:5000"}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
