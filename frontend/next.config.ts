import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    proxyClientMaxBodySize: 1024 * 1024 * 1024, // 1 GiB uploads through local Next proxy
  },
};

export default nextConfig;
