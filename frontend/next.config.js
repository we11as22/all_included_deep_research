/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    // In Docker, use backend service name; otherwise use localhost or env var
    // Use 127.0.0.1 instead of localhost to avoid IPv6 issues
    const backendUrl = process.env.NEXT_PUBLIC_API_URL 
      ? process.env.NEXT_PUBLIC_API_URL 
      : process.env.NODE_ENV === 'production' 
        ? 'http://backend:8000'
        : 'http://127.0.0.1:8000';
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
