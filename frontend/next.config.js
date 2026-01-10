/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  images: {
    domains: ['localhost'],
  },
  async rewrites() {
    // For SSR (server-side), try to use Docker internal hostname first
    // Fallback to NEXT_PUBLIC_API_URL or localhost
    // This allows the app to work both in Docker and locally
    let backendUrl;
    
    // Check if we have a specific backend URL for SSR
    if (process.env.BACKEND_URL) {
      backendUrl = process.env.BACKEND_URL;
    } else if (process.env.NODE_ENV === 'production') {
      // In production (Docker), try internal hostname first
      backendUrl = 'http://backend:8000';
    } else {
      // Development or fallback
      backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    }
    
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
