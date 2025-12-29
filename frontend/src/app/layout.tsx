import type { Metadata } from 'next';
import { Space_Grotesk } from 'next/font/google';
import '@/styles/globals.css';
import { cn } from '@/lib/utils';

const spaceGrotesk = Space_Grotesk({ subsets: ['latin'], weight: ['400', '500', '600', '700'] });

export const metadata: Metadata = {
  title: 'All-Included Deep Research',
  description: 'Comprehensive deep research system with memory integration',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn('min-h-screen bg-background antialiased', spaceGrotesk.className)}>
        {children}
      </body>
    </html>
  );
}
