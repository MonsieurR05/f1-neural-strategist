import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "F1 Neural Race Strategist",
  description:
    "Explore Formula 1 race strategy with a neural lap-time model and a Monte Carlo race simulator.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <header className="border-b border-line bg-surface-1">
          <div className="mx-auto flex max-w-[1400px] items-center justify-between gap-4 px-4 py-3">
            <Link href="/" className="flex items-baseline gap-2">
              <span className="text-sm font-semibold tracking-wide text-ink">NEURAL RACE STRATEGIST</span>
              <span className="hidden text-xs text-ink-3 sm:inline">F1 strategy lab</span>
            </Link>
            <span className="text-xs text-ink-3">Unofficial · not associated with Formula 1</span>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
