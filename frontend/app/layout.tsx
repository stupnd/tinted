import type { Metadata, Viewport } from "next";
import { Inter, Fraunces } from "next/font/google";
import { Navbar } from "@/components/layout/Navbar";
import { WakeServer } from "@/components/layout/WakeServer";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

// Editorial serif for the one hero line per screen. Optical sizing keeps it
// from looking spindly at display sizes.
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
  axes: ["SOFT", "WONK", "opsz"],
});

export const metadata: Metadata = {
  title: "Tinted — Your Perfect Shade",
  description:
    "Find your foundation match and build a skincare routine, from a photo or a two-minute quiz.",
};

export const viewport: Viewport = {
  themeColor: "#0b0b0c",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="min-h-screen bg-bg text-text antialiased">
        {/* First thing in the tab order — keyboard users shouldn't have to
            traverse the nav on every page. */}
        <a href="#main" className="sr-only-focusable">
          Skip to content
        </a>
        <WakeServer />
        <Navbar />
        <main id="main" className="pt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
