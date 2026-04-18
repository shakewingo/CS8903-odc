import type { Metadata } from "next";
import { Crimson_Pro, Manrope } from "next/font/google";
import "./globals.css";

const crimsonPro = Crimson_Pro({
  variable: "--font-crimson-pro",
  subsets: ["latin"],
  display: "swap",
});

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Lake Malawi — Sustainable Land-Use Allocation",
  description:
    "RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin",
  manifest: "/manifest.json",
  icons: {
    icon: "/globe.svg",
    apple: "/icon-192.png",
  },
};

export const viewport = {
  themeColor: "#0C1B2A",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${crimsonPro.variable} ${manrope.variable}`}>
        {children}
      </body>
    </html>
  );
}
