import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "VLM-SteerHarness | In-Flight XAI & Steering Control Center",
  description:
    "In-flight mechanistic observability, latent probing, and dynamic residual-stream steering harness for Vision-Language Models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-screen bg-[#09090b] text-zinc-100 flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
        <Navbar />
        <main className="flex-1 w-full max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
        <footer className="border-t border-zinc-800/80 py-4 px-6 text-center text-xs text-zinc-500">
          <div className="max-w-[1440px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>
              VLM-SteerHarness &bull; In-Flight XAI Observability &amp; Active Steering Middleware
            </span>
            <span className="font-mono text-zinc-600">
              Duke University AIPI &bull; Responsible AI / XAI &bull; Apache 2.0
            </span>
          </div>
        </footer>
      </body>
    </html>
  );
}
