"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Cpu,
  Layers,
  ShieldCheck,
  WifiOff,
} from "lucide-react";
import { fetchHealth } from "@/lib/api";
import type { HealthStatus } from "@/types/harness";

const NAV_ITEMS = [
  { href: "/", label: "Steering Studio", icon: Activity },
  { href: "/benchmarks", label: "Benchmarks & Efficacy", icon: BarChart3 },
  { href: "/architecture", label: "Architecture & Calibration", icon: Layers },
];

export function Navbar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const res = await fetchHealth();
        if (active) {
          setHealth(res);
          setOffline(false);
        }
      } catch {
        if (active) {
          setHealth(null);
          setOffline(true);
        }
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-zinc-800/80 bg-[#09090b]/85 backdrop-blur-md">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-indigo-500/20 to-emerald-500/20 border border-indigo-500/30 flex items-center justify-center shadow-inner group-hover:border-indigo-400/50 transition-colors">
            <ShieldCheck className="h-5 w-5 text-indigo-400" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-semibold tracking-tight text-zinc-100 text-sm sm:text-base">
                VLM-Steer
              </span>
              <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/25">
                v1.0
              </span>
            </div>
            <span className="text-[11px] text-zinc-500 hidden md:block">
              In-Flight Mechanistic Observability &amp; Active Steering
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        <nav className="flex items-center gap-1 bg-zinc-900/80 p-1 rounded-lg border border-zinc-800/80">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium transition-all ${
                  isActive
                    ? "bg-zinc-800 text-zinc-100 shadow-sm border border-zinc-700/60"
                    : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Runtime Status Indicator */}
        <div className="hidden lg:flex items-center">
          {health ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <Cpu className="h-3.5 w-3.5 text-emerald-400" />
              <span className="font-mono text-emerald-300">
                {health.model_id} &bull; {health.num_layers}L/{health.hidden_dim}D
              </span>
            </div>
          ) : offline ? (
            <div
              title="Start Python API with: uvicorn api.server:app --port 8000"
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-rose-500/10 border border-rose-500/25 text-xs text-rose-300"
            >
              <WifiOff className="h-3.5 w-3.5 text-rose-400" />
              <span className="font-mono">API Offline (Port 8000)</span>
            </div>
          ) : (
            <div className="px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-xs font-mono text-zinc-500">
              Connecting...
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
