"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { Activity, Github, Layers3, ShieldCheck, Sparkles } from 'lucide-react';

export const Navbar = () => {
    const navItems = ['How It Works', 'Scoring Logic', 'Market Signals'];

    return (
        <header className="sticky top-0 z-50 border-b border-slate-200/60 bg-white/70 backdrop-blur-xl">
            <div className="mx-auto flex h-[4.5rem] w-full max-w-7xl items-center justify-between gap-6 px-4 sm:px-6 lg:px-8">
                <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-900 via-blue-700 to-teal-600 text-white shadow-lg shadow-blue-900/20">
                        <Layers3 className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                        <p className="truncate text-base font-extrabold text-slate-900 sm:text-lg">ATS Analyzer Suite</p>
                        <p className="truncate text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-blue-700/80">
                            Resume Intelligence Platform
                        </p>
                    </div>
                </div>

                <nav className="hidden items-center gap-6 md:flex">
                    {navItems.map((item) => (
                        <button
                            key={item}
                            type="button"
                            className="text-sm font-semibold text-slate-600 transition-colors hover:text-slate-950"
                        >
                            {item}
                        </button>
                    ))}
                </nav>

                <div className="flex items-center gap-2 sm:gap-3">
                    <div className="hidden items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-bold text-emerald-800 lg:flex">
                        <ShieldCheck className="h-3.5 w-3.5" />
                        ATS Calibrated
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="hidden border border-transparent text-slate-600 hover:border-slate-200 hover:bg-white sm:flex"
                    >
                        <Activity className="mr-2 h-4 w-4" />
                        Changelog
                    </Button>
                    <Button
                        variant="default"
                        size="sm"
                        className="rounded-full bg-gradient-to-r from-blue-900 via-blue-800 to-teal-600 px-4 text-white shadow-md shadow-blue-900/25 hover:from-blue-800 hover:to-teal-500"
                    >
                        <Sparkles className="mr-2 h-4 w-4" />
                        Dashboard
                    </Button>
                    <Button
                        variant="outline"
                        size="icon-sm"
                        className="hidden border-slate-300 bg-white text-slate-700 hover:bg-slate-100 sm:inline-flex"
                        aria-label="GitHub Repository"
                    >
                        <Github className="h-4 w-4" />
                    </Button>
                </div>
            </div>
        </header>
    );
};
