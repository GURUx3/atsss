"use client";

import React, { useMemo } from 'react';
import {
    CheckCircle2,
    XCircle,
    ArrowLeft,
    FileText,
    AlertCircle,
    ChevronRight,
    Search
} from 'lucide-react';
import { AnalysisResult } from '@/lib/api';

interface Job {
    id?: string | number;
    title?: string;
    company?: string;
    location?: string;
    url?: string;
    description?: string;
    created?: string;
}

interface ResultsDashboardProps {
    data: AnalysisResult;
    jobs: Job[];
    onBack: () => void;
}

export const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ data, jobs, onBack }) => {
    const score = Math.min(100, Math.max(0, Number(data.score) || 0));

    const criticalTotal = data.skills.matched_critical.length + data.skills.missing_critical.length;
    const criticalCoverage = criticalTotal > 0
        ? Math.round((data.skills.matched_critical.length / criticalTotal) * 100)
        : 0;

    const strengthRating = useMemo(() => {
        if (score >= 80) return "Exceptional Alignment";
        if (score >= 60) return "Moderate Fit";
        return "Critical Gap Detected";
    }, [score]);

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header */}
            <header className="flex items-center justify-between section-border pb-6">
                <div>
                    <button
                        onClick={onBack}
                        className="flex items-center text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground mb-4 transition-colors"
                    >
                        <ArrowLeft className="mr-2 h-3 w-3" />
                        Back to upload
                    </button>
                    <h2 className="text-3xl font-semibold tracking-tight">Analysis Report</h2>
                    <p className="text-muted mt-1">{data.candidate.name || 'Candidate'} • {data.role || 'Role Intelligence'}</p>
                </div>
                <div className="text-right">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Report ID</p>
                    <p className="font-mono text-xs mt-1">ATS-{Math.random().toString(36).substr(2, 9).toUpperCase()}</p>
                </div>
            </header>

            {/* Split Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* Left Column: Stats */}
                <div className="lg:col-span-4 space-y-6">
                    <div className="card-enterprise">
                        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">ATS Alignment Score</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-6xl font-semibold">{score}</span>
                            <span className="text-xl text-muted">/100</span>
                        </div>
                        <div className="mt-4 pt-4 border-t border-border">
                            <p className="font-semibold">{strengthRating}</p>
                            <p className="text-sm text-muted mt-1">Based on {criticalTotal} critical skill checkpoints.</p>
                        </div>
                    </div>

                    <div className="card-enterprise">
                        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Match Percentage</p>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{criticalCoverage}%</span>
                            <span className="text-xs text-muted">{data.skills.matched_critical.length} of {criticalTotal} matched</span>
                        </div>
                        <div className="w-full bg-secondary h-1">
                            <div
                                className="bg-foreground h-full transition-all duration-1000"
                                style={{ width: `${criticalCoverage}%` }}
                            />
                        </div>
                    </div>

                    <div className="card-enterprise">
                        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">Candidate Contact</p>
                        <div className="space-y-3">
                            <div className="flex items-center text-sm">
                                <span className="w-24 text-muted">Email</span>
                                <span className="font-medium">{data.candidate.email || '—'}</span>
                            </div>
                            <div className="flex items-center text-sm">
                                <span className="w-24 text-muted">Phone</span>
                                <span className="font-medium">{data.candidate.phone || '—'}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column: Keyword Analysis & Suggestions */}
                <div className="lg:col-span-8 space-y-6">
                    <div className="card-enterprise">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold">Keyword Intelligence</h3>
                            <span className="text-xs font-medium px-2 py-0.5 border border-border bg-secondary">
                                {data.skills.matched_critical.length + data.skills.matched_bonus.length} Matches Found
                            </span>
                        </div>

                        <div className="overflow-hidden border border-border">
                            <table className="w-full text-left border-collapse">
                                <thead className="bg-secondary">
                                    <tr>
                                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b border-border">Keyword / Skill</th>
                                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b border-border">Priority</th>
                                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground border-b border-border">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.skills.matched_critical.map((skill) => (
                                        <tr key={skill} className="border-b border-border last:border-0">
                                            <td className="px-4 py-3 text-sm font-medium">{skill}</td>
                                            <td className="px-4 py-3 text-xs font-medium uppercase text-muted-foreground">Critical</td>
                                            <td className="px-4 py-3">
                                                <div className="flex items-center text-xs font-semibold text-foreground">
                                                    <CheckCircle2 className="h-3 w-3 mr-1.5" />
                                                    Matched
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                    {data.skills.missing_critical.map((skill) => (
                                        <tr key={skill} className="border-b border-border last:border-0 bg-secondary/30">
                                            <td className="px-4 py-3 text-sm font-medium">{skill}</td>
                                            <td className="px-4 py-3 text-xs font-medium uppercase text-muted-foreground">Critical</td>
                                            <td className="px-4 py-3">
                                                <div className="flex items-center text-xs font-semibold text-muted-foreground">
                                                    <XCircle className="h-3 w-3 mr-1.5 text-foreground" />
                                                    Missing
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div className="card-enterprise">
                        <h3 className="text-lg font-semibold mb-6">Actionable Suggestions</h3>
                        <div className="space-y-4">
                            {data.skills.missing_critical.length > 0 && (
                                <div className="flex gap-4 p-4 border border-border bg-secondary/50">
                                    <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
                                    <div>
                                        <p className="font-semibold text-sm">Address Missing Critical Skills</p>
                                        <p className="text-sm text-muted mt-1 leading-relaxed">
                                            The ATS identified {data.skills.missing_critical.length} missing keywords essential for this role.
                                            Incorporate terms like "{data.skills.missing_critical[0]}" in your experience sections.
                                        </p>
                                    </div>
                                </div>
                            )}
                            <div className="flex gap-4 p-4 border border-border">
                                <Search className="h-5 w-5 shrink-0 mt-0.5" />
                                <div>
                                    <p className="font-semibold text-sm">Optimize for Technical Parsing</p>
                                    <p className="text-sm text-muted mt-1 leading-relaxed">
                                        Use standard section headers and bulleted lists. Avoid complex tables or multi-column layouts
                                        in the source PDF to ensure 100% data extraction accuracy.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Opportunities Section */}
            <section className="pt-8 border-t border-border">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h3 className="text-2xl font-semibold">Matched Opportunities</h3>
                        <p className="text-muted mt-1">Live listings aligned with your profile and the analyzed role.</p>
                    </div>
                </div>

                {jobs.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {jobs.map((job) => (
                            <a
                                key={job.id || Math.random()}
                                href={job.url || "#"}
                                target="_blank"
                                className="card-enterprise group hover:border-foreground transition-all duration-200"
                            >
                                <div className="flex justify-between items-start mb-4">
                                    <h4 className="font-semibold group-hover:underline underline-offset-4">{job.title}</h4>
                                    <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                                </div>
                                <div className="space-y-2">
                                    <p className="text-sm font-medium">{job.company}</p>
                                    <p className="text-xs text-muted font-medium uppercase tracking-wider">{job.location || 'Remote'}</p>
                                </div>
                            </a>
                        ))}
                    </div>
                ) : (
                    <div className="py-12 border border-dashed border-border text-center">
                        <p className="text-sm text-muted font-medium">No live market listings found for this specific role yet.</p>
                    </div>
                )}
            </section>
        </div>
    );
};

