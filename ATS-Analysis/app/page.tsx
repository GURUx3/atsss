"use client";

import React, { useState } from 'react';
import { Hero } from '@/components/Hero';
import { FileUpload } from '@/components/FileUpload';
import { ResultsDashboard } from '@/components/ResultsDashboard';
import { Insights } from '@/components/Insights';
import { Footer } from '@/components/Footer';
import { AnalysisResult, analyzeResume, fetchJobs } from '@/lib/api';

type Job = {
    id?: string | number;
    title?: string;
    company?: string;
    location?: string;
    url?: string;
    description?: string;
    created?: string;
};

export default function Home() {
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [jobs, setJobs] = useState<Job[]>([]);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const handleFileUpload = async (file: File) => {
        setIsLoading(true);
        setErrorMessage(null);

        try {
            const response = await analyzeResume(file);

            if (!response.success || !response.data) {
                setErrorMessage(response.error || response.message || 'Analysis failed. Please try again.');
                setResult(null);
                setJobs([]);
                return;
            }

            setResult(response.data);

            if (response.data.role) {
                const jobsResponse = await fetchJobs(response.data.role);
                if (jobsResponse.success && jobsResponse.data?.jobs) {
                    setJobs(jobsResponse.data.jobs);
                } else {
                    setJobs([]);
                }
            } else {
                setJobs([]);
            }
        } catch (error) {
            console.error('Error analyzing resume:', error);
            setErrorMessage('Unexpected server error. Check backend availability and try again.');
            setResult(null);
            setJobs([]);
        } finally {
            setIsLoading(false);
        }
    };

    const resetAnalysis = () => {
        setResult(null);
        setJobs([]);
        setErrorMessage(null);
    };

    return (
        <main className="min-h-screen bg-white selection:bg-black selection:text-white">
            {!result ? (
                <>
                    <Hero />

                    <section className="py-8-scale bg-white">
                        <div className="container-custom">
                            <div className="max-w-4xl">
                                <h2 className="mb-8">Resume Analysis</h2>
                                <div className="card-enterprise">
                                    <FileUpload onFileUpload={handleFileUpload} isLoading={isLoading} />
                                    {errorMessage && (
                                        <div className="mt-6 p-4 border border-border bg-secondary text-sm font-medium">
                                            {errorMessage}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </section>

                    <Insights />
                </>
            ) : (
                <section className="py-8-scale">
                    <div className="container-custom">
                        <ResultsDashboard data={result} jobs={jobs} onBack={resetAnalysis} />
                    </div>
                </section>
            )}

            <Footer />
        </main>
    );
}

