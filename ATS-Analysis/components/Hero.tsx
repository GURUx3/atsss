import React from 'react';

export const Hero: React.FC = () => {
    return (
        <section className="py-8-scale section-border">
            <div className="container-custom">
                <div className="max-w-3xl">
                    <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-4">
                        ATS-Analysis Platform
                    </p>
                    <h1 className="mb-6">
                        Enterprise-grade resume intelligence
                        and ATS calibration.
                    </h1>
                    <p className="text-lg text-muted mb-8 leading-relaxed">
                        A serious technical tool for processing resumes, exposing skill gaps,
                        and evaluating applicant strength against production-grade criteria.
                    </p>
                    <div className="flex items-center gap-4">
                        <button className="btn-primary">
                            Analyze Resume
                        </button>
                    </div>
                </div>

            </div>
        </section>
    );
};
