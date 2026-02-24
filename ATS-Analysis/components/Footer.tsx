import React from 'react';

export const Footer: React.FC = () => {
    return (
        <footer className="py-8-scale section-border">
            <div className="container-custom">
                <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                    <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                        ATS-Analysis © {new Date().getFullYear()}
                    </p>
                    <p className="text-xs text-muted-foreground">
                        Precision Resume Intelligence Platform
                    </p>
                </div>
            </div>
        </footer>
    );
};
