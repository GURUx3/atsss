import React from 'react';
import { Target, Zap, Shield, Search } from 'lucide-react';

export const Insights: React.FC = () => {
    const insightItems = [
        {
            icon: Target,
            title: "Strategic Alignment",
            description: "High-density keyword mapping ensures your resume passes initial algorithmic filters for senior-level Technical Product roles."
        },
        {
            icon: Zap,
            title: "Performance Metrics",
            description: "Extraction of measurable impact statements (KPIs, percentages, growth) significantly increases profile strength ratings."
        },
        {
            icon: Shield,
            title: "Parsing Integrity",
            description: "Source code structure validated for 100% compatibility with modern Applicant Tracking Systems (Workday, Greenhouse, Lever)."
        },
        {
            icon: Search,
            title: "Market Velocity",
            description: "Live opportunity matching maps your specific technical stack to current market demands and high-priority openings."
        }
    ];

    return (
        <section className="py-4-scale">
            <div className="container-custom">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 border border-border">
                    {insightItems.map((item, index) => (
                        <div key={index} className="p-6 border-b md:border-b-0 md:border-r border-border last:border-r-0">
                            <item.icon className="h-5 w-5 mb-4 text-foreground" />
                            <h4 className="font-semibold mb-2">{item.title}</h4>
                            <p className="text-sm text-muted leading-relaxed">
                                {item.description}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
