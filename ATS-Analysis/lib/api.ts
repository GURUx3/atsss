import axios from 'axios';

// By default we proxy `/api` requests to the backend via Next.js rewrites
// (see next.config.ts).  This means the client can just talk to the same
// origin/port it was served from.
//
// In some deployments you may still want to override the backend address
// directly; set `NEXT_PUBLIC_API_BASE_URL` to the full URL (including `/api`).
const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? 
    process.env.NEXT_PUBLIC_API_PROXY
        ? `${process.env.NEXT_PUBLIC_API_PROXY}/api`
        : '/api';

// log the chosen baseURL so developers can see which backend address is being used
console.log('api.ts using base URL ->', API_BASE_URL);

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
});

export interface CandidateInfo {
    name: string;
    email: string | null;
    phone: string | null;
    linkedin?: string | null;
    github?: string | null;
    organizations: string[];
    locations: string[];
    years_of_experience?: number | null;
    education?: string[];
    languages_spoken?: string[];
}

export interface SkillsAnalysis {
    matched_critical: string[];
    missing_critical: string[];
    matched_bonus: string[];
    missing_bonus: string[];
    all_detected_skills?: string[];
}

export interface ScoreBreakdown {
    critical_skills: number;
    bonus_skills: number;
    experience: number;
    education: number;
    completeness: number;
    total: number;
}

export interface SectionPresence {
    summary: boolean;
    experience: boolean;
    education: boolean;
    skills: boolean;
    projects: boolean;
    certifications: boolean;
    achievements: boolean;
}

export interface AnalysisResult {
    candidate: CandidateInfo;
    role: string;
    score: number;
    score_breakdown?: ScoreBreakdown;
    skills: SkillsAnalysis;
    sections?: SectionPresence;
    recommendations?: string[];
    semantic_match_score?: number;
}

export interface ApiResponse<T> {
    success: boolean;
    data: T;
    error: string | null;
    message: string;
}

export interface JobListing {
    id?: string | number;
    title?: string;
    company?: string;
    location?: string;
    url?: string;
    description?: string;
    created?: string;
}

const buildNetworkError = <T>(fallbackData: T, errorMessage: string): ApiResponse<T> => ({
    success: false,
    data: fallbackData,
    error: errorMessage,
    message: "Network error",
});

const toApiErrorResponse = <T>(error: unknown, fallbackData: T): ApiResponse<T> => {
    if (axios.isAxiosError<ApiResponse<T>>(error)) {
        if (error.response?.data) {
            return error.response.data;
        }
        return buildNetworkError(fallbackData, error.message);
    }

    if (error instanceof Error) {
        return buildNetworkError(fallbackData, error.message);
    }

    return buildNetworkError(fallbackData, "Unexpected error");
};

export const analyzeResume = async (file: File): Promise<ApiResponse<AnalysisResult>> => {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await api.post('/analyze', formData);
        return response.data;
    } catch (error: unknown) {
        const emptyResult: AnalysisResult = {
            candidate: {
                name: '',
                email: null,
                phone: null,
                organizations: [],
                locations: [],
            },
            role: '',
            score: 0,
            skills: {
                matched_critical: [],
                missing_critical: [],
                matched_bonus: [],
                missing_bonus: [],
            },
        };

        return toApiErrorResponse(error, emptyResult);
    }
};

export const fetchJobs = async (role: string): Promise<ApiResponse<{ jobs: JobListing[], count: number }>> => {
    try {
        const response = await axios.get(`${API_BASE_URL}/jobs`, {
            params: { role }
        });
        return response.data;
    } catch (error: unknown) {
        return toApiErrorResponse(error, { jobs: [], count: 0 });
    }
};
