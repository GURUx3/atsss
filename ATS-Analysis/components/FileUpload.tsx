"use client";

import React, { useMemo, useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, FileText, Loader2, Upload, X } from 'lucide-react';
import { cn } from '@/lib/utils';

const ACCEPTED_MIME_TYPES = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
];
const ACCEPTED_EXTENSIONS = ['pdf', 'docx'];
const MAX_FILE_SIZE_MB = 10;

interface FileUploadProps {
    onFileUpload: (file: File) => void;
    isLoading: boolean;
}

const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 MB';
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
};

export const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, isLoading }) => {
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const fileMeta = useMemo(() => {
        if (!file) return null;
        const extension = file.name.split('.').pop()?.toUpperCase() ?? 'FILE';
        return {
            extension,
            size: formatFileSize(file.size),
        };
    }, [file]);

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            validateAndSetFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            validateAndSetFile(e.target.files[0]);
        }
    };

    const validateAndSetFile = (uploadedFile: File) => {
        const extension = uploadedFile.name.split('.').pop()?.toLowerCase() ?? '';
        const hasValidType = ACCEPTED_MIME_TYPES.includes(uploadedFile.type) || ACCEPTED_EXTENSIONS.includes(extension);

        if (!hasValidType) {
            setFile(null);
            setError('Unsupported file type. Please upload a PDF or DOCX.');
            return;
        }

        if (uploadedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
            setFile(null);
            setError(`File is too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`);
            return;
        }

        setError(null);
        setFile(uploadedFile);
    };

    const handleSubmit = () => {
        if (!file) {
            setError('Select a resume file before starting analysis.');
            return;
        }
        onFileUpload(file);
    };

    const clearFile = () => {
        setFile(null);
        setError(null);
        if (inputRef.current) {
            inputRef.current.value = '';
        }
    };

    const openFilePicker = () => inputRef.current?.click();

    return (
        <div className="w-full space-y-6">
            <div
                className={cn(
                    'relative border-2 border-dashed transition-all duration-200 py-12 px-6 text-center',
                    dragActive && !file ? 'border-foreground bg-secondary' : 'border-border bg-white',
                    file ? 'border-foreground bg-white' : 'hover:border-foreground'
                )}
                style={{ borderRadius: '0px' }}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    ref={inputRef}
                    type="file"
                    className="hidden"
                    accept=".pdf,.docx"
                    onChange={handleChange}
                    disabled={isLoading}
                />

                <div className="flex flex-col items-center justify-center space-y-4">
                    {file ? (
                        <div className="flex flex-col items-center">
                            <CheckCircle2 className="h-8 w-8 text-foreground mb-4" />
                            <h3 className="text-lg font-semibold">{file.name}</h3>
                            <p className="text-muted text-sm mb-6">
                                {fileMeta?.extension} • {fileMeta?.size}
                            </p>
                            <div className="flex items-center gap-4">
                                <button
                                    onClick={clearFile}
                                    disabled={isLoading}
                                    className="btn-outline h-10"
                                >
                                    <X className="mr-2 h-4 w-4" />
                                    Remove
                                </button>
                                <button
                                    onClick={handleSubmit}
                                    disabled={isLoading}
                                    className="btn-primary h-10 flex items-center"
                                >
                                    {isLoading ? (
                                        <>
                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        'Start Analysis'
                                    )}
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center">
                            <Upload className="h-10 w-10 text-muted-foreground mb-4" />
                            <h3 className="text-xl font-semibold mb-2">Upload Resume</h3>
                            <p className="text-muted max-w-sm mb-6">
                                Drag and drop your file here, or click to browse. Supporting PDF and DOCX up to 10MB.
                            </p>
                            <button
                                onClick={openFilePicker}
                                className="btn-outline"
                            >
                                Browse Files
                            </button>
                        </div>
                    )}
                </div>
            </div>

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-t border-border pt-4">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    PDF, DOCX accepted
                </p>
                {error && (
                    <p className="inline-flex items-center text-xs font-semibold text-foreground">
                        <AlertCircle className="mr-2 h-3 w-3" />
                        {error}
                    </p>
                )}
            </div>
        </div>
    );
};

