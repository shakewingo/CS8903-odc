'use client';

import { Loader2 } from 'lucide-react';

interface ResultsGridProps {
  summary: {
    cellsChanged: number;
    totalEsvGain: number;
    maxCellGain: number;
    pctAreaModified: number;
  } | null;
}

export default function ResultsGrid({ summary }: ResultsGridProps) {
  if (!summary) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 size={24} className="animate-spin text-text-on-dark-muted" />
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div className="bg-bg-dark-card rounded-xl p-3 border border-border-dark text-center">
        <p className="text-xl font-semibold text-accent-green-light font-mono">{summary.cellsChanged}</p>
        <p className="text-[10px] text-text-on-dark-muted mt-1">Cells Changed</p>
      </div>
      <div className="bg-bg-dark-card rounded-xl p-3 border border-border-dark text-center">
        <p className="text-xl font-semibold text-accent-green-light font-mono">
          {summary.totalEsvGain >= 0 ? '+' : ''}${summary.totalEsvGain.toFixed(0)}
        </p>
        <p className="text-[10px] text-text-on-dark-muted mt-1">Total ESV Gain</p>
      </div>
      <div className="bg-bg-dark-card rounded-xl p-3 border border-border-dark text-center">
        <p className="text-xl font-semibold text-accent-green-light font-mono">
          {summary.maxCellGain >= 0 ? '+' : ''}${summary.maxCellGain.toFixed(0)}
        </p>
        <p className="text-[10px] text-text-on-dark-muted mt-1">Max Cell Gain</p>
      </div>
      <div className="bg-bg-dark-card rounded-xl p-3 border border-border-dark text-center">
        <p className="text-xl font-semibold text-accent-green-light font-mono">
          {summary.pctAreaModified.toFixed(1)}%
        </p>
        <p className="text-[10px] text-text-on-dark-muted mt-1">Area Modified</p>
      </div>
    </div>
  );
}
