'use client';

import { ArrowRight, TrendingUp } from 'lucide-react';

const LC_META: Record<number, { label: string; color: string }> = {
  1: { label: 'Water', color: '#419bdf' },
  2: { label: 'Trees', color: '#397d49' },
  4: { label: 'Flooded', color: '#7a87c6' },
  5: { label: 'Crops', color: '#e49635' },
  7: { label: 'Built Area', color: '#c4281b' },
  8: { label: 'Bare Ground', color: '#a59b8f' },
  11: { label: 'Rangeland', color: '#e3e2c3' },
};

interface Change {
  row: number;
  col: number;
  fromType: number;
  toType: number;
  esvDelta: number;
}

interface RankingsTableProps {
  changes: Change[];
}

export default function RankingsTable({ changes }: RankingsTableProps) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp size={14} className="text-accent-green-light" />
        <span className="text-xs font-semibold text-text-on-dark">Top ESV Change Rankings</span>
        <span className="text-[10px] text-text-on-dark-muted ml-auto">
          Showing {Math.min(changes.length, 15)} of {changes.length} changed cells
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="results-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Cell</th>
              <th>From</th>
              <th></th>
              <th>To</th>
              <th className="text-right">ESV Change</th>
            </tr>
          </thead>
          <tbody>
            {changes.slice(0, 15).map((change, i) => (
              <tr key={i}>
                <td className="text-text-on-dark-muted font-mono text-xs">#{i + 1}</td>
                <td className="text-text-on-dark font-mono text-xs">({change.row}, {change.col})</td>
                <td>
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: LC_META[change.fromType]?.color }} />
                    <span className="text-text-on-dark text-xs">{LC_META[change.fromType]?.label}</span>
                  </div>
                </td>
                <td className="text-text-on-dark-muted"><ArrowRight size={12} /></td>
                <td>
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: LC_META[change.toType]?.color }} />
                    <span className="text-text-on-dark text-xs">{LC_META[change.toType]?.label}</span>
                  </div>
                </td>
                <td className="text-right">
                  <span className="text-accent-green-light font-semibold text-xs font-mono">+${change.esvDelta.toFixed(1)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
