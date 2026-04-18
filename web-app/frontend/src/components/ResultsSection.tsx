'use client';

import { useState, useEffect } from 'react';
import { ArrowRight, TrendingUp, Loader2 } from 'lucide-react';

const LC_META: Record<number, { label: string; color: string }> = {
  1: { label: 'Water', color: '#419bdf' },
  2: { label: 'Trees', color: '#397d49' },
  4: { label: 'Flooded', color: '#7a87c6' },
  5: { label: 'Crops', color: '#e49635' },
  7: { label: 'Built Area', color: '#c4281b' },
  8: { label: 'Bare Ground', color: '#a59b8f' },
  11: { label: 'Rangeland', color: '#e3e2c3' },
};

const ESV: Record<number, number> = {
  1: 554, 2: 238, 4: 1136, 5: 246, 7: 295, 8: 0, 11: 184,
};

type GridPairData = {
  before: Array<{ fractions: Record<number, number> }>;
  after: Array<{ fractions: Record<number, number> }>;
  changes: Array<{ row: number; col: number; fromType: number; toType: number; esvDelta: number }>;
};

/* Generate mock before/after grids (only called client-side) */
function makeMockGridPair(size: number): GridPairData {
  const before: GridPairData['before'] = [];
  const after: GridPairData['after'] = [];
  const changes: GridPairData['changes'] = [];

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const isNearCenter = Math.abs(r - size / 2) < size / 3 && Math.abs(c - size / 2) < size / 3;
      const frac: Record<number, number> = {};

      if (isNearCenter) {
        frac[1] = 0.3 + Math.random() * 0.2;
        frac[2] = 0.1 + Math.random() * 0.15;
        frac[11] = 0.1 + Math.random() * 0.15;
        frac[5] = Math.random() * 0.1;
      } else {
        frac[5] = 0.2 + Math.random() * 0.25;
        frac[11] = 0.2 + Math.random() * 0.2;
        frac[7] = 0.1 + Math.random() * 0.15;
        frac[2] = Math.random() * 0.1;
      }

      const total = Object.values(frac).reduce((a, b) => a + b, 0);
      for (const k of Object.keys(frac)) frac[Number(k)] /= total;

      before.push({ fractions: { ...frac } });

      const changed = Math.random() < 0.25;
      const afterFrac = { ...frac };

      if (changed) {
        const shiftFrom = frac[11] > frac[5] ? 11 : 5;
        const shiftAmt = (afterFrac[shiftFrom] || 0) * (0.3 + Math.random() * 0.4);
        afterFrac[shiftFrom] = (afterFrac[shiftFrom] || 0) - shiftAmt;
        afterFrac[2] = (afterFrac[2] || 0) + shiftAmt;

        const beforeEsv = Object.entries(frac).reduce((s, [id, f]) => s + f * (ESV[Number(id)] || 0), 0);
        const afterEsv = Object.entries(afterFrac).reduce((s, [id, f]) => s + f * (ESV[Number(id)] || 0), 0);

        changes.push({ row: r, col: c, fromType: shiftFrom, toType: 2, esvDelta: afterEsv - beforeEsv });
      }

      after.push({ fractions: afterFrac });
    }
  }

  changes.sort((a, b) => b.esvDelta - a.esvDelta);
  return { before, after, changes };
}

function MiniGrid({ cells, size, highlights, label }: {
  cells: Array<{ fractions: Record<number, number> }>;
  size: number;
  highlights?: Set<number>;
  label: string;
}) {
  return (
    <div>
      <p className="text-xs font-semibold text-text-on-dark-muted uppercase tracking-wider mb-3">{label}</p>
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}>
        {cells.map((cell, i) => {
          const totalEsv = Object.entries(cell.fractions).reduce((s, [id, f]) => s + f * (ESV[Number(id)] || 0), 0);
          const intensity = Math.min(totalEsv / 800, 1);
          const bg = `rgb(${Math.round(220 + 35 * intensity)}, ${Math.round(220 - 140 * intensity)}, ${Math.round(220 - 150 * intensity)})`;
          const isHighlighted = highlights?.has(i);

          return (
            <div key={i} className="aspect-square rounded-[2px] relative" style={{ backgroundColor: bg }}>
              {isHighlighted && (
                <div className="absolute inset-0 border-2 border-accent-green-light rounded-[2px]" />
              )}
              <div className="absolute bottom-0 left-0 right-0 h-[3px] flex">
                {Object.entries(cell.fractions)
                  .filter(([, v]) => v > 0.01)
                  .sort(([, a], [, b]) => b - a)
                  .map(([id, frac]) => (
                    <div key={id} style={{ width: `${frac * 100}%`, backgroundColor: LC_META[Number(id)]?.color || '#ccc' }} />
                  ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const GRID_SIZE = 10;

export default function ResultsSection() {
  const [data, setData] = useState<GridPairData | null>(null);

  // Generate mock data only on the client to avoid hydration mismatch
  useEffect(() => {
    setData(makeMockGridPair(GRID_SIZE));
  }, []);

  return (
    <section id="results" className="section-dark">
      <div className="section-inner">
        <p className="heading-sub text-accent-green-light mb-2">Output</p>
        <h2 className="heading-section text-text-on-dark">Optimization Results</h2>
        <p className="text-text-on-dark-muted max-w-2xl mb-10 leading-relaxed">
          Compare the original land-use distribution with the optimized
          allocation. Highlighted cells indicate where the model recommended
          changes to increase ecosystem value.
        </p>

        {!data ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 size={24} className="animate-spin text-text-on-dark-muted" />
          </div>
        ) : (
          <>
            {/* Before / After Grids */}
            <div className="grid md:grid-cols-[1fr_auto_1fr] gap-6 items-center mb-16 max-w-3xl mx-auto">
              <MiniGrid cells={data.before} size={GRID_SIZE} label="Before (Original)" />
              <div className="flex flex-col items-center gap-2 text-text-on-dark-muted">
                <ArrowRight size={24} />
                <span className="text-[10px] uppercase tracking-widest">Optimized</span>
              </div>
              <MiniGrid
                cells={data.after}
                size={GRID_SIZE}
                highlights={new Set(data.changes.map((c) => c.row * GRID_SIZE + c.col))}
                label="After (Optimized)"
              />
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12 max-w-3xl mx-auto">
              <div className="bg-bg-dark-card rounded-xl p-4 border border-border-dark text-center">
                <p className="text-2xl font-semibold text-accent-green-light font-mono">{data.changes.length}</p>
                <p className="text-xs text-text-on-dark-muted mt-1">Cells Changed</p>
              </div>
              <div className="bg-bg-dark-card rounded-xl p-4 border border-border-dark text-center">
                <p className="text-2xl font-semibold text-accent-green-light font-mono">
                  +${data.changes.reduce((s, c) => s + c.esvDelta, 0).toFixed(0)}
                </p>
                <p className="text-xs text-text-on-dark-muted mt-1">Total ESV Gain</p>
              </div>
              <div className="bg-bg-dark-card rounded-xl p-4 border border-border-dark text-center">
                <p className="text-2xl font-semibold text-accent-green-light font-mono">
                  +${data.changes[0]?.esvDelta.toFixed(0) || 0}
                </p>
                <p className="text-xs text-text-on-dark-muted mt-1">Max Cell Gain</p>
              </div>
              <div className="bg-bg-dark-card rounded-xl p-4 border border-border-dark text-center">
                <p className="text-2xl font-semibold text-accent-green-light font-mono">
                  {((data.changes.length / (GRID_SIZE * GRID_SIZE)) * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-text-on-dark-muted mt-1">Area Modified</p>
              </div>
            </div>

            {/* Top Changes Table */}
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp size={16} className="text-accent-green-light" />
                <h3 className="text-sm font-semibold text-text-on-dark">Top ESV Change Rankings</h3>
                <span className="text-xs text-text-on-dark-muted ml-auto">
                  Showing {Math.min(data.changes.length, 15)} of {data.changes.length} changed cells
                </span>
              </div>

              <div className="bg-bg-dark-card rounded-xl border border-border-dark overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Cell (Row, Col)</th>
                        <th>From</th>
                        <th></th>
                        <th>To</th>
                        <th className="text-right">ESV Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.changes.slice(0, 15).map((change, i) => (
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
            </div>
          </>
        )}
      </div>
    </section>
  );
}
