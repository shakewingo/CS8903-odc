'use client';

import { useState, useEffect } from 'react';
import { Loader2, Lock } from 'lucide-react';

/* ── Land cover types with defaults from config.py ── */
const LAND_COVERS = [
  { id: 1, label: 'Water', color: '#419bdf', defaultEsv: 554, ref: 'Zuze 2013: Lake Chiuta wetland anchor' },
  { id: 2, label: 'Trees', color: '#397d49', defaultEsv: 238, ref: 'Costanza 2014 ratio 0.43 × anchor' },
  { id: 4, label: 'Flooded', color: '#7a87c6', defaultEsv: 1136, ref: 'Costanza 2014 ratio 2.05 × anchor (Inland Wetlands)' },
  { id: 5, label: 'Crops', color: '#e49635', defaultEsv: 246, ref: 'Costanza 2014 ratio 0.44 × anchor. Default is non-regenerative. +35% suggested for regenerative scenario.' },
  { id: 7, label: 'Built Area', color: '#c4281b', defaultEsv: 295, ref: 'Costanza 2014 ratio 0.53 × anchor (Urban)' },
  { id: 8, label: 'Bare Ground', color: '#a59b8f', defaultEsv: 0, ref: 'Desert — negligible ecosystem value' },
  { id: 11, label: 'Rangeland', color: '#e3e2c3', defaultEsv: 184, ref: 'Costanza 2014 ratio 0.33 × anchor (Grass/Rangelands)' },
];

const COLOR_MAP: Record<number, string> = {};
LAND_COVERS.forEach((lc) => { COLOR_MAP[lc.id] = lc.color; });

const DEFAULT_ESV_MAP: Record<number, number> = {};
LAND_COVERS.forEach((lc) => { DEFAULT_ESV_MAP[lc.id] = lc.defaultEsv; });

const REGEN_CROP_VALUE = Math.round(246 * 1.35); // 332

type CellData = { row: number; col: number; fractions: Record<number, number> };

/* ── Generate mock grid (only called client-side) ── */
function generateMockGrid(rows: number, cols: number): CellData[] {
  const grid: CellData[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const isNearCenter = Math.abs(r - rows / 2) < rows / 4 && Math.abs(c - cols / 2) < cols / 4;
      const isEdge = r < 2 || r > rows - 3 || c < 2 || c > cols - 3;

      const fractions: Record<number, number> = {};
      if (isNearCenter) {
        fractions[1] = 0.3 + Math.random() * 0.3;
        fractions[2] = 0.1 + Math.random() * 0.2;
        fractions[4] = Math.random() * 0.15;
        fractions[5] = Math.random() * 0.1;
        fractions[11] = Math.random() * 0.1;
      } else if (isEdge) {
        fractions[5] = 0.2 + Math.random() * 0.3;
        fractions[7] = 0.1 + Math.random() * 0.2;
        fractions[11] = 0.2 + Math.random() * 0.2;
        fractions[2] = Math.random() * 0.1;
        fractions[8] = Math.random() * 0.1;
      } else {
        fractions[2] = 0.15 + Math.random() * 0.2;
        fractions[5] = 0.1 + Math.random() * 0.2;
        fractions[11] = 0.15 + Math.random() * 0.2;
        fractions[1] = Math.random() * 0.15;
        fractions[7] = Math.random() * 0.1;
      }

      const total = Object.values(fractions).reduce((a, b) => a + b, 0);
      for (const k of Object.keys(fractions)) {
        fractions[Number(k)] = fractions[Number(k)] / total;
      }
      grid.push({ row: r, col: c, fractions });
    }
  }
  return grid;
}

function HeatmapCell({ fractions, esvValues }: { fractions: Record<number, number>; esvValues: Record<number, number> }) {
  const totalEsv = Object.entries(fractions).reduce(
    (sum, [id, frac]) => sum + frac * (esvValues[Number(id)] || 0), 0
  );
  const intensity = Math.min(totalEsv / 800, 1);
  const bgColor = `rgb(${Math.round(220 + 35 * intensity)}, ${Math.round(220 - 140 * intensity)}, ${Math.round(220 - 150 * intensity)})`;

  const sortedFracs = Object.entries(fractions)
    .filter(([, v]) => v > 0.01)
    .sort(([, a], [, b]) => b - a);

  return (
    <div className="heatmap-cell" style={{ backgroundColor: bgColor }} title={`ESV: $${totalEsv.toFixed(0)}/ha/yr`}>
      <div className="fraction-bar">
        {sortedFracs.map(([id, frac]) => (
          <div key={id} style={{ width: `${frac * 100}%`, backgroundColor: COLOR_MAP[Number(id)] || '#ccc' }} />
        ))}
      </div>
    </div>
  );
}

const GRID_SIZE = 10;

export default function HeatmapSection() {
  const [mockGrid, setMockGrid] = useState<CellData[] | null>(null);
  const [selectedExp, setSelectedExp] = useState('exp2');

  // Generate mock data only on the client to avoid hydration mismatch
  useEffect(() => {
    setMockGrid(generateMockGrid(GRID_SIZE, GRID_SIZE));
  }, []);

  // Listen for experiment changes from ModelSection
  useEffect(() => {
    const handler = (e: Event) => {
      setSelectedExp((e as CustomEvent).detail);
    };
    window.addEventListener('exp-change', handler);
    return () => window.removeEventListener('exp-change', handler);
  }, []);

  const isRegenCrop = selectedExp === 'exp3';
  const esvValues: Record<number, number> = { ...DEFAULT_ESV_MAP };
  if (isRegenCrop) esvValues[5] = REGEN_CROP_VALUE;

  return (
    <section id="heatmap" className="section-dark">
      <div className="section-inner">
        <p className="heading-sub text-accent-green-light mb-2">Ecological Assessment</p>
        <h2 className="heading-section text-text-on-dark">
          Ecosystem Value Heatmap
        </h2>
        <p className="text-text-on-dark-muted max-w-2xl mb-10 leading-relaxed">
          Each cell displays horizontal bars representing land-use type fractions.
          The background color encodes total ecosystem value —
          <span className="text-red-400"> red</span> for high,
          <span className="text-gray-300"> white</span> for low.
        </p>

        <div className="grid lg:grid-cols-[1fr_340px] gap-10 items-start">
          {/* Heatmap Grid */}
          <div>
            {mockGrid ? (
              <div
                className="grid gap-[3px] max-w-[560px]"
                style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)` }}
              >
                {mockGrid.map((cell, i) => (
                  <HeatmapCell key={i} fractions={cell.fractions} esvValues={esvValues} />
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center max-w-[560px] aspect-square rounded-lg border border-border-dark">
                <Loader2 size={24} className="animate-spin text-text-on-dark-muted" />
              </div>
            )}

            {/* Color legend */}
            <div className="flex items-center gap-3 mt-6 text-xs text-text-on-dark-muted">
              <span>Low ESV</span>
              <div
                className="h-3 flex-1 max-w-[200px] rounded-full"
                style={{ background: 'linear-gradient(to right, rgb(220,220,220), rgb(255,80,70))' }}
              />
              <span>High ESV</span>
            </div>

            {/* Land cover bar legend */}
            <div className="flex flex-wrap gap-3 mt-4">
              {LAND_COVERS.map((lc) => (
                <div key={lc.id} className="flex items-center gap-1.5 text-xs text-text-on-dark-muted">
                  <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: lc.color }} />
                  <span>{lc.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ESV Display Panel */}
          <div className="bg-bg-dark-card rounded-xl border border-border-dark p-6">
            <div className="flex items-center justify-between mb-5">
              <h3 className="font-semibold text-text-on-dark text-sm">Ecosystem Service Values</h3>
              <div className="flex items-center gap-1.5 text-xs text-text-on-dark-muted">
                <Lock size={11} />
                <span>Read-only</span>
              </div>
            </div>

            <p className="text-xs text-text-on-dark-muted mb-5 leading-relaxed">
              USD / ha / year. Costanza 2014 ratios scaled to Malawi anchor
              (Zuze 2013: Lake Chiuta wetland $554/ha/yr).
            </p>

            <div className="space-y-3">
              {LAND_COVERS.map((lc) => {
                const currentValue = esvValues[lc.id];
                const isModified = lc.id === 5 && isRegenCrop;
                return (
                  <div key={lc.id} className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: lc.color }} />
                    <span className="text-xs text-text-on-dark w-24 flex-shrink-0">{lc.label}</span>
                    <div className="relative flex-1">
                      <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-xs text-text-on-dark-muted">$</span>
                      <input
                        type="number"
                        value={currentValue}
                        readOnly
                        className="w-full pl-6 pr-2 py-1.5 rounded-md border border-border-dark bg-bg-dark text-sm text-text-on-dark font-mono cursor-not-allowed opacity-70"
                      />
                    </div>
                    {isModified && (
                      <span className="text-[10px] text-accent-gold font-medium whitespace-nowrap">+35%</span>
                    )}
                    <span className="info-tooltip flex-shrink-0" data-tip={lc.ref}>?</span>
                  </div>
                );
              })}
            </div>

            {isRegenCrop && (
              <div className="mt-5 p-3 rounded-lg bg-accent-gold/10 border border-accent-gold/20">
                <p className="text-xs text-accent-gold leading-relaxed">
                  <strong>Regenerative Agriculture Active:</strong> Crop ESV automatically increased
                  by 35% ($246 → ${REGEN_CROP_VALUE}) based on Experiment III configuration.
                </p>
              </div>
            )}

            <div className="mt-4 p-3 rounded-lg bg-accent-blue/8 border border-accent-blue/15">
              <p className="text-xs text-text-on-dark-muted leading-relaxed">
                ESV values are currently loaded from static model configurations.
                Manual editing with custom retraining will be supported in a future update.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
