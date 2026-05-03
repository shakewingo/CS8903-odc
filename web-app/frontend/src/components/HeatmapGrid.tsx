'use client';

import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

export const LAND_COVERS = [
  { id: 1, label: 'Water', color: '#419bdf', defaultEsv: 554, ref: 'Zuze 2013: Lake Chiuta wetland anchor' },
  { id: 2, label: 'Trees', color: '#397d49', defaultEsv: 238, ref: 'Costanza 2014 ratio 0.43 x anchor' },
  { id: 4, label: 'Flooded', color: '#7a87c6', defaultEsv: 1136, ref: 'Costanza 2014 ratio 2.05 x anchor (Inland Wetlands)' },
  { id: 5, label: 'Crops', color: '#e49635', defaultEsv: 246, ref: 'Costanza 2014 ratio 0.44 x anchor. Default is non-regenerative. +35% suggested for regenerative scenario.' },
  { id: 7, label: 'Built Area', color: '#c4281b', defaultEsv: 295, ref: 'Costanza 2014 ratio 0.53 x anchor (Urban)' },
  { id: 8, label: 'Bare Ground', color: '#a59b8f', defaultEsv: 0, ref: 'Desert — negligible ecosystem value' },
  { id: 9, label: 'Snow/Ice', color: '#cfd8dc', defaultEsv: 0, ref: 'Sentinel-2 class 9 — no ESV assigned' },
  { id: 10, label: 'No Data', color: '#777777', defaultEsv: 0, ref: 'Sentinel-2 class 10 — clouds / nodata pixels' },
  { id: 11, label: 'Rangeland', color: '#e3e2c3', defaultEsv: 184, ref: 'Costanza 2014 ratio 0.33 x anchor (Grass/Rangelands)' },
];

export const COLOR_MAP: Record<number, string> = {};
LAND_COVERS.forEach((lc) => { COLOR_MAP[lc.id] = lc.color; });

export const DEFAULT_ESV_MAP: Record<number, number> = {};
LAND_COVERS.forEach((lc) => { DEFAULT_ESV_MAP[lc.id] = lc.defaultEsv; });

export type CellFractions = Record<string, number>;

function HeatmapCell({ fractions, esvValues }: { fractions: CellFractions; esvValues: Record<number, number> }) {
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

interface HeatmapGridProps {
  esvValues: Record<number, number>;
  label: string;
  grid?: CellFractions[][] | null;
  gridSize?: number;
  highlightCells?: Set<string>;
}

export default function HeatmapGrid({ esvValues, label, grid, gridSize = 50, highlightCells }: HeatmapGridProps) {
  if (!grid) {
    return (
      <div>
        <p className="text-xs font-semibold text-text-on-dark-muted uppercase tracking-wider mb-3">{label}</p>
        <div className="flex items-center justify-center aspect-square rounded-lg border border-border-dark">
          <Loader2 size={24} className="animate-spin text-text-on-dark-muted" />
        </div>
      </div>
    );
  }

  return (
    <div>
      <p className="text-xs font-semibold text-text-on-dark-muted uppercase tracking-wider mb-3">{label}</p>
      <div className="grid gap-[1px]" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
        {grid.flatMap((row, r) =>
          row.map((fractions, c) => {
            const key = `${r},${c}`;
            const isHighlighted = highlightCells?.has(key);
            return (
              <div key={key} className="relative">
                <HeatmapCell fractions={fractions} esvValues={esvValues} />
                {isHighlighted && (
                  <div className="absolute inset-0 border border-accent-green-light rounded-[2px] pointer-events-none" />
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
