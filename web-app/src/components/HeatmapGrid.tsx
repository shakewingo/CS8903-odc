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
  { id: 11, label: 'Rangeland', color: '#e3e2c3', defaultEsv: 184, ref: 'Costanza 2014 ratio 0.33 x anchor (Grass/Rangelands)' },
];

export const COLOR_MAP: Record<number, string> = {};
LAND_COVERS.forEach((lc) => { COLOR_MAP[lc.id] = lc.color; });

export const DEFAULT_ESV_MAP: Record<number, number> = {};
LAND_COVERS.forEach((lc) => { DEFAULT_ESV_MAP[lc.id] = lc.defaultEsv; });

type CellData = { row: number; col: number; fractions: Record<number, number> };

export function generateMockGrid(rows: number, cols: number): CellData[] {
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
      for (const k of Object.keys(fractions)) fractions[Number(k)] /= total;
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

interface HeatmapGridProps {
  esvValues: Record<number, number>;
  label: string;
}

export default function HeatmapGrid({ esvValues, label }: HeatmapGridProps) {
  const [mockGrid, setMockGrid] = useState<CellData[] | null>(null);

  useEffect(() => {
    setMockGrid(generateMockGrid(GRID_SIZE, GRID_SIZE));
  }, []);

  return (
    <div>
      <p className="text-xs font-semibold text-text-on-dark-muted uppercase tracking-wider mb-3">{label}</p>
      {mockGrid ? (
        <div className="grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)` }}>
          {mockGrid.map((cell, i) => (
            <HeatmapCell key={i} fractions={cell.fractions} esvValues={esvValues} />
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center aspect-square rounded-lg border border-border-dark">
          <Loader2 size={24} className="animate-spin text-text-on-dark-muted" />
        </div>
      )}
    </div>
  );
}
