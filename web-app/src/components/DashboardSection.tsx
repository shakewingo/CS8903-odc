'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Minimize2, Maximize2 } from 'lucide-react';
import ExperimentToolbar from './ExperimentToolbar';
import HeatmapGrid, { DEFAULT_ESV_MAP, LAND_COVERS } from './HeatmapGrid';
import type { CellFractions } from './HeatmapGrid';
import ResultsGrid from './ResultsGrid';
import AccordionPanel from './AccordionPanel';
import EsvPanel from './EsvPanel';
import ConfigPanel from './ConfigPanel';
import RankingsTable from './RankingsTable';

const MapDashboard = dynamic(() => import('./MapDashboard'), { ssr: false });

const REGEN_CROP_VALUE = Math.round(246 * 1.35);

interface GridData {
  rows: number;
  cols: number;
  grid: Array<Array<{ f: CellFractions; e: number; lat: number; lng: number }>>;
}

interface ResultsData {
  before: CellFractions[][];
  after: CellFractions[][];
  changedCells: number[][];
  esvChanges: Array<{ row: number; col: number; fromType: number; toType: number; esvDelta: number }>;
  summary: { cellsChanged: number; totalEsvGain: number; maxCellGain: number; pctAreaModified: number };
}

export default function DashboardSection() {
  const [selectedExp, setSelectedExp] = useState('exp2');
  const [studyArea, setStudyArea] = useState<{ lat: number; lng: number } | null>(null);
  const [mapExpanded, setMapExpanded] = useState(true);

  const [gridData, setGridData] = useState<GridData | null>(null);
  const [results, setResults] = useState<Record<string, ResultsData>>({});

  // Load grid data
  useEffect(() => {
    fetch('/data/grid_data.json')
      .then((r) => r.json())
      .then((data) => setGridData(data))
      .catch(() => console.warn('Failed to load grid data'));
  }, []);

  // Load results for all experiments
  useEffect(() => {
    const exps = ['exp1', 'exp2', 'exp3'];
    Promise.all(
      exps.map((exp) =>
        fetch(`/data/results_${exp}.json`)
          .then((r) => r.json())
          .then((data) => ({ exp, data }))
          .catch(() => null)
      )
    ).then((all) => {
      const map: Record<string, ResultsData> = {};
      for (const item of all) {
        if (item) map[item.exp] = item.data;
      }
      setResults(map);
    });
  }, []);

  const isRegenCrop = selectedExp === 'exp3';
  const esvValues: Record<number, number> = { ...DEFAULT_ESV_MAP };
  if (isRegenCrop) esvValues[5] = REGEN_CROP_VALUE;

  // Extract the "before" grid (from grid_data.json)
  const beforeGrid: CellFractions[][] | null = gridData
    ? gridData.grid.map((row) => row.map((cell) => cell.f))
    : null;

  // Get results for current experiment
  const currentResults = results[selectedExp] || null;
  const afterGrid = currentResults?.after || null;
  const changedCellsSet = currentResults
    ? new Set(currentResults.changedCells.map(([r, c]) => `${r},${c}`))
    : undefined;

  return (
    <section id="dashboard" className="bg-bg-dark min-h-screen">
      {/* Sticky Map */}
      <div className={`sticky top-0 z-20 border-b border-border-dark transition-[height] duration-300 ${mapExpanded ? 'h-[60vh]' : 'h-[30vh]'}`}>
        <MapDashboard studyArea={studyArea} onSelectArea={setStudyArea} />
        <button
          onClick={() => setMapExpanded(!mapExpanded)}
          className="absolute bottom-3 right-3 z-[1000] bg-bg-dark/85 backdrop-blur-sm text-text-on-dark-muted hover:text-text-on-dark rounded-lg p-2 border border-border-dark transition-colors"
          title={mapExpanded ? 'Collapse map' : 'Expand map'}
        >
          {mapExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>
      </div>

      {/* Experiment Toolbar */}
      <ExperimentToolbar selectedExp={selectedExp} onSelectExp={setSelectedExp} />

      {/* Analysis Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Side-by-side: Before | After */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          <div>
            <HeatmapGrid esvValues={esvValues} label="Current ESV Distribution" grid={beforeGrid} />
            {/* Shared legends */}
            <div className="flex items-center gap-3 mt-4 text-xs text-text-on-dark-muted">
              <span>Low ESV</span>
              <div
                className="h-3 flex-1 max-w-[200px] rounded-full"
                style={{ background: 'linear-gradient(to right, rgb(220,220,220), rgb(255,80,70))' }}
              />
              <span>High ESV</span>
            </div>
            <div className="flex flex-wrap gap-3 mt-3">
              {LAND_COVERS.map((lc) => (
                <div key={lc.id} className="flex items-center gap-1.5 text-[10px] text-text-on-dark-muted">
                  <div className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: lc.color }} />
                  <span>{lc.label}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <HeatmapGrid
              esvValues={esvValues}
              label="Optimized Allocation"
              grid={afterGrid}
              highlightCells={changedCellsSet}
            />
            {/* Summary Stats */}
            <div className="mt-4">
              <ResultsGrid summary={currentResults?.summary || null} />
            </div>
          </div>
        </div>

        {/* Collapsible Accordion Panels */}
        <div className="space-y-3">
          <AccordionPanel
            title="Configuration Details"
            hint="Note: below is currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update."
          >
            <ConfigPanel selectedExp={selectedExp} />
          </AccordionPanel>
          <AccordionPanel
            title="ESV Values & Sources"
            hint="Note: below is currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update."
          >
            <EsvPanel esvValues={esvValues} isRegenCrop={isRegenCrop} />
          </AccordionPanel>
          <AccordionPanel title="Change Rankings">
            <RankingsTable changes={currentResults?.esvChanges || []} />
          </AccordionPanel>
        </div>
      </div>
    </section>
  );
}
