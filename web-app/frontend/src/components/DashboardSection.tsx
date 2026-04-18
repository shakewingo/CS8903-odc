'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { Minimize2, Maximize2, Loader2 } from 'lucide-react';
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
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  const [gridLoading, setGridLoading] = useState(false);
  const [inferLoading, setInferLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // AbortControllers for cancelling in-flight requests
  const gridAbortRef = useRef<AbortController | null>(null);
  const inferAbortRef = useRef<AbortController | null>(null);

  // Load static fallback data on mount (pre-computed for default center)
  useEffect(() => {
    fetch('/data/grid_data.json')
      .then((r) => r.json())
      .then((data) => setGridData(data))
      .catch(() => console.warn('Failed to load static grid data'));

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

  // Fetch grid from backend when study area changes
  const fetchGrid = useCallback(async (lat: number, lng: number) => {
    gridAbortRef.current?.abort();
    const controller = new AbortController();
    gridAbortRef.current = controller;

    setGridLoading(true);
    setError(null);
    // Clear old results when changing study area
    setResults({});

    try {
      const res = await fetch(`${API_URL}/api/grid`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Grid generation failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setGridData(data);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') return;
      setError(e instanceof Error ? e.message : 'Grid generation failed');
    } finally {
      setGridLoading(false);
    }
  }, []);

  // Fetch inference results from backend
  const fetchInference = useCallback(async (lat: number, lng: number, experiment: string) => {
    inferAbortRef.current?.abort();
    const controller = new AbortController();
    inferAbortRef.current = controller;

    setInferLoading(true);

    try {
      const res = await fetch(`${API_URL}/api/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng, experiment }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Inference failed' }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResults((prev) => ({ ...prev, [experiment]: data }));
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'AbortError') return;
      setError(e instanceof Error ? e.message : 'Inference failed');
    } finally {
      setInferLoading(false);
    }
  }, []);

  // When study area changes, fetch new grid + inference for current experiment
  useEffect(() => {
    if (!studyArea) return;
    fetchGrid(studyArea.lat, studyArea.lng).then(() => {
      fetchInference(studyArea.lat, studyArea.lng, selectedExp);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studyArea]);

  // When experiment changes with an active study area, fetch inference
  useEffect(() => {
    if (!studyArea) return;
    // Skip if we already have results for this experiment
    if (results[selectedExp]) return;
    fetchInference(studyArea.lat, studyArea.lng, selectedExp);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedExp]);

  const isRegenCrop = selectedExp === 'exp3';
  const esvValues: Record<number, number> = { ...DEFAULT_ESV_MAP };
  if (isRegenCrop) esvValues[5] = REGEN_CROP_VALUE;

  // Extract the "before" grid
  const beforeGrid: CellFractions[][] | null = gridData
    ? gridData.grid.map((row) => row.map((cell) => cell.f))
    : null;

  // Get results for current experiment
  const currentResults = results[selectedExp] || null;
  const afterGrid = currentResults?.after || null;
  const changedCellsSet = currentResults
    ? new Set(currentResults.changedCells.map(([r, c]) => `${r},${c}`))
    : undefined;

  const isLoading = gridLoading || inferLoading;

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

      {/* Loading / Error Banner */}
      {isLoading && (
        <div className="flex items-center justify-center gap-2 py-3 bg-bg-dark-card border-b border-border-dark text-sm text-text-on-dark-muted">
          <Loader2 size={16} className="animate-spin" />
          {gridLoading ? 'Generating grid from satellite data...' : 'Running model inference...'}
        </div>
      )}
      {error && (
        <div className="flex items-center justify-center py-3 bg-red-900/30 border-b border-red-800/50 text-sm text-red-300">
          {error}
        </div>
      )}

      {/* Analysis Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Side-by-side: Before | After */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          <div>
            <HeatmapGrid esvValues={esvValues} label="Current ESV Distribution" grid={gridLoading ? null : beforeGrid} />
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
              grid={inferLoading ? null : afterGrid}
              highlightCells={changedCellsSet}
            />
            {/* Summary Stats */}
            <div className="mt-4">
              <ResultsGrid summary={inferLoading ? null : (currentResults?.summary || null)} />
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
