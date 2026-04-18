'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { Maximize2, Minimize2 } from 'lucide-react';
import ExperimentToolbar from './ExperimentToolbar';
import HeatmapGrid, { DEFAULT_ESV_MAP, LAND_COVERS } from './HeatmapGrid';
import ResultsGrid from './ResultsGrid';
import AccordionPanel from './AccordionPanel';
import EsvPanel from './EsvPanel';
import ConfigPanel from './ConfigPanel';
import RankingsTable from './RankingsTable';

// Leaflet must be loaded client-side only
const MapDashboard = dynamic(() => import('./MapDashboard'), { ssr: false });

const REGEN_CROP_VALUE = Math.round(246 * 1.35);

export default function DashboardSection() {
  const [selectedExp, setSelectedExp] = useState('exp2');
  const [studyArea, setStudyArea] = useState<{ lat: number; lng: number } | null>(null);
  const [mapExpanded, setMapExpanded] = useState(true);

  const isRegenCrop = selectedExp === 'exp3';
  const esvValues: Record<number, number> = { ...DEFAULT_ESV_MAP };
  if (isRegenCrop) esvValues[5] = REGEN_CROP_VALUE;

  // Mock changes data for rankings (in production this comes from model inference)
  const mockChanges = [
    { row: 3, col: 7, fromType: 11, toType: 2, esvDelta: 54.2 },
    { row: 5, col: 2, fromType: 5, toType: 2, esvDelta: 41.8 },
    { row: 8, col: 4, fromType: 11, toType: 2, esvDelta: 38.1 },
    { row: 1, col: 9, fromType: 5, toType: 2, esvDelta: 35.6 },
    { row: 6, col: 6, fromType: 11, toType: 2, esvDelta: 29.4 },
  ];

  return (
    <section id="dashboard" className="bg-bg-dark min-h-screen">
      {/* Sticky Map — expandable/collapsible */}
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
        {/* Side-by-side: Heatmap (before) | Results (after) */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          <div>
            <HeatmapGrid esvValues={esvValues} label="Current ESV Distribution" />
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
          <ResultsGrid />
        </div>

        {/* Collapsible Accordion Panels */}
        <div className="space-y-3">
          <AccordionPanel
            title="Configuration Details"
            hint="Note: Above are currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update."
          >
            <ConfigPanel selectedExp={selectedExp} />
          </AccordionPanel>
          <AccordionPanel
            title="ESV Values & Sources"
            hint="Note: Above are currently loaded from static model configurations. Manual editing with custom retraining will be supported in a future update."
          >
            <EsvPanel esvValues={esvValues} isRegenCrop={isRegenCrop} />
          </AccordionPanel>
          <AccordionPanel title="Change Rankings">
            <RankingsTable changes={mockChanges} />
          </AccordionPanel>
        </div>
      </div>
    </section>
  );
}
