'use client';

import { Lock } from 'lucide-react';
import { LAND_COVERS } from './HeatmapGrid';

interface EsvPanelProps {
  esvValues: Record<number, number>;
  isRegenCrop: boolean;
}

const REGEN_CROP_VALUE = Math.round(246 * 1.35);

export default function EsvPanel({ esvValues, isRegenCrop }: EsvPanelProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-text-on-dark-muted leading-relaxed">
          USD / ha / year. Costanza 2014 ratios scaled to Malawi anchor (Zuze 2013: $554/ha/yr).
        </p>
        <div className="flex items-center gap-1.5 text-xs text-text-on-dark-muted">
          <Lock size={11} />
          <span>Read-only</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {LAND_COVERS.map((lc) => {
          const currentValue = esvValues[lc.id];
          const isModified = lc.id === 5 && isRegenCrop;
          return (
            <div key={lc.id} className="flex items-center gap-2 bg-bg-dark-card rounded-lg p-2.5">
              <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: lc.color }} />
              <div className="flex-1 min-w-0">
                <p className="text-[10px] text-text-on-dark-muted truncate">{lc.label}</p>
                <p className="text-sm font-mono text-text-on-dark font-semibold">
                  ${currentValue}
                  {isModified && <span className="text-accent-gold text-[10px] ml-1">+35%</span>}
                </p>
              </div>
              <span className="info-tooltip flex-shrink-0" data-tip={lc.ref}>?</span>
            </div>
          );
        })}
      </div>

      {isRegenCrop && (
        <div className="mt-4 p-3 rounded-lg bg-accent-gold/10 border border-accent-gold/20">
          <p className="text-xs text-accent-gold leading-relaxed">
            <strong>Regenerative Agriculture Active:</strong> Crop ESV increased by 35% ($246 → ${REGEN_CROP_VALUE}).
          </p>
        </div>
      )}

      <div className="mt-3 p-3 rounded-lg bg-accent-blue/8 border border-accent-blue/15">
        <p className="text-xs text-text-on-dark-muted leading-relaxed">
          ESV values are currently loaded from static model configurations.
          Manual editing with custom retraining will be supported in a future update.
        </p>
      </div>
    </div>
  );
}
