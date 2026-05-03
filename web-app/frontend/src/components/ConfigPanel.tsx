'use client';

import { Info, FlaskConical, Lock } from 'lucide-react';

const EXPERIMENTS_CONFIG: Record<string, { title: string; wandb: string; config: Record<string, string | number | boolean> }> = {
  exp1: {
    title: 'Pure Eco-Value',
    wandb: 'n2947wpo',
    config: { reward_scale: 1, spatial_scale: 0, w_tree: '—', w_crop: '—', w_built: '—', w_buf: '—', et_dcs_tolerance: 1, regen_crop: false, learning_rate: 3e-4, total_timesteps: 500_000, max_steps: 500, n_augment: 5 },
  },
  exp2: {
    title: 'Eco-Value + Spatial',
    wandb: 'umgje44z',
    config: { reward_scale: 1, spatial_scale: 1.0, w_tree: 1.0, w_crop: 3.0, w_built: 3.0, w_buf: 5.0, et_dcs_tolerance: 1, regen_crop: false, learning_rate: 1e-4, total_timesteps: 500_000, max_steps: 500, n_augment: 5 },
  },
  exp3: {
    title: 'Spatial + Regenerative',
    wandb: '7ybfz89t',
    config: { reward_scale: 1, spatial_scale: 1.0, w_tree: 1.0, w_crop: 3.0, w_built: 3.0, w_buf: 5.0, et_dcs_tolerance: 1, regen_crop: true, learning_rate: 1e-4, total_timesteps: 500_000, max_steps: 500, n_augment: 5 },
  },
};

const CONFIG_PARAMS = [
  { key: 'reward_scale', label: 'Reward Scale', hint: 'Divides raw reward to normalize gradient magnitudes' },
  { key: 'spatial_scale', label: 'Spatial Scale', hint: 'Overall weight for spatial reward component; 0 disables spatial rewards' },
  { key: 'w_tree', label: 'w_tree', hint: 'Priority weight for tree contiguity bonus' },
  { key: 'w_crop', label: 'w_crop', hint: 'Priority weight for crop contiguity bonus' },
  { key: 'w_built', label: 'w_built', hint: 'Priority weight for built-area contiguity bonus' },
  { key: 'w_buf', label: 'w_buf', hint: 'Priority weight for water-buffer zone penalty' },
  { key: 'et_dcs_tolerance', label: 'ET Tolerance', hint: 'Evapotranspiration decrease tolerance for early termination' },
  { key: 'regen_crop', label: 'Regen. Crops', hint: 'Whether regenerative agriculture multiplier (1.35x) is applied' },
  { key: 'learning_rate', label: 'Learning Rate', hint: 'Adam optimizer learning rate for PPO' },
  { key: 'total_timesteps', label: 'Total Timesteps', hint: 'Total training timesteps for PPO agent' },
  { key: 'max_steps', label: 'Max Steps', hint: 'Maximum steps per episode before termination' },
  { key: 'n_augment', label: 'Data Augment', hint: 'Number of data augmentation rounds (rotation/flip)' },
];

function formatValue(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'Yes (1.35x)' : 'No';
  if (typeof value === 'string') return value;
  if (value >= 1000) return value.toLocaleString();
  if (value < 0.01 && value > 0) return value.toExponential();
  return String(value);
}

interface ConfigPanelProps {
  selectedExp: string;
}

export default function ConfigPanel({ selectedExp }: ConfigPanelProps) {
  const exp = EXPERIMENTS_CONFIG[selectedExp] || EXPERIMENTS_CONFIG.exp2;

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <FlaskConical size={14} className="text-accent-green" />
        <span className="text-xs font-semibold text-text-on-dark">{exp.title}</span>
        <div className="flex items-center gap-1.5 text-xs text-text-on-dark-muted ml-auto">
          <Lock size={11} />
          <span>Read-only</span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
        {CONFIG_PARAMS.map((param) => (
          <div key={param.key} className="bg-bg-dark-card rounded-lg p-2.5 relative group">
            <div className="flex items-center gap-1 mb-1">
              <p className="text-[10px] font-medium text-text-on-dark-muted uppercase tracking-wider">{param.label}</p>
              <div className="relative">
                <Info size={10} className="text-text-on-dark-muted cursor-help" />
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50">
                  <div className="bg-bg-dark text-text-on-dark text-[11px] px-3 py-2 rounded-lg shadow-lg whitespace-normal w-max max-w-[220px] leading-relaxed">
                    {param.hint}
                  </div>
                </div>
              </div>
            </div>
            <p className="text-sm font-semibold text-text-on-dark font-mono">
              {formatValue(exp.config[param.key])}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
