'use client';

import { useState, useEffect } from 'react';
import { Cpu, Loader2, CheckCircle2, FlaskConical, Info } from 'lucide-react';

type ConfigParam = {
  key: string;
  label: string;
  hint: string;
  value: string | number;
};

const EXPERIMENTS = [
  {
    id: 'exp1',
    label: 'Experiment I',
    title: 'Pure Eco-Value',
    desc: 'Rewards purely by total ecosystem value increase. No spatial reward considered.',
    wandb: 'n2947wpo',
    color: '#2E86AB',
    config: {
      reward_scale: 1,
      spatial_scale: 0,
      w_tree: '—',
      w_crop: '—',
      w_built: '—',
      w_buf: '—',
      et_dcs_tolerance: 1,
      regen_crop: false,
      learning_rate: 3e-4,
      total_timesteps: 500_000,
      max_steps: 500,
      n_augment: 5,
    },
  },
  {
    id: 'exp2',
    label: 'Experiment II',
    title: 'Eco-Value + Spatial',
    desc: 'Rewards eco-value increase plus spatial rewards: contiguity bonus and water buffer zone penalty.',
    wandb: 'umgje44z',
    color: '#1B6B4A',
    config: {
      reward_scale: 1,
      spatial_scale: 1.0,
      w_tree: 1.0,
      w_crop: 3.0,
      w_built: 3.0,
      w_buf: 5.0,
      et_dcs_tolerance: 1,
      regen_crop: false,
      learning_rate: 1e-4,
      total_timesteps: 500_000,
      max_steps: 500,
      n_augment: 5,
    },
  },
  {
    id: 'exp3',
    label: 'Experiment III',
    title: 'Spatial + Regenerative',
    desc: 'Same as Exp II, but with regenerative agriculture multiplier (1.35×) applied to crop eco-value.',
    wandb: '7ybfz89t',
    color: '#C49A2A',
    config: {
      reward_scale: 1,
      spatial_scale: 1.0,
      w_tree: 1.0,
      w_crop: 3.0,
      w_built: 3.0,
      w_buf: 5.0,
      et_dcs_tolerance: 1,
      regen_crop: true,
      learning_rate: 1e-4,
      total_timesteps: 500_000,
      max_steps: 500,
      n_augment: 5,
    },
  },
];

const CONFIG_PARAMS: { key: string; label: string; hint: string }[] = [
  { key: 'reward_scale', label: 'Reward Scale', hint: 'Divides raw reward to normalize gradient magnitudes' },
  { key: 'spatial_scale', label: 'Spatial Scale', hint: 'Overall weight for spatial reward component; 0 disables spatial rewards entirely' },
  { key: 'w_tree', label: 'w_tree', hint: 'Priority weight for tree contiguity bonus (convolution-based)' },
  { key: 'w_crop', label: 'w_crop', hint: 'Priority weight for crop contiguity bonus' },
  { key: 'w_built', label: 'w_built', hint: 'Priority weight for built-area contiguity bonus' },
  { key: 'w_buf', label: 'w_buf', hint: 'Priority weight for water-buffer zone penalty' },
  { key: 'et_dcs_tolerance', label: 'ET Tolerance', hint: 'Evapotranspiration decrease tolerance for early episode termination' },
  { key: 'regen_crop', label: 'Regen. Crops', hint: 'Whether regenerative agriculture multiplier (1.35×) is applied to crop ESV' },
  { key: 'learning_rate', label: 'Learning Rate', hint: 'Adam optimizer learning rate for PPO' },
  { key: 'total_timesteps', label: 'Total Timesteps', hint: 'Total training timesteps for PPO agent' },
  { key: 'max_steps', label: 'Max Steps', hint: 'Maximum steps per episode before termination' },
  { key: 'n_augment', label: 'Data Augment', hint: 'Number of data augmentation rounds during training (rotation/flip)' },
];

function formatValue(value: string | number | boolean): string {
  if (typeof value === 'boolean') return value ? 'Yes (1.35×)' : 'No';
  if (typeof value === 'string') return value;
  if (value >= 1000) return value.toLocaleString();
  if (value < 0.01 && value > 0) return value.toExponential();
  return String(value);
}

export default function ModelSection() {
  const [selectedExp, setSelectedExp] = useState('exp2');
  const [loaded, setLoaded] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Notify HeatmapSection when experiment changes
  useEffect(() => {
    window.dispatchEvent(new CustomEvent('exp-change', { detail: selectedExp }));
  }, [selectedExp]);

  const handleLoad = () => {
    setLoading(true);
    setTimeout(() => {
      setLoaded(selectedExp);
      setLoading(false);
    }, 2000);
  };

  const handleSelect = (id: string) => {
    setSelectedExp(id);
    setLoaded(null);
  };

  const activeExp = EXPERIMENTS.find((e) => e.id === selectedExp)!;
  const configRows: ConfigParam[] = CONFIG_PARAMS.map((p) => ({
    ...p,
    value: activeExp.config[p.key as keyof typeof activeExp.config] as string | number,
  }));

  return (
    <section id="model" className="section-cream">
      <div className="section-inner">
        <p className="heading-sub mb-2">Optimization Engine</p>
        <h2 className="heading-section text-text-primary">
          Load Model for Inference
        </h2>
        <p className="text-text-secondary max-w-2xl mb-10 leading-relaxed">
          Select a pre-trained experiment configuration to load. Each model
          was trained with different reward structures to compare optimization
          strategies.
        </p>

        {/* Experiment Cards */}
        <div className="grid md:grid-cols-3 gap-5 mb-10">
          {EXPERIMENTS.map((exp) => (
            <button
              key={exp.id}
              onClick={() => handleSelect(exp.id)}
              className={`text-left rounded-xl p-6 transition-all border-2 ${
                selectedExp === exp.id
                  ? 'bg-bg-card border-accent-green shadow-lg'
                  : 'bg-bg-card border-border-warm hover:border-accent-green/40'
              }`}
              style={{
                borderTopColor: selectedExp === exp.id ? exp.color : undefined,
                borderTopWidth: selectedExp === exp.id ? '3px' : undefined,
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <span
                  className="text-xs font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full"
                  style={{
                    backgroundColor: `${exp.color}20`,
                    color: exp.color,
                  }}
                >
                  {exp.label}
                </span>
                {loaded === exp.id && (
                  <CheckCircle2 size={16} className="text-accent-green" />
                )}
              </div>
              <h3 className="font-semibold text-text-primary text-lg mb-2">
                {exp.title}
              </h3>
              <p className="text-sm text-text-secondary leading-relaxed">
                {exp.desc}
              </p>
            </button>
          ))}
        </div>

        {/* Config & Load */}
        <div className="grid lg:grid-cols-[1fr_auto] gap-6 items-start">
          {/* Config display */}
          <div className="bg-bg-card rounded-xl border border-border-warm p-6">
            <div className="flex items-center gap-2 mb-5">
              <FlaskConical size={16} className="text-accent-green" />
              <h3 className="text-sm font-semibold text-text-primary">
                Configuration — {activeExp.title}
              </h3>
              <a
                href={`https://wandb.ai/shakewin/CS8903-odc/runs/${activeExp.wandb}/overview`}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto text-xs text-accent-green hover:text-accent-green-light transition-colors"
              >
                View on W&B →
              </a>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {configRows.map((param) => (
                <div key={param.key} className="bg-bg-warm rounded-lg p-3 relative group">
                  <div className="flex items-center gap-1 mb-1">
                    <p className="text-[10px] font-medium text-text-muted uppercase tracking-wider">{param.label}</p>
                    <div className="relative">
                      <Info size={10} className="text-text-muted cursor-help" />
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50">
                        <div className="bg-bg-dark text-text-on-dark text-[11px] px-3 py-2 rounded-lg shadow-lg whitespace-normal w-max max-w-[220px] leading-relaxed">
                          {param.hint}
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="text-base font-semibold text-text-primary font-mono">
                    {formatValue(param.value)}
                  </p>
                </div>
              ))}
            </div>

            <div className="mt-5 p-3 rounded-lg bg-accent-blue/8 border border-accent-blue/15">
              <p className="text-xs text-text-secondary leading-relaxed">
                <strong className="text-text-primary">Phase 1 (current):</strong> Static pre-trained models with fixed configurations.
                Retraining with manual config values will be supported in a future update (~15 min training time + model caching).
              </p>
            </div>
          </div>

          {/* Load button */}
          <div className="flex flex-col items-center gap-3">
            <button
              onClick={handleLoad}
              disabled={loading || loaded === selectedExp}
              className={`w-48 py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all ${
                loaded === selectedExp
                  ? 'bg-accent-green/10 text-accent-green border-2 border-accent-green/30 cursor-default'
                  : loading
                    ? 'bg-accent-green/60 text-white cursor-wait'
                    : 'bg-accent-green text-white hover:bg-accent-green/90 shadow-lg shadow-accent-green/20'
              }`}
            >
              {loaded === selectedExp ? (
                <>
                  <CheckCircle2 size={16} />
                  Model Loaded
                </>
              ) : loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <Cpu size={16} />
                  Load Model
                </>
              )}
            </button>
            {loaded === selectedExp && (
              <p className="text-xs text-accent-green">
                Ready for inference
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
