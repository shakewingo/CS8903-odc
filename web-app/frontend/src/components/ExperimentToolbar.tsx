'use client';

const EXPERIMENTS = [
  { id: 'exp1', label: 'Exp I', title: 'Pure Eco-Value', color: '#2E86AB' },
  { id: 'exp2', label: 'Exp II', title: 'Eco-Value + Spatial', color: '#1B6B4A' },
  { id: 'exp3', label: 'Exp III', title: 'Spatial + Regenerative', color: '#C49A2A' },
];

interface ExperimentToolbarProps {
  selectedExp: string;
  onSelectExp: (id: string) => void;
}

export default function ExperimentToolbar({ selectedExp, onSelectExp }: ExperimentToolbarProps) {
  return (
    <div className="flex items-center gap-2 px-4 py-3 bg-bg-dark-card/60 border-b border-border-dark">
      <span className="text-xs text-text-on-dark-muted font-medium uppercase tracking-wider mr-2">
        Experiment
      </span>
      {EXPERIMENTS.map((exp) => (
        <button
          key={exp.id}
          onClick={() => onSelectExp(exp.id)}
          className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
            selectedExp === exp.id
              ? 'text-white shadow-md'
              : 'text-text-on-dark-muted hover:text-text-on-dark bg-transparent hover:bg-white/5'
          }`}
          style={
            selectedExp === exp.id
              ? { backgroundColor: exp.color }
              : undefined
          }
          title={exp.title}
        >
          {exp.label}
          <span className="hidden sm:inline"> — {exp.title}</span>
        </button>
      ))}
    </div>
  );
}
