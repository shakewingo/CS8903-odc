'use client';

import { useState } from 'react';
import { ChevronRight, HelpCircle } from 'lucide-react';

interface AccordionPanelProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  hint?: string;
}

export default function AccordionPanel({ title, children, defaultOpen = false, hint }: AccordionPanelProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-border-dark rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-bg-dark-card text-text-on-dark text-sm font-semibold hover:bg-bg-dark-card/80 transition-colors"
      >
        <span>{title}</span>
        <div className="flex items-center gap-2">
          {hint && (
            <div className="relative group/hint" onClick={(e) => e.stopPropagation()}>
              <HelpCircle size={14} className="text-text-on-dark-muted hover:text-accent-green-light cursor-help transition-colors" />
              <div className="absolute bottom-full right-0 mb-2 hidden group-hover/hint:block z-50">
                <div className="bg-bg-dark text-text-on-dark text-[11px] px-3 py-2 rounded-lg shadow-lg whitespace-normal w-max max-w-[300px] leading-relaxed border border-border-dark">
                  {hint}
                </div>
              </div>
            </div>
          )}
          <ChevronRight
            size={16}
            className={`text-text-on-dark-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
          />
        </div>
      </button>
      {open && (
        <div className="px-5 py-4 bg-bg-dark border-t border-border-dark">
          {children}
        </div>
      )}
    </div>
  );
}
