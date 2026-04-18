'use client';

import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

interface AccordionPanelProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export default function AccordionPanel({ title, children, defaultOpen = false }: AccordionPanelProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-border-dark rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3.5 bg-bg-dark-card text-text-on-dark text-sm font-semibold hover:bg-bg-dark-card/80 transition-colors"
      >
        <span>{title}</span>
        <ChevronRight
          size={16}
          className={`text-text-on-dark-muted transition-transform duration-200 ${open ? 'rotate-90' : ''}`}
        />
      </button>
      {open && (
        <div className="px-5 py-4 bg-bg-dark border-t border-border-dark">
          {children}
        </div>
      )}
    </div>
  );
}
