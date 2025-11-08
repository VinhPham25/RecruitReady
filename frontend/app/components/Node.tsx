"use client";

import { useState } from "react";

interface NodeProps {
  title: string;
  content: string;
  defaultExpanded?: boolean;
}

export default function Node({ title, content, defaultExpanded = false }: NodeProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="node-container">
      <div
        className="node-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3>{title}</h3>
        <span className="expand-icon">{isExpanded ? "−" : "+"}</span>
      </div>

      {isExpanded && (
        <div className="node-content">
          <p>{content}</p>
        </div>
      )}
    </div>
  );
}
