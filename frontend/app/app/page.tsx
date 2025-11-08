"use client";

import { useEffect, useState } from "react";
import Node from "../components/Node";

interface LiveData {
  message: string;
}

export default function HomePage() {
  const [data, setData] = useState<LiveData | null>(null);
  const videoUrl = "http://localhost:5000/video_feed"; // Change to your Flask URL

  useEffect(() => {
    const eventSource = new EventSource("http://localhost:5000/live-data");

    eventSource.onmessage = (event) => {
      setData(JSON.parse(event.data));
    };

    return () => eventSource.close();
  }, []);

  const zoomIn = () => {
    document.getElementById("video-stream")!.style.transform = "scale(1.2)";
  };

  const zoomOut = () => {
    document.getElementById("video-stream")!.style.transform = "scale(1)";
  };

  return (
    <div className="main-container">
      <div className="video-container">
        <h1>Live Video Stream</h1>

        <img
          id="video-stream"
          src={videoUrl}
          alt="Live Stream"
          style={{ transition: "transform 0.3s" }}
        />

        <div className="controls">
          <button onClick={zoomIn}>+</button>
          <button onClick={zoomOut}>-</button>
        </div>
      </div>

      <div className="data-container">
        <div className="nodes-wrapper">
          <Node
            title="Live Data"
            content={data ? data.message : "Waiting for server data..."}
          />
          <Node
            title="Speech Analysis"
            content="Speech pacing, clarity, and volume metrics will appear here."
          />
          <Node
            title="Visual Metrics"
            content="Eye contact, posture, and gesture analysis will appear here."
          />
          <Node
            title="Engagement Score"
            content="Overall engagement and performance metrics will appear here."
          />
        </div>
      </div>
    </div>
  );
}