import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Play, Pause, SkipBack, SkipForward, Maximize2 } from "lucide-react";

interface SessionReplayData {
  timestamp: number;
  ballPosition: { x: number; y: number };
  footUsed: "left" | "right" | "other";
  velocity: number;
  height: number;
  detected: boolean;
}

interface SessionReplayDashboardProps {
  sessionData?: SessionReplayData[];
  duration?: number;
}

export function SessionReplayDashboard({ 
  sessionData = generateMockSessionData(),
  duration = 30 
}: SessionReplayDashboardProps) {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);

  const currentFrame = sessionData.find(
    (data) => Math.abs(data.timestamp - currentTime) < 0.1
  ) || sessionData[0];

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleSkipBackward = () => {
    setCurrentTime(Math.max(0, currentTime - 2));
  };

  const handleSkipForward = () => {
    setCurrentTime(Math.min(duration, currentTime + 2));
  };

  const handleTimeChange = (value: number[]) => {
    setCurrentTime(value[0]);
  };

  // Get events at current time
  const recentEvents = sessionData.filter(
    (data) => data.timestamp >= currentTime - 1 && data.timestamp <= currentTime
  );

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle>Interactive Session Replay</CardTitle>
            <CardDescription>
              Review your session frame-by-frame with detailed metrics
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm">
            <Maximize2 className="w-4 h-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Video/Court Visualization */}
        <div className="relative aspect-video bg-muted rounded-lg overflow-hidden border border-border">
          {/* Soccer field representation */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#2ECC71]/5 to-[#2ECC71]/10">
            <svg className="w-full h-full" viewBox="0 0 400 300">
              {/* Field lines */}
              <line x1="200" y1="0" x2="200" y2="300" stroke="#2ECC71" strokeWidth="1" opacity="0.2" strokeDasharray="5,5" />
              <line x1="0" y1="150" x2="400" y2="150" stroke="#2ECC71" strokeWidth="1" opacity="0.2" strokeDasharray="5,5" />
              
              {/* Ball trail */}
              {sessionData
                .filter((d) => d.timestamp <= currentTime)
                .slice(-10)
                .map((data, idx, arr) => (
                  <circle
                    key={idx}
                    cx={data.ballPosition.x * 400}
                    cy={data.ballPosition.y * 300}
                    r={2}
                    fill="#2ECC71"
                    opacity={0.3 + (idx / arr.length) * 0.7}
                  />
                ))}
              
              {/* Current ball position */}
              {currentFrame && (
                <>
                  <circle
                    cx={currentFrame.ballPosition.x * 400}
                    cy={currentFrame.ballPosition.y * 300}
                    r={8}
                    fill={currentFrame.detected ? "#2ECC71" : "#F39C12"}
                    opacity="0.9"
                  />
                  <circle
                    cx={currentFrame.ballPosition.x * 400}
                    cy={currentFrame.ballPosition.y * 300}
                    r={12}
                    fill="none"
                    stroke={currentFrame.detected ? "#2ECC71" : "#F39C12"}
                    strokeWidth="2"
                    opacity="0.5"
                  >
                    <animate
                      attributeName="r"
                      from="12"
                      to="20"
                      dur="1s"
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      from="0.5"
                      to="0"
                      dur="1s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </>
              )}
            </svg>
          </div>

          {/* Overlay info */}
          <div className="absolute top-4 left-4 right-4 flex justify-between">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-background/90 backdrop-blur-sm text-sm font-mono font-semibold">
              <span className="text-muted-foreground">Time:</span>
              <span className="text-foreground">{currentTime.toFixed(1)}s</span>
            </div>
            {currentFrame && (
              <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-sm text-sm font-medium ${
                currentFrame.detected ? "bg-[#2ECC71]/20 text-[#2ECC71]" : "bg-[#F39C12]/20 text-[#F39C12]"
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  currentFrame.detected ? "bg-[#2ECC71]" : "bg-[#F39C12]"
                }`} />
                {currentFrame.detected ? "Detected" : "Predicted"}
              </div>
            )}
          </div>

          {/* Live metrics */}
          {currentFrame && (
            <div className="absolute bottom-4 left-4 right-4 grid grid-cols-3 gap-2">
              <div className="px-3 py-2 rounded-lg bg-background/90 backdrop-blur-sm">
                <div className="text-xs text-muted-foreground mb-1">Velocity</div>
                <div className="text-lg font-mono font-bold text-[#3498DB]">
                  {currentFrame.velocity.toFixed(1)} m/s
                </div>
              </div>
              <div className="px-3 py-2 rounded-lg bg-background/90 backdrop-blur-sm">
                <div className="text-xs text-muted-foreground mb-1">Height</div>
                <div className="text-lg font-mono font-bold text-[#2ECC71]">
                  {currentFrame.height.toFixed(2)} m
                </div>
              </div>
              <div className="px-3 py-2 rounded-lg bg-background/90 backdrop-blur-sm">
                <div className="text-xs text-muted-foreground mb-1">Foot</div>
                <div className="text-lg font-bold capitalize">
                  {currentFrame.footUsed}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Timeline with markers */}
        <div className="space-y-3">
          <div className="relative h-12 bg-muted rounded-lg overflow-hidden">
            {/* Event markers */}
            {sessionData.map((data, idx) => (
              <div
                key={idx}
                className="absolute top-0 bottom-0 w-1"
                style={{ left: `${(data.timestamp / duration) * 100}%` }}
              >
                <div
                  className={`w-full h-full ${
                    data.detected ? "bg-[#2ECC71]" : "bg-[#F39C12]"
                  } opacity-60 hover:opacity-100 transition-opacity cursor-pointer`}
                  title={`${data.timestamp.toFixed(1)}s - ${data.footUsed} foot`}
                  onClick={() => setCurrentTime(data.timestamp)}
                />
              </div>
            ))}
            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 w-0.5 bg-foreground z-10"
              style={{ left: `${(currentTime / duration) * 100}%` }}
            >
              <div className="absolute -top-1 -left-2 w-4 h-4 bg-foreground rounded-full shadow-lg" />
            </div>
          </div>

          {/* Slider */}
          <Slider
            value={[currentTime]}
            onValueChange={handleTimeChange}
            max={duration}
            step={0.1}
            className="w-full"
          />

          {/* Playback controls */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleSkipBackward}>
                <SkipBack className="w-4 h-4" />
              </Button>
              <Button variant="default" size="sm" onClick={handlePlayPause}>
                {isPlaying ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4 ml-0.5" />
                )}
              </Button>
              <Button variant="outline" size="sm" onClick={handleSkipForward}>
                <SkipForward className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">Speed:</span>
              <div className="flex gap-1">
                {[0.5, 1, 1.5, 2].map((speed) => (
                  <Button
                    key={speed}
                    variant={playbackSpeed === speed ? "default" : "ghost"}
                    size="sm"
                    className="w-12 h-8 text-xs"
                    onClick={() => setPlaybackSpeed(speed)}
                  >
                    {speed}×
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Recent events */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-muted-foreground">Recent Events</h4>
          <div className="space-y-1">
            {recentEvents.length === 0 ? (
              <p className="text-sm text-muted-foreground">No events in this timeframe</p>
            ) : (
              recentEvents.slice(-5).reverse().map((event, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/50 text-sm"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs text-muted-foreground">
                      {event.timestamp.toFixed(1)}s
                    </span>
                    <span className="capitalize">{event.footUsed} foot</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs">
                      {event.velocity.toFixed(1)} m/s
                    </span>
                    <div
                      className={`px-2 py-0.5 rounded text-xs font-medium ${
                        event.detected
                          ? "bg-[#2ECC71]/20 text-[#2ECC71]"
                          : "bg-[#F39C12]/20 text-[#F39C12]"
                      }`}
                    >
                      {event.detected ? "Detected" : "Predicted"}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper function to generate mock data
function generateMockSessionData(): SessionReplayData[] {
  const data: SessionReplayData[] = [];
  const touches = 15;
  
  for (let i = 0; i < touches; i++) {
    const timestamp = i * 2 + Math.random() * 0.5;
    data.push({
      timestamp,
      ballPosition: {
        x: 0.4 + Math.random() * 0.2 + Math.sin(i * 0.5) * 0.1,
        y: 0.4 + Math.random() * 0.2 + Math.cos(i * 0.5) * 0.1,
      },
      footUsed: Math.random() > 0.5 ? "right" : "left",
      velocity: 3.5 + Math.random() * 2,
      height: 1.0 + Math.random() * 0.8,
      detected: Math.random() > 0.2,
    });
  }
  
  return data.sort((a, b) => a.timestamp - b.timestamp);
}
