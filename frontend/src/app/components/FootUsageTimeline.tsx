import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

interface FootEvent {
  timestamp: number;
  foot: "left" | "right" | "other";
  successful: boolean;
}

interface FootUsageTimelineProps {
  events?: FootEvent[];
  duration?: number;
}

export function FootUsageTimeline({ 
  events = generateMockFootEvents(),
  duration = 30 
}: FootUsageTimelineProps) {
  // Calculate statistics
  const leftFootEvents = events.filter(e => e.foot === "left");
  const rightFootEvents = events.filter(e => e.foot === "right");
  const otherEvents = events.filter(e => e.foot === "other");
  
  const leftPercentage = (leftFootEvents.length / events.length) * 100;
  const rightPercentage = (rightFootEvents.length / events.length) * 100;
  const otherPercentage = (otherEvents.length / events.length) * 100;
  
  const leftSuccessRate = leftFootEvents.length > 0
    ? (leftFootEvents.filter(e => e.successful).length / leftFootEvents.length) * 100
    : 0;
  const rightSuccessRate = rightFootEvents.length > 0
    ? (rightFootEvents.filter(e => e.successful).length / rightFootEvents.length) * 100
    : 0;

  // Calculate alternation patterns
  const alternations = [];
  for (let i = 1; i < events.length; i++) {
    if (events[i].foot !== events[i - 1].foot && 
        events[i].foot !== "other" && 
        events[i - 1].foot !== "other") {
      alternations.push(i);
    }
  }
  const alternationRate = (alternations.length / (events.length - 1)) * 100;

  // Group events into time segments
  const segmentDuration = 5; // 5 seconds per segment
  const segments = Math.ceil(duration / segmentDuration);
  const segmentData = Array.from({ length: segments }, (_, idx) => {
    const startTime = idx * segmentDuration;
    const endTime = (idx + 1) * segmentDuration;
    const segmentEvents = events.filter(e => e.timestamp >= startTime && e.timestamp < endTime);
    
    return {
      segment: idx + 1,
      startTime,
      endTime,
      left: segmentEvents.filter(e => e.foot === "left").length,
      right: segmentEvents.filter(e => e.foot === "right").length,
      other: segmentEvents.filter(e => e.foot === "other").length,
      total: segmentEvents.length,
    };
  });

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Foot Usage Timeline</CardTitle>
        <CardDescription>
          Analyze which foot you use and track alternation patterns
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 rounded-lg bg-[#3498DB]/10 border border-[#3498DB]/30">
            <div className="text-xs text-muted-foreground mb-1">Right Foot</div>
            <div className="text-2xl font-mono font-bold text-[#3498DB]">
              {Math.round(rightPercentage)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {rightFootEvents.length} touches
            </div>
          </div>
          <div className="p-3 rounded-lg bg-[#2ECC71]/10 border border-[#2ECC71]/30">
            <div className="text-xs text-muted-foreground mb-1">Left Foot</div>
            <div className="text-2xl font-mono font-bold text-[#2ECC71]">
              {Math.round(leftPercentage)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {leftFootEvents.length} touches
            </div>
          </div>
          <div className="p-3 rounded-lg bg-[#95A5A6]/10 border border-[#95A5A6]/30">
            <div className="text-xs text-muted-foreground mb-1">Other</div>
            <div className="text-2xl font-mono font-bold text-[#95A5A6]">
              {Math.round(otherPercentage)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {otherEvents.length} touches
            </div>
          </div>
          <div className="p-3 rounded-lg bg-[#F39C12]/10 border border-[#F39C12]/30">
            <div className="text-xs text-muted-foreground mb-1">Alternation</div>
            <div className="text-2xl font-mono font-bold text-[#F39C12]">
              {Math.round(alternationRate)}%
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {alternations.length} switches
            </div>
          </div>
        </div>

        <Tabs defaultValue="timeline" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="timeline">Timeline View</TabsTrigger>
            <TabsTrigger value="distribution">Distribution</TabsTrigger>
          </TabsList>

          <TabsContent value="timeline" className="space-y-4 mt-4">
            {/* Main Timeline */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Time progression</span>
                <span>{events.length} total touches</span>
              </div>
              
              <div className="relative h-16 bg-muted rounded-lg overflow-hidden">
                {events.map((event, idx) => {
                  const position = (event.timestamp / duration) * 100;
                  const color = event.foot === "left" 
                    ? "#2ECC71" 
                    : event.foot === "right" 
                    ? "#3498DB" 
                    : "#95A5A6";
                  
                  return (
                    <div
                      key={idx}
                      className="absolute top-0 bottom-0 w-1 hover:w-2 transition-all cursor-pointer group"
                      style={{ left: `${position}%`, backgroundColor: color }}
                      title={`${event.timestamp.toFixed(1)}s - ${event.foot} foot`}
                    >
                      {event.successful ? null : (
                        <div className="absolute -top-1 left-0 right-0 h-1 bg-[#E54D4D]" />
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Time markers */}
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0s</span>
                <span>{(duration / 2).toFixed(0)}s</span>
                <span>{duration}s</span>
              </div>
            </div>

            {/* Segment breakdown */}
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-muted-foreground">
                5-Second Segments
              </h4>
              <div className="space-y-2">
                {segmentData.map((segment) => (
                  <div key={segment.segment} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">
                        {segment.startTime}s - {segment.endTime}s
                      </span>
                      <span className="font-mono text-muted-foreground">
                        {segment.total} touches
                      </span>
                    </div>
                    <div className="flex h-6 rounded-md overflow-hidden">
                      {segment.right > 0 && (
                        <div
                          className="bg-[#3498DB] flex items-center justify-center text-xs font-semibold text-white"
                          style={{ width: `${(segment.right / segment.total) * 100}%` }}
                        >
                          {segment.right > 1 && segment.right}
                        </div>
                      )}
                      {segment.left > 0 && (
                        <div
                          className="bg-[#2ECC71] flex items-center justify-center text-xs font-semibold text-white"
                          style={{ width: `${(segment.left / segment.total) * 100}%` }}
                        >
                          {segment.left > 1 && segment.left}
                        </div>
                      )}
                      {segment.other > 0 && (
                        <div
                          className="bg-[#95A5A6] flex items-center justify-center text-xs font-semibold text-white"
                          style={{ width: `${(segment.other / segment.total) * 100}%` }}
                        >
                          {segment.other > 1 && segment.other}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="distribution" className="space-y-4 mt-4">
            {/* Foot comparison */}
            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Right Foot Success Rate</span>
                  <span className="text-sm font-mono font-bold text-[#3498DB]">
                    {Math.round(rightSuccessRate)}%
                  </span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#3498DB] transition-all"
                    style={{ width: `${rightSuccessRate}%` }}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Left Foot Success Rate</span>
                  <span className="text-sm font-mono font-bold text-[#2ECC71]">
                    {Math.round(leftSuccessRate)}%
                  </span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#2ECC71] transition-all"
                    style={{ width: `${leftSuccessRate}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Pattern analysis */}
            <div className="p-4 rounded-lg bg-muted/30 border border-border space-y-3">
              <h4 className="text-sm font-semibold">Pattern Analysis</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dominant foot:</span>
                  <span className="font-medium capitalize">
                    {rightPercentage > leftPercentage ? "Right" : "Left"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Balance score:</span>
                  <span className="font-medium">
                    {Math.round(100 - Math.abs(rightPercentage - leftPercentage))}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Longest streak:</span>
                  <span className="font-medium">
                    {getLongestStreak(events)} touches
                  </span>
                </div>
              </div>
            </div>

            {/* Event sequence visualization */}
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-muted-foreground">
                Touch Sequence
              </h4>
              <div className="flex flex-wrap gap-1">
                {events.map((event, idx) => (
                  <div
                    key={idx}
                    className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold text-white ${
                      event.foot === "left"
                        ? "bg-[#2ECC71]"
                        : event.foot === "right"
                        ? "bg-[#3498DB]"
                        : "bg-[#95A5A6]"
                    } ${!event.successful ? "ring-2 ring-[#E54D4D]" : ""}`}
                    title={`Touch ${idx + 1}: ${event.foot} foot ${event.successful ? "✓" : "✗"}`}
                  >
                    {event.foot === "left" ? "L" : event.foot === "right" ? "R" : "O"}
                  </div>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-4 p-3 rounded-lg bg-muted/30 border border-border">
          <span className="text-xs font-semibold text-muted-foreground">Legend:</span>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#3498DB]" />
            <span>Right foot</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#2ECC71]" />
            <span>Left foot</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#95A5A6]" />
            <span>Other (head, chest, thigh)</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-muted ring-2 ring-[#E54D4D]" />
            <span>Unsuccessful touch</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function generateMockFootEvents(): FootEvent[] {
  const events: FootEvent[] = [];
  const totalTouches = 25;
  let currentTime = 0;
  let lastFoot: "left" | "right" | "other" = "right";
  
  for (let i = 0; i < totalTouches; i++) {
    currentTime += 0.8 + Math.random() * 0.8;
    
    // Simulate alternating with occasional same-foot touches
    let foot: "left" | "right" | "other";
    if (Math.random() > 0.85) {
      foot = "other";
    } else if (Math.random() > 0.3) {
      foot = lastFoot === "left" ? "right" : "left";
    } else {
      foot = lastFoot;
    }
    
    lastFoot = foot;
    
    events.push({
      timestamp: Number(currentTime.toFixed(2)),
      foot,
      successful: Math.random() > 0.15, // 85% success rate
    });
  }
  
  return events;
}

function getLongestStreak(events: FootEvent[]): number {
  let maxStreak = 0;
  let currentStreak = 0;
  let lastFoot: string | null = null;
  
  for (const event of events) {
    if (event.foot === lastFoot && event.foot !== "other") {
      currentStreak++;
      maxStreak = Math.max(maxStreak, currentStreak);
    } else {
      currentStreak = 1;
      lastFoot = event.foot;
    }
  }
  
  return maxStreak;
}
