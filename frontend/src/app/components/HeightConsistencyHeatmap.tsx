import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Info } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";

interface HeightDataPoint {
  touch: number;
  height: number;
  consistency: number; // 0-100 score
}

interface HeightConsistencyHeatmapProps {
  data?: HeightDataPoint[];
  targetHeight?: number;
}

export function HeightConsistencyHeatmap({ 
  data = generateMockHeightData(),
  targetHeight = 1.3 
}: HeightConsistencyHeatmapProps) {
  const maxHeight = Math.max(...data.map(d => d.height));
  const minHeight = Math.min(...data.map(d => d.height));
  const avgHeight = data.reduce((sum, d) => sum + d.height, 0) / data.length;
  
  // Calculate consistency zones
  const heightRanges = [
    { min: 0, max: 0.8, label: "Too Low", color: "#E54D4D", bgColor: "bg-[#E54D4D]" },
    { min: 0.8, max: 1.1, label: "Below Target", color: "#F39C12", bgColor: "bg-[#F39C12]" },
    { min: 1.1, max: 1.5, label: "Optimal", color: "#2ECC71", bgColor: "bg-[#2ECC71]" },
    { min: 1.5, max: 1.8, label: "Above Target", color: "#F39C12", bgColor: "bg-[#F39C12]" },
    { min: 1.8, max: 3, label: "Too High", color: "#E54D4D", bgColor: "bg-[#E54D4D]" },
  ];

  const getColorForHeight = (height: number) => {
    const range = heightRanges.find(r => height >= r.min && height < r.max);
    return range?.color || "#95A5A6";
  };

  const getIntensity = (consistency: number) => {
    // Higher consistency = more opaque
    return Math.max(0.3, consistency / 100);
  };

  // Group data into time buckets (e.g., every 5 touches)
  const bucketSize = 5;
  const buckets = [];
  for (let i = 0; i < data.length; i += bucketSize) {
    const bucket = data.slice(i, i + bucketSize);
    const avgBucketHeight = bucket.reduce((sum, d) => sum + d.height, 0) / bucket.length;
    const avgBucketConsistency = bucket.reduce((sum, d) => sum + d.consistency, 0) / bucket.length;
    buckets.push({
      startTouch: i + 1,
      endTouch: Math.min(i + bucketSize, data.length),
      avgHeight: avgBucketHeight,
      avgConsistency: avgBucketConsistency,
      touches: bucket,
    });
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle>Height Consistency Heatmap</CardTitle>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="w-4 h-4 text-muted-foreground" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs text-xs">
                      Visualizes how consistent your ball height is across touches.
                      Darker colors indicate more consistent technique in that range.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <CardDescription>
              Track your ball control consistency across the session
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Average Height</div>
            <div className="text-xl font-mono font-bold">{avgHeight.toFixed(2)}m</div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Height Range</div>
            <div className="text-xl font-mono font-bold">
              {(maxHeight - minHeight).toFixed(2)}m
            </div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Consistency</div>
            <div className="text-xl font-mono font-bold text-[#2ECC71]">
              {Math.round(data.reduce((sum, d) => sum + d.consistency, 0) / data.length)}%
            </div>
          </div>
        </div>

        {/* Main heatmap */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Time progression →</span>
            <span className="text-xs text-muted-foreground">
              {data.length} total touches
            </span>
          </div>

          {/* Heatmap grid */}
          <div className="space-y-2">
            {heightRanges.slice().reverse().map((range, rangeIdx) => (
              <div key={rangeIdx} className="flex items-center gap-3">
                <div className="w-24 text-xs text-muted-foreground text-right">
                  {range.label}
                  <div className="text-[10px] text-muted-foreground/60">
                    {range.min.toFixed(1)}-{range.max.toFixed(1)}m
                  </div>
                </div>
                
                <div className="flex-1 flex gap-1">
                  {buckets.map((bucket, bucketIdx) => {
                    const touchesInRange = bucket.touches.filter(
                      t => t.height >= range.min && t.height < range.max
                    );
                    const count = touchesInRange.length;
                    const avgConsistency = count > 0
                      ? touchesInRange.reduce((sum, t) => sum + t.consistency, 0) / count
                      : 0;
                    const intensity = count > 0 ? getIntensity(avgConsistency) : 0.1;
                    
                    return (
                      <TooltipProvider key={bucketIdx}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div
                              className="flex-1 h-10 rounded transition-all hover:ring-2 hover:ring-foreground/20 cursor-pointer"
                              style={{
                                backgroundColor: range.color,
                                opacity: count > 0 ? intensity : 0.1,
                              }}
                            />
                          </TooltipTrigger>
                          <TooltipContent>
                            <div className="text-xs space-y-1">
                              <div className="font-semibold">
                                Touches {bucket.startTouch}-{bucket.endTouch}
                              </div>
                              <div>Height range: {range.label}</div>
                              <div>Touches in range: {count}/{bucket.touches.length}</div>
                              {count > 0 && (
                                <div>Consistency: {Math.round(avgConsistency)}%</div>
                              )}
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Target line indicator */}
          <div className="relative mt-4 pt-4 border-t border-border">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="w-3 h-3 border-2 border-dashed border-[#3498DB] rounded-sm" />
              <span>Target height: {targetHeight}m</span>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-4 p-3 rounded-lg bg-muted/30 border border-border">
          <span className="text-xs font-semibold text-muted-foreground">Legend:</span>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#2ECC71", opacity: 0.8 }} />
            <span>High consistency</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: "#2ECC71", opacity: 0.3 }} />
            <span>Low consistency</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-muted" />
            <span>No data</span>
          </div>
        </div>

        {/* Individual touch breakdown */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-muted-foreground">Touch-by-Touch Heights</h4>
          <div className="flex flex-wrap gap-1">
            {data.map((point, idx) => (
              <TooltipProvider key={idx}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div
                      className="w-6 h-6 rounded cursor-pointer transition-transform hover:scale-110"
                      style={{
                        backgroundColor: getColorForHeight(point.height),
                        opacity: getIntensity(point.consistency),
                      }}
                    />
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="text-xs space-y-1">
                      <div className="font-semibold">Touch #{point.touch}</div>
                      <div>Height: {point.height.toFixed(2)}m</div>
                      <div>Consistency: {Math.round(point.consistency)}%</div>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function generateMockHeightData(): HeightDataPoint[] {
  const data: HeightDataPoint[] = [];
  const touches = 30;
  let baseHeight = 1.2;
  
  for (let i = 0; i < touches; i++) {
    // Simulate gradual improvement and occasional mistakes
    baseHeight += (Math.random() - 0.5) * 0.3;
    baseHeight = Math.max(0.5, Math.min(2.2, baseHeight));
    
    // Consistency increases with practice
    const baseConsistency = 50 + (i / touches) * 30;
    const consistency = Math.min(95, baseConsistency + (Math.random() - 0.5) * 20);
    
    data.push({
      touch: i + 1,
      height: Number(baseHeight.toFixed(2)),
      consistency: Math.round(consistency),
    });
  }
  
  return data;
}
