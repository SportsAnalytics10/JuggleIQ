import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface TouchRhythmData {
  interval: number; // seconds between touches
  timestamp: number;
  category: "fast" | "optimal" | "slow";
}

interface TouchRhythmGraphProps {
  data?: TouchRhythmData[];
}

export function TouchRhythmGraph({ data = generateMockRhythmData() }: TouchRhythmGraphProps) {
  // Calculate rhythm statistics
  const avgInterval = data.reduce((sum, d) => sum + d.interval, 0) / data.length;
  const minInterval = Math.min(...data.map(d => d.interval));
  const maxInterval = Math.max(...data.map(d => d.interval));
  
  const fastTouches = data.filter(d => d.category === "fast").length;
  const optimalTouches = data.filter(d => d.category === "optimal").length;
  const slowTouches = data.filter(d => d.category === "slow").length;
  
  // Calculate rhythm consistency (lower standard deviation = more consistent)
  const variance = data.reduce((sum, d) => sum + Math.pow(d.interval - avgInterval, 2), 0) / data.length;
  const stdDev = Math.sqrt(variance);
  const consistencyScore = Math.max(0, 100 - (stdDev * 50)); // Normalize to 0-100

  // Group data into time buckets for the chart
  const bucketSize = 5; // 5-touch buckets
  const bucketedData = [];
  for (let i = 0; i < data.length; i += bucketSize) {
    const bucket = data.slice(i, i + bucketSize);
    const avgBucketInterval = bucket.reduce((sum, d) => sum + d.interval, 0) / bucket.length;
    
    bucketedData.push({
      name: `${i + 1}-${Math.min(i + bucketSize, data.length)}`,
      interval: Number(avgBucketInterval.toFixed(2)),
      fast: bucket.filter(d => d.category === "fast").length,
      optimal: bucket.filter(d => d.category === "optimal").length,
      slow: bucket.filter(d => d.category === "slow").length,
    });
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle>Touch Rhythm Graph</CardTitle>
        <CardDescription>
          Analyze the timing and consistency of your juggling rhythm
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Avg Interval</div>
            <div className="text-2xl font-mono font-bold">{avgInterval.toFixed(2)}s</div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Range</div>
            <div className="text-2xl font-mono font-bold">
              {(maxInterval - minInterval).toFixed(2)}s
            </div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Consistency</div>
            <div className="text-2xl font-mono font-bold text-[#2ECC71]">
              {Math.round(consistencyScore)}%
            </div>
          </div>
          <div className="p-3 rounded-lg bg-muted/50">
            <div className="text-xs text-muted-foreground mb-1">Touches/Min</div>
            <div className="text-2xl font-mono font-bold text-[#3498DB]">
              {Math.round(60 / avgInterval)}
            </div>
          </div>
        </div>

        {/* Main Graph - Interval Distribution */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-muted-foreground">
            Interval Distribution by Touch Group
          </h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bucketedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                <XAxis 
                  dataKey="name" 
                  label={{ value: 'Touch Range', position: 'insideBottom', offset: -5 }}
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  label={{ value: 'Avg Interval (s)', angle: -90, position: 'insideLeft' }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="interval" fill="#3498DB" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rhythm Category Breakdown */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-muted-foreground">Rhythm Categories</h4>
          
          {/* Visual breakdown */}
          <div className="space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#2ECC71]" />
                  <span>Optimal (0.8-1.2s)</span>
                </div>
                <span className="font-mono font-bold text-[#2ECC71]">
                  {optimalTouches} touches ({Math.round((optimalTouches / data.length) * 100)}%)
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#2ECC71] transition-all"
                  style={{ width: `${(optimalTouches / data.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#F39C12]" />
                  <span>Fast (&lt;0.8s)</span>
                </div>
                <span className="font-mono font-bold text-[#F39C12]">
                  {fastTouches} touches ({Math.round((fastTouches / data.length) * 100)}%)
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#F39C12] transition-all"
                  style={{ width: `${(fastTouches / data.length) * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#E54D4D]" />
                  <span>Slow (&gt;1.2s)</span>
                </div>
                <span className="font-mono font-bold text-[#E54D4D]">
                  {slowTouches} touches ({Math.round((slowTouches / data.length) * 100)}%)
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#E54D4D] transition-all"
                  style={{ width: `${(slowTouches / data.length) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Timeline visualization */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-muted-foreground">
            Rhythm Timeline
          </h4>
          <div className="relative h-20 bg-muted rounded-lg overflow-hidden p-2">
            <div className="flex items-end justify-between h-full gap-0.5">
              {data.map((touch, idx) => {
                const heightPercentage = Math.min(100, (touch.interval / maxInterval) * 100);
                const color = 
                  touch.category === "optimal" ? "#2ECC71" :
                  touch.category === "fast" ? "#F39C12" :
                  "#E54D4D";
                
                return (
                  <div
                    key={idx}
                    className="flex-1 rounded-t transition-all hover:opacity-80 cursor-pointer"
                    style={{
                      height: `${heightPercentage}%`,
                      backgroundColor: color,
                      minHeight: "4px",
                    }}
                    title={`Touch ${idx + 1}: ${touch.interval.toFixed(2)}s interval`}
                  />
                );
              })}
            </div>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Start</span>
            <span>End</span>
          </div>
        </div>

        {/* Stacked category chart */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-muted-foreground">
            Category Distribution Over Time
          </h4>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bucketedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                <XAxis 
                  dataKey="name" 
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  label={{ value: 'Touch Count', angle: -90, position: 'insideLeft' }}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Bar dataKey="optimal" stackId="a" fill="#2ECC71" name="Optimal" radius={[0, 0, 0, 0]} />
                <Bar dataKey="fast" stackId="a" fill="#F39C12" name="Fast" radius={[0, 0, 0, 0]} />
                <Bar dataKey="slow" stackId="a" fill="#E54D4D" name="Slow" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Insights */}
        <div className="p-4 rounded-lg bg-accent/5 border border-accent/30 space-y-3">
          <h4 className="text-sm font-semibold flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Rhythm Analysis
          </h4>
          <div className="space-y-2 text-sm">
            {consistencyScore >= 80 ? (
              <p className="text-muted-foreground">
                ✓ Excellent rhythm consistency! Your timing between touches is very stable.
              </p>
            ) : consistencyScore >= 60 ? (
              <p className="text-muted-foreground">
                → Good rhythm, but there's room for improvement in maintaining consistent intervals.
              </p>
            ) : (
              <p className="text-muted-foreground">
                ⚠ Work on maintaining a more consistent rhythm between touches.
              </p>
            )}
            
            {optimalTouches / data.length >= 0.7 ? (
              <p className="text-muted-foreground">
                ✓ Great tempo control! Most of your touches fall within the optimal range.
              </p>
            ) : (
              <p className="text-muted-foreground">
                → Try to maintain a rhythm between 0.8-1.2 seconds for better control.
              </p>
            )}

            {fastTouches > slowTouches && fastTouches / data.length > 0.3 ? (
              <p className="text-muted-foreground">
                ⚡ You tend to rush your touches. Slow down slightly for better accuracy.
              </p>
            ) : slowTouches > fastTouches && slowTouches / data.length > 0.3 ? (
              <p className="text-muted-foreground">
                🐌 Your touches are a bit slow. Try to speed up slightly to maintain momentum.
              </p>
            ) : null}
          </div>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-4 p-3 rounded-lg bg-muted/30 border border-border">
          <span className="text-xs font-semibold text-muted-foreground">Target Rhythm:</span>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#2ECC71]" />
            <span>Optimal (0.8-1.2s) - Best control</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#F39C12]" />
            <span>Fast (&lt;0.8s) - May lose control</span>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <div className="w-4 h-4 rounded bg-[#E54D4D]" />
            <span>Slow (&gt;1.2s) - Risk of dropping</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function generateMockRhythmData(): TouchRhythmData[] {
  const data: TouchRhythmData[] = [];
  const touches = 30;
  let currentTime = 0;
  
  for (let i = 0; i < touches; i++) {
    // Generate somewhat realistic interval with variation
    const baseInterval = 1.0;
    const variation = (Math.random() - 0.5) * 0.6;
    const interval = Math.max(0.4, baseInterval + variation);
    
    currentTime += interval;
    
    let category: "fast" | "optimal" | "slow";
    if (interval < 0.8) {
      category = "fast";
    } else if (interval <= 1.2) {
      category = "optimal";
    } else {
      category = "slow";
    }
    
    data.push({
      interval: Number(interval.toFixed(2)),
      timestamp: Number(currentTime.toFixed(2)),
      category,
    });
  }
  
  return data;
}
