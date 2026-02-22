import { useState } from "react";
import { useNavigate } from "react-router";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { MetricCard } from "./MetricCard";
import { ArrowLeft, Download, Share2, Play, Pause } from "lucide-react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

// Mock data for velocity over time
const velocityData = [
  { time: 0, velocity: 0 },
  { time: 1, velocity: 4.2 },
  { time: 2, velocity: 4.5 },
  { time: 3, velocity: 3.8 },
  { time: 4, velocity: 4.6 },
  { time: 5, velocity: 4.3 },
  { time: 6, velocity: 4.8 },
  { time: 7, velocity: 4.1 },
  { time: 8, velocity: 4.4 },
  { time: 9, velocity: 3.9 },
  { time: 10, velocity: 4.7 },
];

const heightData = [
  { time: 0, height: 0 },
  { time: 1, height: 1.2 },
  { time: 2, height: 1.4 },
  { time: 3, height: 1.1 },
  { time: 4, height: 1.5 },
  { time: 5, height: 1.3 },
  { time: 6, height: 1.6 },
  { time: 7, height: 1.2 },
  { time: 8, height: 1.4 },
  { time: 9, height: 1.1 },
  { time: 10, height: 1.5 },
];

export function ResultsScreen() {
  const navigate = useNavigate();
  const [isPlaying, setIsPlaying] = useState(false);

  return (
    <div className="container mx-auto py-8 px-4 max-w-[1280px]">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <Button variant="ghost" onClick={() => navigate("/")} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>
        <div className="flex gap-2">
          <Button variant="default" onClick={() => navigate("/analytics")} className="gap-2">
            Advanced Analytics
          </Button>
          <Button variant="outline" className="gap-2">
            <Share2 className="w-4 h-4" />
            Share
          </Button>
          <Button variant="outline" className="gap-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Title */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 tracking-tight">Analysis Results</h1>
        <p className="text-xl text-muted-foreground">
          Your juggling session from {new Date().toLocaleDateString()}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content - Video Player */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="shadow-sm">
            <CardHeader>
              <CardTitle>Annotated Video</CardTitle>
              <CardDescription>
                Watch your session with AI-detected juggles highlighted
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
                {/* Mock Video Player */}
                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                  <div className="text-center space-y-4">
                    <div className="w-20 h-20 rounded-full bg-background/80 backdrop-blur-sm flex items-center justify-center mx-auto cursor-pointer hover:bg-background transition-colors"
                      onClick={() => setIsPlaying(!isPlaying)}
                    >
                      {isPlaying ? (
                        <Pause className="w-8 h-8 text-accent" />
                      ) : (
                        <Play className="w-8 h-8 text-accent ml-1" />
                      )}
                    </div>
                    <div className="space-y-2">
                      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#2ECC71]/20 text-[#2ECC71] text-sm font-medium">
                        <div className="w-2 h-2 rounded-full bg-[#2ECC71]" />
                        Detected
                      </div>
                      <div className="inline-flex items-center gap-2 px-3 py-1.5 ml-2 rounded-full bg-[#F39C12]/20 text-[#F39C12] text-sm font-medium">
                        <div className="w-2 h-2 rounded-full bg-[#F39C12]" />
                        Predicted
                      </div>
                    </div>
                  </div>
                </div>
                {/* Progress Bar */}
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-background/50">
                  <div className="h-full w-1/3 bg-accent"></div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Charts */}
          <Tabs defaultValue="velocity" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="velocity">Ball Velocity</TabsTrigger>
              <TabsTrigger value="height">Peak Height</TabsTrigger>
            </TabsList>
            <TabsContent value="velocity">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Velocity Over Time</CardTitle>
                  <CardDescription>Ball velocity measured in m/s</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={velocityData}>
                      <defs>
                        <linearGradient id="colorVelocity" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3498DB" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#3498DB" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                      <XAxis dataKey="time" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Velocity (m/s)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Area type="monotone" dataKey="velocity" stroke="#3498DB" strokeWidth={2} fillOpacity={1} fill="url(#colorVelocity)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="height">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Peak Height Over Time</CardTitle>
                  <CardDescription>Ball height measured in meters</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={heightData}>
                      <defs>
                        <linearGradient id="colorHeight" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#2ECC71" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#2ECC71" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                      <XAxis dataKey="time" label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Height (m)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Area type="monotone" dataKey="height" stroke="#2ECC71" strokeWidth={2} fillOpacity={1} fill="url(#colorHeight)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Sidebar - Metrics */}
        <div className="space-y-4">
          <MetricCard
            value="12"
            label="Juggles Detected"
            trend="up"
            color="green"
          />
          <MetricCard
            value="8"
            label="Current Streak"
            trend="up"
            color="green"
          />
          <MetricCard
            value="1.4m"
            label="Peak Height"
            color="blue"
          />
          <MetricCard
            value="4.5"
            label="Avg Velocity (m/s)"
            color="blue"
          />
          <MetricCard
            value="0.15m"
            label="Lateral Drift"
            trend="down"
            color="yellow"
          />
          <MetricCard
            value="92%"
            label="Accuracy Score"
            trend="up"
            color="green"
          />

          {/* Insights Card */}
          <Card className="shadow-sm bg-accent/5 border-accent/30">
            <CardHeader>
              <CardTitle className="text-lg">AI Insights</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex gap-3">
                <div className="text-2xl">🎯</div>
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Great consistency!</p>
                  <p className="text-xs text-muted-foreground">
                    Your juggle height is very stable, showing good control.
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="text-2xl">⚡</div>
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Watch your drift</p>
                  <p className="text-xs text-muted-foreground">
                    Try to keep the ball more centered to maintain better balance.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="mt-8 flex justify-center">
        <Button size="lg" onClick={() => navigate("/")} className="gap-2">
          Analyze Another Session
        </Button>
      </div>
    </div>
  );
}