import { useNavigate } from "react-router";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { ArrowLeft } from "lucide-react";
import { SessionReplayDashboard } from "./SessionReplayDashboard";
import { HeightConsistencyHeatmap } from "./HeightConsistencyHeatmap";
import { FootUsageTimeline } from "./FootUsageTimeline";
import { TouchRhythmGraph } from "./TouchRhythmGraph";
import { SkillEvolutionVisualizer } from "./SkillEvolutionVisualizer";

export function AdvancedAnalyticsScreen() {
  const navigate = useNavigate();

  return (
    <div className="container mx-auto py-8 px-4 max-w-[1280px]">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <Button variant="ghost" onClick={() => navigate("/results")} className="gap-2">
          <ArrowLeft className="w-4 h-4" />
          Back to Results
        </Button>
      </div>

      {/* Title */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 tracking-tight">Advanced Analytics</h1>
        <p className="text-xl text-muted-foreground">
          Deep dive into your performance with interactive visualizations
        </p>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="replay" className="w-full">
        <TabsList className="grid w-full grid-cols-5 mb-6">
          <TabsTrigger value="replay">Session Replay</TabsTrigger>
          <TabsTrigger value="height">Height Analysis</TabsTrigger>
          <TabsTrigger value="footwork">Footwork</TabsTrigger>
          <TabsTrigger value="rhythm">Rhythm</TabsTrigger>
          <TabsTrigger value="evolution">Evolution</TabsTrigger>
        </TabsList>

        <TabsContent value="replay" className="space-y-6">
          <SessionReplayDashboard />
        </TabsContent>

        <TabsContent value="height" className="space-y-6">
          <HeightConsistencyHeatmap />
        </TabsContent>

        <TabsContent value="footwork" className="space-y-6">
          <FootUsageTimeline />
        </TabsContent>

        <TabsContent value="rhythm" className="space-y-6">
          <TouchRhythmGraph />
        </TabsContent>

        <TabsContent value="evolution" className="space-y-6">
          <SkillEvolutionVisualizer />
        </TabsContent>
      </Tabs>
    </div>
  );
}
