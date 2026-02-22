import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";
import { TrendingUp, Award, Target } from "lucide-react";

interface SessionSummary {
  date: string;
  juggles: number;
  avgHeight: number;
  avgVelocity: number;
  consistency: number;
  accuracy: number;
  duration: number; // minutes
}

interface SkillMetrics {
  control: number;
  power: number;
  consistency: number;
  footwork: number;
  technique: number;
}

interface SkillEvolutionVisualizerProps {
  sessions?: SessionSummary[];
  currentSkills?: SkillMetrics;
}

export function SkillEvolutionVisualizer({ 
  sessions = generateMockSessions(),
  currentSkills = generateCurrentSkills()
}: SkillEvolutionVisualizerProps) {
  // Calculate overall progress
  const firstSession = sessions[0];
  const lastSession = sessions[sessions.length - 1];
  
  const jugglesImprovement = lastSession.juggles - firstSession.juggles;
  const consistencyImprovement = lastSession.consistency - firstSession.consistency;
  const accuracyImprovement = lastSession.accuracy - firstSession.accuracy;
  
  const totalSessions = sessions.length;
  const totalJuggles = sessions.reduce((sum, s) => sum + s.juggles, 0);
  const totalDuration = sessions.reduce((sum, s) => sum + s.duration, 0);
  const avgSessionQuality = sessions.reduce((sum, s) => sum + s.accuracy, 0) / totalSessions;

  // Calculate skill level
  const overallSkill = Object.values(currentSkills).reduce((sum, val) => sum + val, 0) / 5;
  const skillLevel = 
    overallSkill >= 90 ? { label: "Elite", color: "#7C5AE6" } :
    overallSkill >= 75 ? { label: "Advanced", color: "#2ECC71" } :
    overallSkill >= 60 ? { label: "Intermediate", color: "#3498DB" } :
    overallSkill >= 40 ? { label: "Developing", color: "#F39C12" } :
    { label: "Beginner", color: "#95A5A6" };

  // Prepare radar chart data
  const radarData = [
    { skill: "Control", value: currentSkills.control, fullMark: 100 },
    { skill: "Power", value: currentSkills.power, fullMark: 100 },
    { skill: "Consistency", value: currentSkills.consistency, fullMark: 100 },
    { skill: "Footwork", value: currentSkills.footwork, fullMark: 100 },
    { skill: "Technique", value: currentSkills.technique, fullMark: 100 },
  ];

  // Calculate milestones
  const milestones = [
    { reached: totalJuggles >= 1000, label: "1000 Total Juggles", icon: "🎯" },
    { reached: totalSessions >= 10, label: "10 Training Sessions", icon: "🏆" },
    { reached: lastSession.juggles >= 50, label: "50 Juggles in One Session", icon: "⭐" },
    { reached: avgSessionQuality >= 85, label: "85% Avg Accuracy", icon: "💪" },
    { reached: totalDuration >= 60, label: "1 Hour Training Time", icon: "⏱️" },
  ];

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle>Skill Evolution Visualizer</CardTitle>
            <CardDescription>
              Track your improvement over time across all metrics
            </CardDescription>
          </div>
          <div className={`px-3 py-1.5 rounded-full text-sm font-bold`} style={{ backgroundColor: `${skillLevel.color}20`, color: skillLevel.color }}>
            {skillLevel.label}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-gradient-to-br from-[#2ECC71]/10 to-[#2ECC71]/5 border border-[#2ECC71]/30">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-[#2ECC71]" />
              <span className="text-xs text-muted-foreground">Improvement</span>
            </div>
            <div className="text-2xl font-mono font-bold text-[#2ECC71]">
              +{jugglesImprovement}
            </div>
            <div className="text-xs text-muted-foreground mt-1">juggles/session</div>
          </div>
          
          <div className="p-4 rounded-lg bg-gradient-to-br from-[#3498DB]/10 to-[#3498DB]/5 border border-[#3498DB]/30">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-[#3498DB]" />
              <span className="text-xs text-muted-foreground">Total Sessions</span>
            </div>
            <div className="text-2xl font-mono font-bold text-[#3498DB]">
              {totalSessions}
            </div>
            <div className="text-xs text-muted-foreground mt-1">{totalDuration} minutes</div>
          </div>
          
          <div className="p-4 rounded-lg bg-gradient-to-br from-[#F39C12]/10 to-[#F39C12]/5 border border-[#F39C12]/30">
            <div className="flex items-center gap-2 mb-2">
              <Award className="w-4 h-4 text-[#F39C12]" />
              <span className="text-xs text-muted-foreground">Total Juggles</span>
            </div>
            <div className="text-2xl font-mono font-bold text-[#F39C12]">
              {totalJuggles.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground mt-1">across all sessions</div>
          </div>
          
          <div className="p-4 rounded-lg bg-gradient-to-br from-[#7C5AE6]/10 to-[#7C5AE6]/5 border border-[#7C5AE6]/30">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-2xl">🎯</span>
            </div>
            <div className="text-2xl font-mono font-bold" style={{ color: skillLevel.color }}>
              {Math.round(overallSkill)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">skill rating</div>
          </div>
        </div>

        <Tabs defaultValue="progress" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="progress">Progress</TabsTrigger>
            <TabsTrigger value="skills">Skill Radar</TabsTrigger>
            <TabsTrigger value="milestones">Milestones</TabsTrigger>
          </TabsList>

          <TabsContent value="progress" className="space-y-4 mt-4">
            {/* Juggles over time */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground">
                Juggles Per Session
              </h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sessions}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis 
                      label={{ value: 'Juggles', angle: -90, position: 'insideLeft' }}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="juggles" 
                      stroke="#2ECC71" 
                      strokeWidth={3}
                      dot={{ fill: "#2ECC71", r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Multi-metric comparison */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground">
                Performance Metrics Over Time
              </h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={sessions}>
                    <defs>
                      <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#2ECC71" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#2ECC71" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="colorConsistency" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3498DB" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3498DB" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E5E5" />
                    <XAxis 
                      dataKey="date" 
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis 
                      label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#2ECC71" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorAccuracy)"
                      name="Accuracy"
                    />
                    <Area 
                      type="monotone" 
                      dataKey="consistency" 
                      stroke="#3498DB" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#colorConsistency)"
                      name="Consistency"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Improvement indicators */}
            <div className="grid grid-cols-3 gap-4">
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-xs text-muted-foreground mb-2">Juggles</div>
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-[#2ECC71]" />
                  <span className="text-lg font-mono font-bold text-[#2ECC71]">
                    +{Math.round((jugglesImprovement / firstSession.juggles) * 100)}%
                  </span>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-xs text-muted-foreground mb-2">Consistency</div>
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-[#3498DB]" />
                  <span className="text-lg font-mono font-bold text-[#3498DB]">
                    +{consistencyImprovement.toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50">
                <div className="text-xs text-muted-foreground mb-2">Accuracy</div>
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-[#2ECC71]" />
                  <span className="text-lg font-mono font-bold text-[#2ECC71]">
                    +{accuracyImprovement.toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="skills" className="space-y-4 mt-4">
            {/* Radar Chart */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground text-center">
                Current Skill Profile
              </h4>
              <div className="h-80 flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#E5E5E5" />
                    <PolarAngleAxis 
                      dataKey="skill" 
                      tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                    />
                    <PolarRadiusAxis 
                      angle={90} 
                      domain={[0, 100]}
                      tick={{ fontSize: 10 }}
                    />
                    <Radar 
                      name="Skills" 
                      dataKey="value" 
                      stroke="#2ECC71" 
                      fill="#2ECC71" 
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Individual skill breakdowns */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground">
                Skill Breakdown
              </h4>
              {Object.entries(currentSkills).map(([skill, value]) => {
                const color = 
                  value >= 80 ? "#2ECC71" :
                  value >= 60 ? "#3498DB" :
                  value >= 40 ? "#F39C12" :
                  "#E54D4D";
                
                return (
                  <div key={skill} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="capitalize font-medium">{skill}</span>
                      <span className="font-mono font-bold" style={{ color }}>
                        {value}/100
                      </span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full transition-all rounded-full"
                        style={{ 
                          width: `${value}%`,
                          backgroundColor: color
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Recommendations */}
            <div className="p-4 rounded-lg bg-accent/5 border border-accent/30 space-y-2">
              <h4 className="text-sm font-semibold flex items-center gap-2">
                <span className="text-2xl">💡</span>
                Training Recommendations
              </h4>
              <div className="space-y-2 text-sm text-muted-foreground">
                {currentSkills.control < 70 && (
                  <p>• Focus on ball control drills to improve touch sensitivity</p>
                )}
                {currentSkills.power < 70 && (
                  <p>• Work on generating more power in your touches</p>
                )}
                {currentSkills.consistency < 70 && (
                  <p>• Practice maintaining steady rhythm and height</p>
                )}
                {currentSkills.footwork < 70 && (
                  <p>• Alternate feet more frequently to improve footwork balance</p>
                )}
                {currentSkills.technique < 70 && (
                  <p>• Focus on proper form and technique fundamentals</p>
                )}
                {Object.values(currentSkills).every(v => v >= 70) && (
                  <p>• Excellent all-around skills! Challenge yourself with advanced drills</p>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="milestones" className="space-y-4 mt-4">
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-muted-foreground">
                Achievement Progress
              </h4>
              
              <div className="space-y-3">
                {milestones.map((milestone, idx) => (
                  <div
                    key={idx}
                    className={`flex items-center gap-4 p-4 rounded-lg border-2 transition-all ${
                      milestone.reached
                        ? "bg-[#2ECC71]/10 border-[#2ECC71]/30"
                        : "bg-muted/30 border-border"
                    }`}
                  >
                    <div className={`text-3xl ${milestone.reached ? "scale-110" : "grayscale opacity-50"}`}>
                      {milestone.icon}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium">{milestone.label}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {milestone.reached ? "Completed!" : "Keep training to unlock"}
                      </div>
                    </div>
                    {milestone.reached && (
                      <div className="text-[#2ECC71] font-bold text-2xl">✓</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Next milestone progress */}
            <div className="p-4 rounded-lg bg-gradient-to-br from-[#7C5AE6]/10 to-[#7C5AE6]/5 border border-[#7C5AE6]/30">
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <span>🎯</span>
                Next Milestone
              </h4>
              <div className="space-y-2">
                {!milestones.find(m => !m.reached) ? (
                  <p className="text-sm text-muted-foreground">
                    🎉 All milestones completed! You're a juggling master!
                  </p>
                ) : (
                  <>
                    <p className="text-sm font-medium">
                      {milestones.find(m => !m.reached)?.label}
                    </p>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#7C5AE6] transition-all rounded-full"
                        style={{ width: "65%" }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">65% complete</p>
                  </>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function generateMockSessions(): SessionSummary[] {
  const sessions: SessionSummary[] = [];
  const dates = 10;
  let baseJuggles = 15;
  let baseConsistency = 60;
  let baseAccuracy = 70;
  
  for (let i = 0; i < dates; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (dates - i));
    
    // Simulate improvement with some variation
    baseJuggles += Math.random() * 5 + 1;
    baseConsistency = Math.min(95, baseConsistency + Math.random() * 3);
    baseAccuracy = Math.min(98, baseAccuracy + Math.random() * 2.5);
    
    sessions.push({
      date: `${date.getMonth() + 1}/${date.getDate()}`,
      juggles: Math.round(baseJuggles),
      avgHeight: Number((1.2 + Math.random() * 0.3).toFixed(2)),
      avgVelocity: Number((4.0 + Math.random() * 1).toFixed(1)),
      consistency: Math.round(baseConsistency),
      accuracy: Math.round(baseAccuracy),
      duration: Math.round(5 + Math.random() * 3),
    });
  }
  
  return sessions;
}

function generateCurrentSkills(): SkillMetrics {
  return {
    control: Math.round(70 + Math.random() * 20),
    power: Math.round(65 + Math.random() * 25),
    consistency: Math.round(75 + Math.random() * 15),
    footwork: Math.round(60 + Math.random() * 30),
    technique: Math.round(72 + Math.random() * 18),
  };
}
