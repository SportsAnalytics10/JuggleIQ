import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Upload,
  Play,
  Pause,
  Share2,
  Download,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { MetricCard } from "./MetricCard";
import { SoccerBallLoader } from "./SoccerBallLoader";
import { SessionReplayDashboard } from "./SessionReplayDashboard";
import { HeightConsistencyHeatmap } from "./HeightConsistencyHeatmap";
import { FootUsageTimeline } from "./FootUsageTimeline";
import { TouchRhythmGraph } from "./TouchRhythmGraph";
import { SkillEvolutionVisualizer } from "./SkillEvolutionVisualizer";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// ── Mock chart data ────────────────────────────────────────────────────
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

const processingSteps = [
  { label: "Uploading video", progress: 15 },
  { label: "Detecting ball movement", progress: 35 },
  { label: "Analyzing juggle patterns", progress: 55 },
  { label: "Calculating metrics", progress: 78 },
  { label: "Generating insights", progress: 100 },
];

// ── Player showcase data ───────────────────────────────────────────────
const playerCards = [
  { name: "Marcus Silva", subtitle: "Ball tracking + trajectory" },
  { name: "Emma Rodriguez", subtitle: "Ball tracking + trajectory" },
  { name: "James Chen", subtitle: "Ball tracking + trajectory" },
  { name: "Sofia Martins", subtitle: "Ball tracking + trajectory" },
  { name: "Alex Johnson", subtitle: "Ball tracking + trajectory" },
];

const duplicatedCards = [...playerCards, ...playerCards];
const cardWidth = 320;
const gapWidth = 20;
const totalOneSet = playerCards.length * (cardWidth + gapWidth);

// ── Component ──────────────────────────────────────────────────────────
export function HomePageV2() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [smoothProgress, setSmoothProgress] = useState(0);
  const animationRef = useRef<number>();
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  // Carousel scroll state
  const scrollRef = useRef<HTMLDivElement>(null);
  const isPausedRef = useRef(false);

  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        setIsProcessing(true);
        setCurrentStep(0);
        setSmoothProgress(0);
      }
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    []
  );

  // Step advancement while processing
  useEffect(() => {
    if (!isProcessing) return;

    const stepDuration = 1800;
    timerRef.current = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < processingSteps.length - 1) {
          return prev + 1;
        } else {
          clearInterval(timerRef.current);
          setTimeout(() => {
            setIsProcessing(false);
            setCurrentStep(0);
            setSmoothProgress(0);
          }, 1000);
          return prev;
        }
      });
    }, stepDuration);

    return () => clearInterval(timerRef.current);
  }, [isProcessing]);

  // Smooth progress interpolation
  useEffect(() => {
    if (!isProcessing) return;

    const targetProgress = processingSteps[currentStep].progress;

    const animate = () => {
      setSmoothProgress((prev) => {
        const diff = targetProgress - prev;
        if (Math.abs(diff) < 0.5) return targetProgress;
        return prev + diff * 0.04;
      });
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isProcessing, currentStep]);

  // Carousel infinite scroll
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    let scrollAnimationId: number;
    let position = 0;
    const speed = 0.5;

    const animateScroll = () => {
      if (!isPausedRef.current) {
        position += speed;
        if (position >= totalOneSet) {
          position = 0;
        }
        container.style.transform = `translateX(-${position}px)`;
      }
      scrollAnimationId = requestAnimationFrame(animateScroll);
    };

    scrollAnimationId = requestAnimationFrame(animateScroll);
    return () => cancelAnimationFrame(scrollAnimationId);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* ═══════════════════════════════════════════════════════════════
          HERO SECTION — "Master the art of juggling"
          ═══════════════════════════════════════════════════════════════ */}
      <section className="pt-12 pb-10 px-4">
        <div className="max-w-[800px] mx-auto text-center">
          <div
            className="text-[#2ECC71] text-[11px] uppercase tracking-[0.15em] mb-4"
            style={{ fontWeight: 600 }}
          >
            AI-POWERED JUGGLING COACH
          </div>
          <h1
            className="text-foreground text-[48px] leading-[1.1] tracking-[-0.03em] mb-5"
            style={{ fontWeight: 800 }}
          >
            Master the art of juggling.
          </h1>
          <p className="text-[17px] leading-[1.6] text-muted-foreground">
            Upload your session. Get instant feedback. Level up your touch.
          </p>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════
          PELÉ VIDEO BAR
          ═══════════════════════════════════════════════════════════════ */}
      <section className="relative mb-10">
        <div className="relative w-full h-[240px] overflow-hidden flex items-center justify-center">
          {/* Edge gradients */}
          <div
            className="absolute left-0 top-0 bottom-0 w-[200px] z-10 pointer-events-none"
            style={{
              background:
                "linear-gradient(to right, var(--color-background) 0%, transparent 100%)",
            }}
          />
          <div
            className="absolute right-0 top-0 bottom-0 w-[200px] z-10 pointer-events-none"
            style={{
              background:
                "linear-gradient(to left, var(--color-background) 0%, transparent 100%)",
            }}
          />

          {/* Video frame */}
          <div
            className="relative w-[800px] h-[240px] rounded-[12px] overflow-hidden"
            style={{ backgroundColor: "rgba(255,255,255,0.06)" }}
          >
            {/* Film grain */}
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.15'/%3E%3C/svg%3E")`,
                mixBlendMode: "overlay",
              }}
            />
            {/* Play icon */}
            <div
              className="absolute inset-0 flex items-center justify-center"
              style={{ filter: "grayscale(1)" }}
            >
              <div className="w-28 h-28 rounded-full border-4 border-white/20 flex items-center justify-center">
                <Play className="w-10 h-10 text-white/40 ml-1" />
              </div>
            </div>
          </div>
        </div>
        <div className="text-center mt-2.5">
          <p
            className="text-[12px] italic text-muted-foreground/40"
          >
            Pelé, 1958
          </p>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════
          PLAYER SHOWCASE CAROUSEL
          ═══════════════════════════════════════════════════════════════ */}
      <section className="mb-14 px-4 md:px-8">
        <div className="mb-5">
          <p
            className="text-[14px] uppercase tracking-[0.1em] text-muted-foreground/60"
          >
            See it in action
          </p>
        </div>

        <div
          className="overflow-hidden mx-auto"
          style={{ maxWidth: `${3 * cardWidth + 2 * gapWidth}px` }}
          onMouseEnter={() => {
            isPausedRef.current = true;
          }}
          onMouseLeave={() => {
            isPausedRef.current = false;
          }}
        >
          <div
            ref={scrollRef}
            className="flex"
            style={{ gap: `${gapWidth}px`, willChange: "transform" }}
          >
            {duplicatedCards.map((player, index) => (
              <div
                key={index}
                className="flex-shrink-0 rounded-[20px] border"
                style={{
                  width: `${cardWidth}px`,
                  height: "420px",
                  backgroundColor: "rgba(255,255,255,0.03)",
                  borderColor: "rgba(255,255,255,0.06)",
                }}
              >
                {/* Image area with ball detection overlay */}
                <div
                  className="relative w-full h-[340px] rounded-t-[20px] overflow-hidden"
                  style={{ backgroundColor: "rgba(255,255,255,0.05)" }}
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-[#1a1a1a] to-[#0d0d0d]" />
                  <svg
                    className="absolute inset-0 w-full h-full"
                    viewBox="0 0 320 340"
                  >
                    <path
                      d="M 80 280 Q 140 180, 200 120 T 280 60"
                      stroke="#2ECC71"
                      strokeWidth="2"
                      fill="none"
                      strokeDasharray="6 4"
                      opacity="0.6"
                    />
                    <circle cx="80" cy="280" r="8" fill="#2ECC71" opacity="0.3" />
                    <circle cx="140" cy="180" r="8" fill="#2ECC71" opacity="0.5" />
                    <circle cx="200" cy="120" r="10" fill="#2ECC71" opacity="0.7" />
                    <circle cx="260" cy="80" r="12" fill="#2ECC71" opacity="0.9" />
                    <circle cx="280" cy="60" r="16" fill="#2ECC71" />
                    <circle
                      cx="280"
                      cy="60"
                      r="24"
                      fill="none"
                      stroke="#2ECC71"
                      strokeWidth="2"
                      opacity="0.4"
                    />
                  </svg>
                  <div className="absolute bottom-8 left-8 right-8">
                    <div
                      className="h-32 rounded-lg"
                      style={{ backgroundColor: "rgba(255,255,255,0.02)" }}
                    />
                  </div>
                </div>
                <div className="p-5">
                  <h3
                    className="text-foreground text-[16px] mb-1"
                    style={{ fontWeight: 700 }}
                  >
                    {player.name}
                  </h3>
                  <p className="text-[12px] text-muted-foreground/50">
                    {player.subtitle}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════
          UPLOAD / PROCESSING SECTION
          ═══════════════════════════════════════════════════════════════ */}
      <div className="container mx-auto px-4 max-w-[1280px]">
        <section className="mb-12">
          <AnimatePresence mode="wait">
            {isProcessing ? (
              <motion.div
                key="processing"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="w-full rounded-[24px] border border-accent/20 flex flex-col items-center justify-center py-12 px-8"
                style={{
                  backgroundColor: "rgba(46,204,113,0.03)",
                  minHeight: 380,
                }}
              >
                <SoccerBallLoader progress={smoothProgress} size={200} />

                <div className="mt-6 text-center space-y-1.5">
                  <AnimatePresence mode="wait">
                    <motion.p
                      key={currentStep}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.25 }}
                      className="text-foreground"
                    >
                      {processingSteps[currentStep].label}…
                    </motion.p>
                  </AnimatePresence>
                  <p className="text-xs text-muted-foreground">
                    Our AI is analyzing your juggling technique
                  </p>
                </div>

                {/* Mini step indicators */}
                <div className="flex items-center gap-2 mt-6">
                  {processingSteps.map((_, idx) => {
                    const done = idx < currentStep;
                    const active = idx === currentStep;
                    return (
                      <div
                        key={idx}
                        className="transition-all duration-300"
                        style={{
                          width: active ? 24 : 8,
                          height: 8,
                          borderRadius: 4,
                          backgroundColor: done
                            ? "#2ECC71"
                            : active
                              ? "#2ECC71"
                              : "rgba(255,255,255,0.1)",
                        }}
                      />
                    );
                  })}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                onClick={handleUploadClick}
                className="w-full rounded-[24px] border flex flex-col items-center justify-center p-10 cursor-pointer group transition-all hover:border-[#2ECC71]/50"
                style={{
                  minHeight: 260,
                  backgroundColor: "rgba(255,255,255,0.02)",
                  borderWidth: "1px",
                  borderStyle: "dashed",
                  borderColor: "rgba(46,204,113,0.25)",
                  boxShadow: "0 0 80px rgba(46,204,113,0.05)",
                }}
              >
                <div className="mb-5 w-16 h-16 rounded-full bg-accent/10 flex items-center justify-center group-hover:bg-accent/20 transition-colors">
                  <Upload className="w-7 h-7 text-[#2ECC71]" />
                </div>
                <h2
                  className="text-foreground text-2xl mb-2 text-center"
                  style={{ fontWeight: 700 }}
                >
                  Upload your juggling video
                </h2>
                <p className="text-sm text-muted-foreground text-center max-w-md mb-6">
                  Drop a video or click to select — AltinhaAI will analyze your
                  technique instantly
                </p>
                <Button
                  className="rounded-xl px-8 py-3 text-white shadow-lg hover:shadow-xl transition-all hover:scale-105 pointer-events-none"
                  style={{ backgroundColor: "#2ECC71" }}
                >
                  Select Video
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            RESULTS SECTION
            ═══════════════════════════════════════════════════════════════ */}
        <section className="mb-12">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2
                className="text-3xl tracking-tight mb-1"
                style={{ fontWeight: 700 }}
              >
                Analysis Results
              </h2>
              <p className="text-muted-foreground">
                Your juggling session from {new Date().toLocaleDateString()}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" className="gap-2">
                <Share2 className="w-4 h-4" />
                Share
              </Button>
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="w-4 h-4" />
                Export
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main — Video + Charts */}
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
                    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-muted to-muted/50">
                      <div className="text-center space-y-4">
                        <div
                          className="w-20 h-20 rounded-full bg-background/80 backdrop-blur-sm flex items-center justify-center mx-auto cursor-pointer hover:bg-background transition-colors"
                          onClick={() => setIsPlaying(!isPlaying)}
                        >
                          {isPlaying ? (
                            <Pause className="w-8 h-8 text-accent" />
                          ) : (
                            <Play className="w-8 h-8 text-accent ml-1" />
                          )}
                        </div>
                        <div className="space-y-2">
                          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#2ECC71]/20 text-[#2ECC71] text-sm">
                            <div className="w-2 h-2 rounded-full bg-[#2ECC71]" />
                            Detected
                          </div>
                          <div className="inline-flex items-center gap-2 px-3 py-1.5 ml-2 rounded-full bg-[#F39C12]/20 text-[#F39C12] text-sm">
                            <div className="w-2 h-2 rounded-full bg-[#F39C12]" />
                            Predicted
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-background/50">
                      <div className="h-full w-1/3 bg-accent" />
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
                      <CardTitle className="text-lg">
                        Velocity Over Time
                      </CardTitle>
                      <CardDescription>
                        Ball velocity measured in m/s
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={velocityData}>
                          <defs>
                            <linearGradient
                              id="v2colorVelocity"
                              x1="0"
                              y1="0"
                              x2="0"
                              y2="1"
                            >
                              <stop
                                offset="5%"
                                stopColor="#3498DB"
                                stopOpacity={0.3}
                              />
                              <stop
                                offset="95%"
                                stopColor="#3498DB"
                                stopOpacity={0}
                              />
                            </linearGradient>
                          </defs>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.06)"
                          />
                          <XAxis
                            dataKey="time"
                            label={{
                              value: "Time (s)",
                              position: "insideBottom",
                              offset: -5,
                            }}
                          />
                          <YAxis
                            label={{
                              value: "Velocity (m/s)",
                              angle: -90,
                              position: "insideLeft",
                            }}
                          />
                          <Tooltip />
                          <Area
                            type="monotone"
                            dataKey="velocity"
                            stroke="#3498DB"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#v2colorVelocity)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="height">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">
                        Peak Height Over Time
                      </CardTitle>
                      <CardDescription>
                        Ball height measured in meters
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={heightData}>
                          <defs>
                            <linearGradient
                              id="v2colorHeight"
                              x1="0"
                              y1="0"
                              x2="0"
                              y2="1"
                            >
                              <stop
                                offset="5%"
                                stopColor="#2ECC71"
                                stopOpacity={0.3}
                              />
                              <stop
                                offset="95%"
                                stopColor="#2ECC71"
                                stopOpacity={0}
                              />
                            </linearGradient>
                          </defs>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="rgba(255,255,255,0.06)"
                          />
                          <XAxis
                            dataKey="time"
                            label={{
                              value: "Time (s)",
                              position: "insideBottom",
                              offset: -5,
                            }}
                          />
                          <YAxis
                            label={{
                              value: "Height (m)",
                              angle: -90,
                              position: "insideLeft",
                            }}
                          />
                          <Tooltip />
                          <Area
                            type="monotone"
                            dataKey="height"
                            stroke="#2ECC71"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#v2colorHeight)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>

            {/* Sidebar — Metrics */}
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
              <MetricCard value="1.4m" label="Peak Height" color="blue" />
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

              {/* Insights */}
              <Card className="shadow-sm bg-accent/5 border-accent/30">
                <CardHeader>
                  <CardTitle className="text-lg">AI Insights</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex gap-3">
                    <div className="text-2xl">🎯</div>
                    <div className="flex-1">
                      <p
                        className="text-sm mb-1"
                        style={{ fontWeight: 500 }}
                      >
                        Great consistency!
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Your juggle height is very stable, showing good control.
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="text-2xl">⚡</div>
                    <div className="flex-1">
                      <p
                        className="text-sm mb-1"
                        style={{ fontWeight: 500 }}
                      >
                        Watch your drift
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Try to keep the ball more centered to maintain better
                        balance.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* ═══════════════════════════════════════════════════════════════
            ADVANCED ANALYTICS SECTION
            ═══════════════════════════════════════════════════════════════ */}
        <section>
          <div className="mb-6">
            <h2
              className="text-3xl tracking-tight mb-1"
              style={{ fontWeight: 700 }}
            >
              Advanced Analytics
            </h2>
            <p className="text-muted-foreground">
              Deep dive into your performance with interactive visualizations
            </p>
          </div>

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
        </section>

        {/* Bottom spacer */}
        <div className="h-12" />
      </div>
    </div>
  );
}
