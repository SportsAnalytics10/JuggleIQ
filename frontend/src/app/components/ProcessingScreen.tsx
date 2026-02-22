import { useEffect, useState, useRef } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { Check } from "lucide-react";
import { SoccerBallLoader } from "./SoccerBallLoader";

const processingSteps = [
  { label: "Uploading video", icon: "📤", progress: 15 },
  { label: "Detecting ball movement", icon: "⚽", progress: 35 },
  { label: "Analyzing juggle patterns", icon: "🔍", progress: 55 },
  { label: "Calculating metrics", icon: "📊", progress: 78 },
  { label: "Generating insights", icon: "✨", progress: 100 },
];

export function ProcessingScreen() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [smoothProgress, setSmoothProgress] = useState(0);
  const animationRef = useRef<number>();

  // Step advancement
  useEffect(() => {
    const stepDuration = 1800;

    const timer = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < processingSteps.length - 1) {
          return prev + 1;
        } else {
          clearInterval(timer);
          setTimeout(() => navigate("/results"), 800);
          return prev;
        }
      });
    }, stepDuration);

    return () => clearInterval(timer);
  }, [navigate]);

  // Smooth progress interpolation
  useEffect(() => {
    const targetProgress = processingSteps[currentStep].progress;

    const animate = () => {
      setSmoothProgress((prev) => {
        const diff = targetProgress - prev;
        if (Math.abs(diff) < 0.5) return targetProgress;
        // Ease toward target
        const next = prev + diff * 0.04;
        return next;
      });
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [currentStep]);

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <div className="max-w-lg w-full flex flex-col items-center gap-8">
        {/* Soccer Ball Loader */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <SoccerBallLoader progress={smoothProgress} size={240} />
        </motion.div>

        {/* Status text */}
        <div className="text-center space-y-2">
          <AnimatePresence mode="wait">
            <motion.h2
              key={currentStep}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="text-xl text-foreground"
            >
              {processingSteps[currentStep].label}
            </motion.h2>
          </AnimatePresence>
          <p className="text-sm text-muted-foreground">
            Our AI is analyzing your juggling technique
          </p>
        </div>

        {/* Processing Steps */}
        <div className="w-full space-y-1">
          {processingSteps.map((step, index) => {
            const isComplete = index < currentStep;
            const isCurrent = index === currentStep;

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1, duration: 0.3 }}
                className={`flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all duration-300 ${
                  isCurrent
                    ? "bg-accent/8 border border-accent/20"
                    : isComplete
                      ? "opacity-60"
                      : "opacity-30"
                }`}
              >
                {/* Step indicator */}
                <div className="w-6 h-6 flex items-center justify-center">
                  {isComplete ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      className="w-5 h-5 rounded-full bg-accent flex items-center justify-center"
                    >
                      <Check className="w-3 h-3 text-white" />
                    </motion.div>
                  ) : isCurrent ? (
                    <div className="relative">
                      <div className="w-2.5 h-2.5 rounded-full bg-accent" />
                      <motion.div
                        className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-accent"
                        animate={{ scale: [1, 2, 1], opacity: [1, 0, 1] }}
                        transition={{
                          duration: 1.5,
                          repeat: Infinity,
                          ease: "easeInOut",
                        }}
                      />
                    </div>
                  ) : (
                    <div className="w-2 h-2 rounded-full bg-muted-foreground/30" />
                  )}
                </div>

                <span
                  className={`text-sm ${
                    isCurrent
                      ? "text-foreground"
                      : isComplete
                        ? "text-muted-foreground"
                        : "text-muted-foreground/50"
                  }`}
                >
                  {step.label}
                </span>

                {isComplete && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="ml-auto text-xs text-accent/70"
                    style={{ fontFamily: "var(--font-family-mono)" }}
                  >
                    DONE
                  </motion.span>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Footer hint */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-xs text-muted-foreground/60 text-center"
          style={{ fontFamily: "var(--font-family-mono)" }}
        >
          ESTIMATED TIME: 10–15 SECONDS
        </motion.p>
      </div>
    </div>
  );
}
